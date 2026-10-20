"""
direction3/finetune.py — LoRA fine-tuning + spectral InfoNCE for decoder-only models.

Main InfoNCE uses last-token at --main_layer (required; final layer collapses for decoders).
Spectral auxiliary uses k=1..4 DCT at --target_layer.

Probe-derived defaults for qwen3-1.7b:
  --main_layer 16   (best frozen last-token retrieval layer)
  --target_layer 16 (same preregistered layer for the spectral auxiliary)

Usage (from spectral-probing/):
    python -m direction3.finetune --model qwen3-1.7b --spectral_mode low_fk --seed 42
"""
import argparse
import hashlib
import json
import random
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer

from direction2.adapter import (filter_frequency_safe, filter_token_safe,
                                guard_run_dir)
from direction2.alignment import (
    DISTANCE_SPECTRAL_MODES,
    DTW_COST_MODES,
    POINT_SPECTRAL_MODES,
    SPATIAL_MODES,
    SPECTRAL_MODES,
    content_masks,
    frequency_distance_loss,
    get_hidden_state,
    infonece_loss,
    last_pool,
    rotating_language_pair,
    spectral_rep,
)


MODELS = {
    'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
    'qwen3-4b':   'Qwen/Qwen3-4B',
}

# LoRA target modules for Qwen3. 'qv' is the PEFT default used by all historical
# runs; 'all' adapts every attention and MLP linear projection.
LORA_TARGET_SETS = {
    'qv': ['q_proj', 'v_proj'],
    'all': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
}

# (main_layer, target_layer)
# main_layer = best last-token layer; target_layer = spectral auxiliary layer
DEFAULT_LAYERS = {
    'qwen3-1.7b': (16, 16),
    'qwen3-4b':   (36, 30),
}


def eval_p1(model, tok, eval_blocks, langs, device, batch_size, main_layer, num_layers):
    """Mean directed P@1 on eval_blocks using last-token at main_layer."""
    need_hidden = main_layer != num_layers
    reps = {}
    with torch.no_grad():
        for lang in langs:
            parts = []
            texts = [' '.join(b['texts'][lang]) for b in eval_blocks]
            for s in range(0, len(texts), batch_size):
                enc = tok(texts[s:s + batch_size], return_tensors='pt',
                          padding=True).to(device)
                masks = content_masks(enc, device)
                out = model(**enc, output_hidden_states=need_hidden)
                H = get_hidden_state(out, main_layer, num_layers)
                r = last_pool(H, masks).cpu().float().numpy()
                parts.append(r)
            r_all = np.concatenate(parts, 0)
            r_all /= np.linalg.norm(r_all, axis=1, keepdims=True).clip(1e-9)
            reps[lang] = r_all
    n = len(eval_blocks)
    scores = []
    for la, lb in permutations(langs, 2):
        S = reps[la] @ reps[lb].T
        scores.append(float(np.mean(np.argmax(S, axis=1) == np.arange(n))))
    return float(np.mean(scores))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3-1.7b', choices=list(MODELS))
    p.add_argument('--target_layer', type=int, default=None,
                   help='layer for spectral auxiliary loss (default: probe-derived per model)')
    p.add_argument('--main_layer', type=int, default=None,
                   help='layer for main InfoNCE last-token (default: probe-derived per model)')
    p.add_argument('--k_indices', nargs='*', type=int, default=[1, 2, 3, 4])
    p.add_argument('--spectral_mode', default='none', choices=SPECTRAL_MODES)
    p.add_argument('--lambda_spectral', type=float, default=0.1)
    p.add_argument('--lambda_main', type=float, default=1.0,
                   help='weight on the main InfoNCE; 0 trains on the auxiliary alone')
    p.add_argument('--dtw_gamma', type=float, default=0.1)
    p.add_argument('--dtw_temperature', type=float, default=0.07)
    p.add_argument('--dtw_blocks', type=int, default=4)
    p.add_argument('--dtw_band_radius', type=int, default=-1,
                   help='Sakoe-Chiba radius; negative disables the constraint')
    p.add_argument('--dtw_position_weight', type=float, default=0.0)
    p.add_argument('--dtw_cost', choices=DTW_COST_MODES, default='equal_k')
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_targets', choices=list(LORA_TARGET_SETS), default='qv',
                   help="modules LoRA adapts: 'qv' (query/value) or 'all' linear projections")
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--manifest',
                   default='direction2/data/UN_Corpus/aligned/un_train15k_manifest.json')
    p.add_argument('--eval_corpus',
                   default='direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json')
    p.add_argument('--dev_size', type=int, default=500)
    p.add_argument('--max_steps', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--block_filter_k', type=int, default=None,
                   help='pin the training population to sequences supporting this frequency')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', default='direction3/results/finetune')
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--save_last', action='store_true',
                   help='also keep the final-step checkpoint under lora_last/')
    p.add_argument('--overwrite_run', action='store_true',
                   help='replace a run directory holding a different experiment')
    p.add_argument('--patience', type=int, default=2,
                   help='early-stop patience in eval periods; set high to disable early stopping')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    default_main, default_target = DEFAULT_LAYERS[args.model]
    main_layer = args.main_layer if args.main_layer is not None else default_main
    target_layer = args.target_layer if args.target_layer is not None else default_target

    k_str = '-'.join(str(k) for k in args.k_indices)
    dtw_sfx = ''
    if args.spectral_mode in DISTANCE_SPECTRAL_MODES:
        dtw_sfx = (f'_g{args.dtw_gamma}_db{args.dtw_blocks}'
                   f'_br{args.dtw_band_radius}_c{args.dtw_cost}')
    # A pinned population is a different experiment; keep it off the unpinned path
    # so a matched baseline never overwrites an already completed run.
    bf_sfx = f'_bf{args.block_filter_k}' if args.block_filter_k is not None else ''
    # Dropping the main loss is a different objective, not a weighting tweak.
    lm_sfx = '' if args.lambda_main == 1.0 else f'_lm{args.lambda_main:g}'
    # A different adapter is a different model; keep it off the q/v run paths.
    lt_sfx = '' if args.lora_targets == 'qv' else f'_l{args.lora_targets}'
    run_name = (f'L{target_layer}_{args.spectral_mode}_k{k_str}'
                f'_r{args.lora_r}_m{main_layer}{dtw_sfx}{bf_sfx}{lm_sfx}{lt_sfx}_s{args.seed}')
    out_dir = Path(args.out_dir) / args.model / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.eval_corpus, encoding='utf-8') as f:
        eval_blocks = json.load(f)

    model_name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    base.config.use_cache = False  # required for gradient checkpointing
    d = base.config.hidden_size
    num_layers = base.config.num_hidden_layers

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_SETS[args.lora_targets],
        bias='none',
    )
    model = get_peft_model(base, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.to(device)
    model.print_trainable_parameters()

    langs = list(manifest[0]['texts'].keys())
    L = len(langs)
    print(f'langs={langs}  n_train_raw={len(manifest)}')
    print(f'run: {run_name}  spectral_mode={args.spectral_mode}  k_indices={args.k_indices}')
    print(f'main_layer={main_layer}  target_layer={target_layer}')

    manifest = filter_token_safe(manifest, tok, langs, args.max_length)
    frequency_filter_k = args.block_filter_k
    if args.spectral_mode in DISTANCE_SPECTRAL_MODES:
        if not args.k_indices:
            raise SystemExit('distance-based spectral modes require --k_indices')
        frequency_filter_k = max(max(args.k_indices), frequency_filter_k or -1)
    if frequency_filter_k is not None:
        manifest = filter_frequency_safe(manifest, tok, langs, frequency_filter_k)
    dev_blocks = filter_token_safe(eval_blocks[:args.dev_size], tok, langs, args.max_length)
    test_blocks = filter_token_safe(eval_blocks[args.dev_size:], tok, langs, args.max_length)
    print(f'n_train={len(manifest)}  dev={len(dev_blocks)}  test={len(test_blocks)}')
    if not manifest or not dev_blocks:
        raise SystemExit(f'empty split after token-safe filter (max_length={args.max_length})')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    eval_corpus_hash = hashlib.md5(Path(args.eval_corpus).read_bytes()).hexdigest()[:8]
    train_filter_hash = hashlib.md5(
        str(sorted((b['document_id'], b['block_id']) for b in manifest)).encode()
    ).hexdigest()[:8]
    dev_ids = str(sorted(b['block_id'] for b in dev_blocks)).encode()
    test_ids = str(sorted(b['block_id'] for b in test_blocks)).encode()
    dev_filter_hash = hashlib.md5(dev_ids).hexdigest()[:8]
    test_filter_hash = hashlib.md5(test_ids).hexdigest()[:8]

    config = {
        'model': args.model, 'hf_model': model_name, 'd': d,
        'target_layer': target_layer, 'main_layer': main_layer,
        'k_indices': args.k_indices,
        'spectral_mode': args.spectral_mode, 'lambda_spectral': args.lambda_spectral,
        'lambda_main': args.lambda_main,
        'dtw_gamma': args.dtw_gamma,
        'dtw_temperature': args.dtw_temperature,
        'dtw_blocks': args.dtw_blocks,
        'dtw_band_radius': args.dtw_band_radius,
        'dtw_position_weight': args.dtw_position_weight,
        'dtw_cost': args.dtw_cost,
        'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout, 'lora_targets': LORA_TARGET_SETS[args.lora_targets],
        'seed': args.seed, 'max_length': args.max_length, 'lr': args.lr,
        'temperature': args.temperature, 'batch_size': args.batch_size,
        'max_steps': args.max_steps, 'eval_every': args.eval_every, 'patience': args.patience,
        'manifest': args.manifest, 'n_train': len(manifest),
        'block_filter_k': frequency_filter_k,
        'train_filter_hash': train_filter_hash,
        'eval_corpus': args.eval_corpus, 'eval_corpus_hash': eval_corpus_hash,
        'dev_size': args.dev_size, 'dev_filter_hash': dev_filter_hash,
        'n_test': len(test_blocks), 'test_filter_hash': test_filter_hash,
    }
    guard_run_dir(out_dir, config, args.overwrite_run)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    best_p1, no_improve, step, epoch = -1.0, 0, 0, 0

    while step < args.max_steps:
        epoch += 1
        rng = random.Random(args.seed + epoch)
        order = list(range(len(manifest)))
        rng.shuffle(order)
        shuffled = [manifest[i] for i in order]

        for start in range(0, len(shuffled), args.batch_size):
            if step >= args.max_steps:
                break
            batch_blocks = shuffled[start:start + args.batch_size]
            if not batch_blocks:
                continue
            B = len(batch_blocks)

            texts = [' '.join(b['texts'][lang]) for b in batch_blocks for lang in langs]

            model.train()
            optimizer.zero_grad()

            enc = tok(texts, return_tensors='pt', padding=True).to(device)
            masks = content_masks(enc, device)

            need_hidden = args.spectral_mode != 'none' or main_layer != num_layers
            out = model(**enc, output_hidden_states=need_hidden)

            # Main loss: InfoNCE on last-token of main_layer (or final layer)
            H_main = get_hidden_state(out, main_layer, num_layers)
            main_reps = last_pool(H_main, masks)
            main_norm = F.normalize(main_reps.float(), dim=-1)
            main_loss = infonece_loss(main_norm, B, L, args.temperature)

            loss = args.lambda_main * main_loss
            if args.spectral_mode != 'none':
                H_mid = out.hidden_states[target_layer]
                content_mask_t = torch.stack(masks)
                if args.spectral_mode in SPATIAL_MODES + POINT_SPECTRAL_MODES:
                    if args.spectral_mode in SPATIAL_MODES:
                        # Same pooling as the main loss, just at target_layer.
                        spec = last_pool(H_mid, masks).float()
                    else:
                        spec = spectral_rep(
                            H_mid, content_mask_t, args.k_indices,
                            args.spectral_mode,
                        )
                    spec_norm = F.normalize(spec, dim=-1)
                    spec_loss = infonece_loss(
                        spec_norm, B, L, args.temperature,
                    )
                else:
                    lang_pair = rotating_language_pair(step, L)
                    band_radius = (None if args.dtw_band_radius < 0 else args.dtw_band_radius)
                    spec_loss = frequency_distance_loss(
                        H_mid, content_mask_t, B, L, args.k_indices,
                        args.spectral_mode, args.dtw_temperature,
                        lang_pair, args.dtw_blocks, args.dtw_gamma,
                        band_radius, args.dtw_position_weight, args.dtw_cost,
                    )
                loss = loss + args.lambda_spectral * spec_loss
            else:
                spec_loss = loss.new_zeros(())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(f'  step {step}  loss={loss.item():.4f}  main={main_loss.item():.4f}'
                      f'  spec={spec_loss.item():.4f}')

            if step % args.eval_every == 0:
                model.eval()
                p1 = eval_p1(model, tok, dev_blocks, langs,
                             device, batch_size=16,
                             main_layer=main_layer, num_layers=num_layers)
                print(f'  [dev] step {step}  mean P@1={p1:.4f}')

                if p1 > best_p1:
                    best_p1 = p1
                    no_improve = 0
                    model.save_pretrained(str(out_dir / 'lora'))
                    print(f'  [checkpoint] best dev P@1={best_p1:.4f} saved')
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        print(f'  early stop at step {step}')
                        break

        if no_improve >= args.patience:
            break

    if best_p1 < 0.0:
        model.save_pretrained(str(out_dir / 'lora'))
        print('  [checkpoint] no eval occurred; saving final model')

    if args.save_last:
        model.save_pretrained(str(out_dir / 'lora_last'))
        print(f'  [checkpoint] final step {step} saved to lora_last/')

    # Final test eval — reload best checkpoint
    base_reloaded = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_reloaded, str(out_dir / 'lora')).to(device)
    model.eval()
    p1_test = eval_p1(model, tok, test_blocks, langs,
                      device, batch_size=16,
                      main_layer=main_layer, num_layers=num_layers)
    print(f'\ntest eval  mean P@1={p1_test:.4f}  ({out_dir})')

    results = {'best_dev_p1': best_p1, 'test_p1': p1_test, 'steps': step,
               'dev_size': args.dev_size, 'n_test': len(test_blocks)}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
