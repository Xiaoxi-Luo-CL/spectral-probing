"""
finetune_full.py — Fine-tune mBERT with LoRA + auxiliary spectral InfoNCE loss.

Direction 2 from step2b.md: the whole model is adapted (LoRA), not just a small
bottleneck adapter. An auxiliary InfoNCE loss on spectral representations at an
intermediate layer encourages cross-lingual alignment in the frequency domain.

Spectral modes for the auxiliary loss (--spectral_mode):
  none        - standard multilingual InfoNCE without a spectral auxiliary
  low_rms     - RMS-pooled DCT coefficients [Direction 2b]
  low_fk      - flattened signed DCT coefficients [Direction 2b]
  rigid_freq  - contrastive rigid same-k frequency distance [Direction 2c]
  freq_dtw    - contrastive direct frequency soft-DTW [Direction 2c]

Usage (from spectral-probing/):
python -m direction2.finetune_full \
    --model mbert --target_layer 8 --spectral_mode low_rms --seed 42 \
    --manifest direction2/data/UN_Corpus/aligned/un_train15k_manifest.json \
    --eval_corpus direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json
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

from direction2.adapter import (filter_frequency_safe, filter_token_safe,
                                guard_run_dir)
from direction2.alignment import (
    DISTANCE_SPECTRAL_MODES,
    DTW_COST_MODES,
    POINT_SPECTRAL_MODES,
    SPECTRAL_MODES,
    content_masks,
    frequency_distance_loss,
    infonece_loss,
    sentence_pool,
    rotating_language_pair,
    spectral_rep,
)

MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr':  'xlm-roberta-base',
}


def eval_p1(model, tok, eval_blocks, langs, device, batch_size, max_length, main_layer=None,
            pooling='mean'):
    """Mean directed P@1 on eval_blocks; uses main_layer hidden states or final layer."""
    need_hidden = main_layer is not None
    reps = {}
    with torch.no_grad():
        for lang in langs:
            parts = []
            texts = [' '.join(b['texts'][lang]) for b in eval_blocks]
            for s in range(0, len(texts), batch_size):
                enc = tok(texts[s:s + batch_size], return_tensors='pt', padding=True,
                          truncation=True, max_length=max_length).to(device)
                masks = content_masks(enc, device)
                out = model(**enc, output_hidden_states=need_hidden)
                H = out.hidden_states[main_layer] if need_hidden else out.last_hidden_state
                r = sentence_pool(H, masks, pooling).cpu().float().numpy()
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
    p.add_argument('--model', default='mbert', choices=list(MODELS))
    p.add_argument('--target_layer', type=int, default=8,
                   help='intermediate layer for spectral auxiliary loss')
    p.add_argument('--main_layer', type=int, default=None,
                   help='layer for main InfoNCE mean-pool (default: final layer)')
    p.add_argument('--k_indices', nargs='*', type=int, default=[1, 2, 3, 4])
    p.add_argument('--spectral_mode', default='low_rms', choices=SPECTRAL_MODES,
                   help='point or distance-based spectral auxiliary loss')
    p.add_argument('--lambda_spectral', type=float, default=0.1,
                   help='weight for spectral auxiliary loss')
    p.add_argument('--pooling', default='mean', choices=['mean', 'cls'],
                   help='main-loss sentence vector: mean of content tokens or [CLS]/<s>')
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
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--manifest',
                   default='direction2/data/UN_Corpus/aligned/un_train15k_manifest.json')
    p.add_argument('--eval_corpus',
                   default='direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json')
    p.add_argument('--dev_size', type=int, default=500)
    p.add_argument('--max_steps', type=int, default=5000)
    p.add_argument('--patience', type=int, default=2,
                   help='early-stop patience in eval periods; set high for fixed steps')
    p.add_argument('--batch_size', type=int, default=8,
                   help='parallel-sentence blocks per step; actual batch = B * L sequences')
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--save_last', action='store_true',
                   help='also keep the final-step checkpoint under lora_last/')
    p.add_argument('--overwrite_run', action='store_true',
                   help='replace a run directory holding a different experiment')
    p.add_argument('--block_filter_k', type=int, default=None,
                   help='pin the training population to sequences supporting this frequency')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', default='direction2/results/finetune_full')
    p.add_argument('--eval_every', type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    k_str = '-'.join(str(k) for k in args.k_indices)
    main_sfx = f'_m{args.main_layer}' if args.main_layer is not None else ''
    dtw_sfx = ''
    if args.spectral_mode in DISTANCE_SPECTRAL_MODES:
        dtw_sfx = (f'_g{args.dtw_gamma}_db{args.dtw_blocks}'
                   f'_br{args.dtw_band_radius}_c{args.dtw_cost}')
    # A pinned population is a different experiment; keep it off the unpinned path
    # so a matched baseline never overwrites an already completed run.
    bf_sfx = f'_bf{args.block_filter_k}' if args.block_filter_k is not None else ''
    # Dropping the main loss is a different objective, not a weighting tweak.
    lm_sfx = '' if args.lambda_main == 1.0 else f'_lm{args.lambda_main:g}'
    # A different auxiliary weight is a different experiment; 0.1 keeps historical names.
    ls_sfx = '' if args.lambda_spectral == 0.1 else f'_ls{args.lambda_spectral:g}'
    cls_sfx = '_cls' if args.pooling == 'cls' else ''
    run_name = (f'L{args.target_layer}_{args.spectral_mode}_k{k_str}'
                f'_r{args.lora_r}{main_sfx}{dtw_sfx}{bf_sfx}{lm_sfx}{ls_sfx}{cls_sfx}_s{args.seed}')
    out_dir = Path(args.out_dir) / args.model / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.eval_corpus, encoding='utf-8') as f:
        eval_blocks = json.load(f)

    model_name = MODELS[args.model]
    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Base model
    base = transformers.AutoModel.from_pretrained(model_name)
    d = base.config.hidden_size

    # Apply LoRA to attention projection layers
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=['query', 'value'],
        bias='none',
    )
    model = get_peft_model(base, lora_cfg)
    model.to(device)
    model.print_trainable_parameters()

    langs = list(manifest[0]['texts'].keys())
    L = len(langs)
    print(f'langs={langs}  n_train_raw={len(manifest)}')
    print(f'run: {run_name}  spectral_mode={args.spectral_mode}  k_indices={args.k_indices}')

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
    if not manifest or not dev_blocks or not test_blocks:
        raise SystemExit(f'empty split after token-safe filter (max_length={args.max_length})')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    eval_corpus_hash = hashlib.md5(Path(args.eval_corpus).read_bytes()).hexdigest()[:8]

    def _blk_hash(b):
        return hashlib.md5(json.dumps(b['texts'], sort_keys=True).encode()).hexdigest()

    train_filter_hash = hashlib.md5(str(sorted((b['document_id'], _blk_hash(b)) for b in manifest)).encode()).hexdigest()[:8]
    dev_filter_hash = hashlib.md5(str(sorted(b['block_id'] for b in dev_blocks)).encode()).hexdigest()[:8]
    test_filter_hash = hashlib.md5(str(sorted(b['block_id'] for b in test_blocks)).encode()).hexdigest()[:8]

    config = {
        'model': args.model, 'hf_model': model_name, 'd': d,
        'target_layer': args.target_layer, 'main_layer': args.main_layer,
        'k_indices': args.k_indices,
        'spectral_mode': args.spectral_mode, 'lambda_spectral': args.lambda_spectral,
        'lambda_main': args.lambda_main, 'pooling': args.pooling,
        'dtw_gamma': args.dtw_gamma,
        'dtw_temperature': args.dtw_temperature,
        'dtw_blocks': args.dtw_blocks,
        'dtw_band_radius': args.dtw_band_radius,
        'dtw_position_weight': args.dtw_position_weight,
        'dtw_cost': args.dtw_cost,
        'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'seed': args.seed, 'max_length': args.max_length, 'lr': args.lr,
        'temperature': args.temperature, 'batch_size': args.batch_size,
        'max_steps': args.max_steps, 'patience': args.patience,
        'eval_every': args.eval_every,
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

            # Block-major: texts[b*L + l] = block b, language l
            texts = [' '.join(b['texts'][lang]) for b in batch_blocks for lang in langs]

            model.train()
            optimizer.zero_grad()

            enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
                      max_length=args.max_length).to(device)
            masks = content_masks(enc, device)

            need_hidden = args.spectral_mode != 'none' or args.main_layer is not None
            out = model(**enc, output_hidden_states=need_hidden)

            # Main loss: InfoNCE on the pooled main_layer (or final layer)
            H_main = out.hidden_states[args.main_layer] if args.main_layer is not None \
                else out.last_hidden_state
            main_reps = sentence_pool(H_main, masks, args.pooling)
            main_norm = F.normalize(main_reps, dim=-1)
            main_loss = infonece_loss(main_norm, B, L, args.temperature)

            loss = args.lambda_main * main_loss
            # Auxiliary spectral loss at target_layer
            if args.spectral_mode != 'none':
                # hidden_states[i] = output after block i (0 = embeddings).
                # Use content_masks (excludes [CLS]/[SEP]) so the DCT is over content subword tokens only.
                H_mid = out.hidden_states[args.target_layer]   # (B*L, T, d)
                content_mask_t = torch.stack(masks)            # (B*L, T) bool
                if args.spectral_mode in POINT_SPECTRAL_MODES:
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
                      f'  spec={spec_loss.item():.4f}  lr={optimizer.param_groups[0]["lr"]:.2e}')

            if step % args.eval_every == 0:
                model.eval()
                p1 = eval_p1(model, tok, dev_blocks, langs,
                             device, batch_size=16, max_length=args.max_length,
                             main_layer=args.main_layer, pooling=args.pooling)
                print(f'  [dev] step {step}  mean P@1={p1:.4f}')

                if p1 > best_p1:
                    best_p1 = p1
                    no_improve = 0
                    model.save_pretrained(str(out_dir / 'lora'))
                    print(f'  [checkpoint] best dev P@1={best_p1:.4f} saved')
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        print(f'  early stop after {no_improve} non-improving evals')
                        break

        if no_improve >= args.patience:
            break

    # Ensure a checkpoint exists (eval may never have fired if max_steps < eval_every)
    if best_p1 < 0.0:
        model.save_pretrained(str(out_dir / 'lora'))
        print('  [checkpoint] no eval occurred; saving final model')

    if args.save_last:
        model.save_pretrained(str(out_dir / 'lora_last'))
        print(f'  [checkpoint] final step {step} saved to lora_last/')

    # Final test eval — reload best checkpoint
    base_reloaded = transformers.AutoModel.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_reloaded, str(out_dir / 'lora')).to(device)
    model.eval()
    p1_test = eval_p1(model, tok, test_blocks, langs,
                      device, batch_size=16, max_length=args.max_length,
                      main_layer=args.main_layer, pooling=args.pooling)
    print(f'\ntest eval  mean P@1={p1_test:.4f}  ({out_dir})')

    results = {'best_dev_p1': best_p1, 'test_p1': p1_test, 'steps': step,
               'dev_size': args.dev_size, 'n_test': len(test_blocks)}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
