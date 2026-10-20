"""
finetune.py — Train a FreqAdapter on the UN 6-way corpus with symmetric multilingual
InfoNCE + identity preservation loss.

Dev/test split: eval_corpus[:dev_size] used for checkpoint selection;
eval_corpus[dev_size:] reported as the held-out test result (never influences training).

Usage (from spectral-probing/):
python -m direction2.finetune \
    --model mbert --target_layer 10 \
    --k_indices 1 2 3 4 --rank 16 --seed 42 \
    --manifest direction2/data/UN_Corpus/aligned/un_train15k_manifest.json \
    --eval_corpus direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json \
    --out_dir direction2/results/adapters
"""
import argparse
import hashlib
import json
import random
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from direction2.adapter import (FreqAdapter, filter_token_safe, guard_run_dir, pool,
                                run_prefix, run_suffix)

MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr':  'xlm-roberta-base',
    'labse': 'sentence-transformers/LaBSE',
}


def content_masks_for_batch(enc, device):
    """Per-sequence content masks: True where word_ids() is not None."""
    return [
        torch.tensor([w is not None for w in enc.word_ids(batch_index=i)], device=device)
        for i in range(enc['input_ids'].shape[0])
    ]


def encode_batch(model, tok, texts, adapter, target_layer, device, max_length, pooling='mean'):
    """Tokenise texts, run prefix+adapter+suffix, return (reps, H_adapted, H_orig, masks)."""
    enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
              max_length=max_length).to(device)
    masks = content_masks_for_batch(enc, device)
    H_orig = run_prefix(model, enc['input_ids'], enc['attention_mask'], target_layer)
    H_adapted = adapter(H_orig, masks)
    H_final = run_suffix(model, H_adapted, enc['attention_mask'], target_layer)
    reps = pool(H_final, masks, pooling)
    return reps, H_adapted, H_orig, masks


def infonece_loss(reps, B, L, tau):
    """Symmetric multilingual InfoNCE over all L*(L-1) directed language pairs.

    reps: (B*L, d) L2-normalised, block-major order:
          reps[b*L + l] = rep of block b, language l → view(B, L, d)[b, l].
    Uses directed permutations so each retrieval direction is an independent task.
    """
    reps = reps.view(B, L, -1)
    targets = torch.arange(B, device=reps.device)
    total = 0.0
    n_pairs = 0
    for la, lb in permutations(range(L), 2):
        S = reps[:, la] @ reps[:, lb].T  # (B, B)
        total = total + F.cross_entropy(S / tau, targets)
        n_pairs += 1
    return total / n_pairs


def eval_p1(model, tok, adapter, target_layer, eval_blocks, langs, device,
            batch_size, max_length, pooling='mean'):
    """Mean directed P@1 across all L*(L-1) language pairs on eval_blocks."""
    reps = {}
    with torch.no_grad():
        for lang in langs:
            lang_reps = []
            texts = [' '.join(b['texts'][lang]) for b in eval_blocks]
            for start in range(0, len(texts), batch_size):
                enc = tok(texts[start: start + batch_size], return_tensors='pt',
                          padding=True, truncation=True, max_length=max_length).to(device)
                masks = content_masks_for_batch(enc, device)
                H_orig = run_prefix(model, enc['input_ids'], enc['attention_mask'], target_layer)
                H_adapted = adapter(H_orig, masks)
                H_final = run_suffix(model, H_adapted, enc['attention_mask'], target_layer)
                lang_reps.append(pool(H_final, masks, pooling).cpu().numpy())
            reps[lang] = np.concatenate(lang_reps, 0)
            reps[lang] /= np.linalg.norm(reps[lang], axis=1, keepdims=True).clip(1e-9)

    n = len(eval_blocks)
    p1_scores = []
    for la, lb in permutations(langs, 2):
        S = reps[la] @ reps[lb].T
        p1 = float(np.mean(np.argmax(S, axis=1) == np.arange(n)))
        p1_scores.append(p1)
    return float(np.mean(p1_scores))


def parse_args():
    p = argparse.ArgumentParser(description='Frequency-domain adapter fine-tuning')
    p.add_argument('--model', default='mbert', choices=list(MODELS))
    p.add_argument('--target_layer', type=int, default=8,
                   help='hidden-state index; L8 = output after 8 transformer blocks')
    p.add_argument('--k_indices', nargs='*', type=int, default=[1, 2, 3, 4],
                   help='DCT mode indices to adapt')
    p.add_argument('--no_spectral', action='store_true',
                   help='hidden-state LoRA instead of DCT adapter')
    p.add_argument('--all_nondc', action='store_true',
                   help='adapt all non-DC modes (k=1..T-1) per sequence; T is sequence-length-dependent')
    p.add_argument('--gate', action='store_true',
                   help='element-wise gate g_k ∈ ℝ^d per mode instead of rank-r (U,V)')
    p.add_argument('--random_k_n', type=int, default=0,
                   help='if >0, sample this many k from [k_min, k_max] using seed')
    p.add_argument('--k_min', type=int, default=5)
    p.add_argument('--k_max', type=int, default=32)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--manifest', default='direction2/data/UN_Corpus/aligned/un_train15k_manifest.json')
    p.add_argument('--eval_corpus', default='direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json')
    p.add_argument('--dev_size', type=int, default=500,
                   help='first N eval blocks used for checkpoint selection; remainder are held-out test')
    p.add_argument('--max_steps', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_length', type=int, default=512,
                   help='tokeniser max length; blocks exceeding this are filtered before training')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lambda_id', type=float, default=0.01)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pooling', default='mean', choices=['mean', 'cls'],
                   help='pooling for InfoNCE reps: mean-pool content tokens or [CLS] token')
    p.add_argument('--out_dir', default='direction2/results/adapters')
    p.add_argument('--overwrite_run', action='store_true',
                   help='archive an existing run directory and start fresh')
    p.add_argument('--eval_every', type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    n_conds = sum([args.no_spectral, args.all_nondc, args.random_k_n > 0, args.gate])
    if n_conds > 1:
        raise SystemExit('at most one of --no_spectral, --all_nondc, --random_k_n, --gate may be set')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    # Resolve k condition
    if args.random_k_n > 0:
        k_pool = list(range(args.k_min, args.k_max + 1))
        k_indices = sorted(random.Random(args.seed).sample(k_pool, args.random_k_n))
        k_str = f'random{args.random_k_n}'
        print(f'random-{args.random_k_n} k_indices (seed={args.seed}): {k_indices}')
    elif args.all_nondc:
        k_indices = []  # determined per-sequence in adapt_sequence
        k_str = 'allnondc'
    else:
        k_indices = args.k_indices or []
        k_str = '-'.join(str(k) for k in k_indices) if (k_indices and not args.no_spectral) else 'nospectral'

    spectral = not args.no_spectral
    all_nondc = args.all_nondc
    pooling_sfx = '_cls' if args.pooling == 'cls' else ''
    if args.gate:
        run_name = f'L{args.target_layer}_gk{k_str}_s{args.seed}{pooling_sfx}'
    else:
        run_name = f'L{args.target_layer}_k{k_str}_r{args.rank}_s{args.seed}{pooling_sfx}'
    out_dir = Path(args.out_dir) / args.model / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest and eval corpus
    with open(args.manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.eval_corpus, encoding='utf-8') as f:
        eval_blocks = json.load(f)

    langs = list(manifest[0]['texts'].keys())
    L = len(langs)
    print(f'langs={langs}  n_train_raw={len(manifest)}')
    print(f'run: {run_name}  spectral={spectral}  all_nondc={all_nondc}  k_indices={k_indices}')

    # Load model
    model_name = MODELS[args.model]
    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = transformers.AutoModel.from_pretrained(model_name).to(device).eval()
    model.requires_grad_(False)
    d = model.config.hidden_size
    # Token-safe filter applied to train, dev, and test with the same tokenizer
    manifest = filter_token_safe(manifest, tok, langs, args.max_length)
    dev_blocks = filter_token_safe(eval_blocks[:args.dev_size], tok, langs, args.max_length)
    test_blocks = filter_token_safe(eval_blocks[args.dev_size:], tok, langs, args.max_length)
    print(f'n_train after filter: {len(manifest)}  eval dev={len(dev_blocks)}  test={len(test_blocks)}')
    if not manifest:
        raise SystemExit(f'training set empty after token-safe filter (max_length={args.max_length})')
    if not dev_blocks:
        raise SystemExit(f'dev set empty after token-safe filter (max_length={args.max_length})')
    if not test_blocks:
        raise SystemExit(f'test set empty after token-safe filter (max_length={args.max_length})')

    # Build adapter
    adapter = FreqAdapter(d, k_indices, rank=args.rank, spectral=spectral,
                          all_nondc=all_nondc, gate=args.gate).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)

    # Save config — identifies all hyperparameters and model-specific filtered data subsets.
    eval_corpus_hash = hashlib.md5(Path(args.eval_corpus).read_bytes()).hexdigest()[:8]

    def _blk_hash(b):
        return hashlib.md5(json.dumps(b['texts'], sort_keys=True).encode()).hexdigest()

    # *_filter_hash: fingerprints the model-specific filtered subset so cross-model comparisons
    # can verify they used the same blocks (mBERT and XLM-R may tokenise differently)
    # manifest blocks use 'document_id'; eval blocks use 'block_id'
    train_filter_hash = hashlib.md5(str(sorted((b['document_id'], _blk_hash(b)) for b in manifest)).encode()).hexdigest()[:8]
    dev_filter_hash = hashlib.md5(str(sorted(b['block_id'] for b in dev_blocks)).encode()).hexdigest()[:8]
    test_filter_hash = hashlib.md5(str(sorted(b['block_id'] for b in test_blocks)).encode()).hexdigest()[:8]
    config = {
        'model': args.model, 'hf_model': model_name, 'd': d,
        'target_layer': args.target_layer, 'k_indices': k_indices,
        'all_nondc': all_nondc, 'gate': args.gate, 'rank': args.rank,
        'spectral': spectral, 'seed': args.seed,
        'pooling': args.pooling, 'max_length': args.max_length, 'lr': args.lr,
        'temperature': args.temperature, 'lambda_id': args.lambda_id,
        'manifest': args.manifest, 'n_train': len(manifest), 'train_filter_hash': train_filter_hash,
        'eval_corpus': args.eval_corpus, 'eval_corpus_hash': eval_corpus_hash,
        'dev_size': args.dev_size, 'dev_filter_hash': dev_filter_hash,
        'n_test': len(test_blocks), 'test_filter_hash': test_filter_hash,
    }
    guard_run_dir(out_dir, config, args.overwrite_run)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    best_p1 = -1.0
    no_improve = 0
    step = 0
    epoch = 0

    while step < args.max_steps:
        epoch += 1
        rng = random.Random(args.seed + epoch)
        order = list(range(len(manifest)))
        rng.shuffle(order)
        shuffled = [manifest[i] for i in order]

        for start in range(0, len(shuffled), args.batch_size):
            if step >= args.max_steps:
                break
            batch_blocks = shuffled[start: start + args.batch_size]
            if not batch_blocks:
                continue
            B = len(batch_blocks)

            # Block-major text list: texts[b*L + l] = block b, language l
            texts = [' '.join(b['texts'][lang]) for b in batch_blocks for lang in langs]

            adapter.train()
            optimizer.zero_grad()

            reps, H_adapted, H_orig, masks = encode_batch(
                model, tok, texts, adapter, args.target_layer, device, args.max_length,
                pooling=args.pooling)

            reps_norm = F.normalize(reps, dim=-1)
            nce_loss = infonece_loss(reps_norm, B, L, args.temperature)

            # Identity preservation: delta is non-zero only at content positions
            delta = H_adapted - H_orig.detach()
            n_content = sum(int(m.sum()) for m in masks)
            id_loss = (delta ** 2).sum() / max(n_content * d, 1)

            loss = nce_loss + args.lambda_id * id_loss
            loss.backward()
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(f'  step {step}  loss={loss.item():.4f}  nce={nce_loss.item():.4f}'
                      f'  id={id_loss.item():.4f}  lr={optimizer.param_groups[0]["lr"]:.2e}')

            if step % args.eval_every == 0:
                adapter.eval()
                p1 = eval_p1(model, tok, adapter, args.target_layer,
                             dev_blocks, langs, device,
                             batch_size=32, max_length=args.max_length, pooling=args.pooling)
                print(f'  [dev] step {step}  mean P@1={p1:.4f}')
                scheduler.step(-p1)

                if p1 > best_p1:
                    best_p1 = p1
                    no_improve = 0
                    torch.save(adapter.state_dict(), out_dir / 'adapter.pt')
                    print(f'  [checkpoint] best dev P@1={best_p1:.4f} saved')
                else:
                    no_improve += 1
                    if no_improve >= 2:
                        print(f'  early stop after {no_improve} non-improving evals')
                        break

        if no_improve >= 2:
            break

    # Final eval on held-out test split (never used for checkpoint selection)
    adapter.load_state_dict(torch.load(out_dir / 'adapter.pt', map_location=device))
    adapter.eval()
    p1_test = eval_p1(model, tok, adapter, args.target_layer, test_blocks, langs, device,
                      batch_size=32, max_length=args.max_length, pooling=args.pooling)
    print(f'\ntest eval  mean P@1={p1_test:.4f}  ({out_dir})')

    results = {'best_dev_p1': best_p1, 'test_p1': p1_test,
               'steps': step, 'dev_size': args.dev_size, 'n_test': len(test_blocks)}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
