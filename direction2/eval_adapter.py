"""
eval_adapter.py — Evaluate a trained FreqAdapter (or frozen baseline) on a test corpus.

Reports bidirectional mean-pool P@1 per language pair and their average.
Optionally reports Low-band translation advantage (RMS DCT vectors) as a DCT-domain
alignment diagnostic. Pairs are reported once (combinations), since retrieval_p1() is
already bidirectional.

Usage (from spectral-probing/):
    python -m direction2.eval_adapter \
        --adapter_dir direction2/results/adapters/mbert/L8_k1-2-3-4_r16_s42 \
        --test_corpus direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json \
        --out_dir direction2/results/eval_adapter

    # Frozen baseline (evaluates held-out test split, matching trained adapter):
    python -m direction2.eval_adapter \
        --frozen --model mbert --target_layer 8 --skip_dev 500 \
        --test_corpus direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json
"""
import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import transformers

from direction2.adapter import FreqAdapter, filter_token_safe, pool, run_prefix, run_suffix
from direction2.metrics import cos_matrix, translation_advantage
from models.encoders import PrismEncoder

MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr':  'xlm-roberta-base',
    'labse': 'sentence-transformers/LaBSE',
}


def content_masks_for_batch(enc, device):
    return [
        torch.tensor([w is not None for w in enc.word_ids(batch_index=i)], device=device)
        for i in range(enc['input_ids'].shape[0])
    ]


def encode_lang(model, tok, adapter, target_layer, texts, device,
                batch_size=32, max_length=512, store_adapted=False, pooling='mean'):
    """Encode all texts for one language; return (reps, adapted_content_list, block_lens).

    reps: (N, d) pooled float32, L2-normalised.
    adapted_content_list: list of (T_i, d) H_adapted numpy arrays; only populated when
        store_adapted=True (pass True only when --low_band is requested, to avoid GB of
        wasted allocation on large corpora like NTREX-25).
    block_lens: (N,) int32 — content token count per block.
    """
    reps_list = []
    adapted_list = []
    lens_list = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            enc = tok(batch, return_tensors='pt', padding=True,
                      truncation=True, max_length=max_length).to(device)
            masks = content_masks_for_batch(enc, device)

            H_orig = run_prefix(model, enc['input_ids'], enc['attention_mask'], target_layer)
            H_adapted = adapter(H_orig, masks)
            H_final = run_suffix(model, H_adapted, enc['attention_mask'], target_layer)
            r = pool(H_final, masks, pooling)
            reps_list.append(r.cpu().float().numpy())

            for i, mask in enumerate(masks):
                lens_list.append(int(mask.sum()))
                if store_adapted:
                    adapted_list.append(H_adapted[i, mask].cpu().float().numpy())

    reps = np.concatenate(reps_list, 0)
    reps /= np.linalg.norm(reps, axis=1, keepdims=True).clip(1e-9)
    return reps, adapted_list, np.array(lens_list, dtype=np.int32)


def low_band_alignment(adapted_a, adapted_b, k_lo=1, k_hi=5):
    """Translation advantage of Low-band DCT RMS vectors between parallel blocks.

    adapted_a, adapted_b: lists of (T_i, d) content-token numpy arrays.
    Returns translation_advantage(cos_matrix(A, B)) where A[i] is the RMS amplitude
    across DCT modes k_lo..k_hi-1 for block i. Comparable to Phase 1a's A_b metric.
    """
    vecs_a, vecs_b = [], []
    for Ha, Hb in zip(adapted_a, adapted_b):
        if Ha.shape[0] < k_hi or Hb.shape[0] < k_hi:
            continue
        Fa = PrismEncoder.dct(torch.from_numpy(Ha).float().T, norm='ortho').numpy()  # (d, T)
        Fb = PrismEncoder.dct(torch.from_numpy(Hb).float().T, norm='ortho').numpy()
        # RMS amplitude of Low band per dimension: (d,) per block
        va = np.sqrt(np.mean(Fa[:, k_lo:k_hi] ** 2, axis=1))
        vb = np.sqrt(np.mean(Fb[:, k_lo:k_hi] ** 2, axis=1))
        vecs_a.append(va)
        vecs_b.append(vb)
    if not vecs_a:
        return float('nan')
    return float(translation_advantage(cos_matrix(np.array(vecs_a), np.array(vecs_b))))


def retrieval_p1(reps_a, reps_b):
    """Bidirectional P@1 (diagonal positives assumed)."""
    S = reps_a @ reps_b.T
    n = min(reps_a.shape[0], reps_b.shape[0])
    p1_ab = float(np.mean(np.argmax(S[:n, :], axis=1) == np.arange(n)))
    p1_ba = float(np.mean(np.argmax(S[:, :n], axis=0) == np.arange(n)))
    return (p1_ab + p1_ba) / 2


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate FreqAdapter retrieval P@1')
    p.add_argument('--adapter_dir', default=None,
                   help='directory with config.json + adapter.pt (omit for frozen)')
    p.add_argument('--frozen', action='store_true',
                   help='evaluate frozen baseline (no adapter)')
    p.add_argument('--model', default=None,
                   help='model key (mbert/xlmr/labse); read from config.json if not set')
    p.add_argument('--target_layer', type=int, default=None,
                   help='hidden-state index; read from config.json if not set')
    p.add_argument('--test_corpus', required=True)
    p.add_argument('--out_dir', default='direction2/results/eval_adapter')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=None,
                   help='tokeniser max length; defaults to config.json value or 512')
    p.add_argument('--skip_dev', type=int, default=None,
                   help='skip first N corpus blocks (dev split); auto-read from config.json '
                        'when --adapter_dir is set; pass 0 to evaluate all blocks')
    p.add_argument('--low_band', action='store_true',
                   help='also compute Low-band translation advantage score')
    p.add_argument('--pooling', default=None, choices=['mean', 'cls'],
                   help='pooling for reps: mean-pool or [CLS]; defaults to config.json value or mean')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    # Resolve config: from adapter_dir or CLI args
    if args.adapter_dir and not args.frozen:
        adir = Path(args.adapter_dir)
        with open(adir / 'config.json') as f:
            cfg = json.load(f)
        model_key = args.model or cfg['model']
        target_layer = args.target_layer if args.target_layer is not None else cfg['target_layer']
        max_length = args.max_length if args.max_length is not None else cfg.get('max_length', 512)
        if args.skip_dev is not None:
            skip_n = max(args.skip_dev, 0)
        elif 'dev_size' in cfg and 'eval_corpus_hash' in cfg:
            # Auto-skip only when the test corpus byte-content matches the training eval corpus
            test_hash = hashlib.md5(Path(args.test_corpus).read_bytes()).hexdigest()[:8]
            if test_hash == cfg['eval_corpus_hash']:
                skip_n = cfg['dev_size']
            else:
                raise SystemExit(
                    f"test corpus hash '{test_hash}' differs from training eval corpus hash "
                    f"'{cfg['eval_corpus_hash']}'. "
                    f"Pass --skip_dev 0 to evaluate all blocks or --skip_dev N to skip the dev split.")
        else:
            skip_n = 0
    else:
        if not args.model or args.target_layer is None:
            raise SystemExit('--model and --target_layer required when --frozen is set')
        model_key = args.model
        target_layer = args.target_layer
        max_length = args.max_length if args.max_length is not None else 512
        if args.skip_dev is None:
            raise SystemExit(
                '--skip_dev is required for frozen evaluation. '
                'Pass --skip_dev 0 to evaluate all blocks or --skip_dev N to match an adapter dev split.')
        skip_n = max(args.skip_dev, 0)
        cfg = {}

    pooling = args.pooling or cfg.get('pooling', 'mean')
    model_name = MODELS[model_key]
    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = transformers.AutoModel.from_pretrained(model_name).to(device).eval()
    model.requires_grad_(False)
    d = model.config.hidden_size

    # Build adapter (identity if frozen)
    pooling_sfx = '_cls' if pooling == 'cls' else ''
    if args.frozen:
        adapter = FreqAdapter(d, [], rank=1, spectral=False).to(device)
        run_label = f'{model_key}_frozen_L{target_layer}{pooling_sfx}'
    else:
        adapter = FreqAdapter(d, cfg.get('k_indices', []),
                              rank=cfg.get('rank', 16),
                              spectral=cfg.get('spectral', True),
                              all_nondc=cfg.get('all_nondc', False),
                              gate=cfg.get('gate', False)).to(device)
        adapter.load_state_dict(torch.load(adir / 'adapter.pt', map_location=device))
        run_label = Path(args.adapter_dir).name
    adapter.eval()

    with open(args.test_corpus, encoding='utf-8') as f:
        test_blocks = json.load(f)
    langs = list(test_blocks[0]['texts'].keys())

    # Skip dev split to evaluate on held-out blocks only
    if skip_n > 0:
        test_blocks = test_blocks[skip_n:]
        print(f'skipped first {skip_n} dev blocks; evaluating {len(test_blocks)} held-out blocks')

    # Token-safe filter: drop blocks where any language exceeds max_length
    n_raw = len(test_blocks)
    test_blocks = filter_token_safe(test_blocks, tok, langs, max_length)
    if len(test_blocks) < n_raw:
        print(f'token-safe filter: dropped {n_raw - len(test_blocks)} blocks '
              f'(max_length={max_length}); evaluating {len(test_blocks)}')

    print(f'corpus: {args.test_corpus}  n={len(test_blocks)}  langs={langs}')
    print(f'evaluating: {run_label}  max_length={max_length}  pooling={pooling}')

    # Encode each language
    reps = {}
    adapted = {}
    for lang in langs:
        print(f'  encoding {lang} ...', end=' ', flush=True)
        texts = [' '.join(b['texts'][lang]) for b in test_blocks]
        r, a_list, _ = encode_lang(model, tok, adapter, target_layer,
                                   texts, device, args.batch_size, max_length,
                                   store_adapted=args.low_band, pooling=pooling)
        reps[lang] = r
        adapted[lang] = a_list
        print('done')

    # Compute P@1 for all unordered language pairs
    # retrieval_p1() is already bidirectional; combinations avoids duplicate rows
    rows = []
    p1_all = []
    for la, lb in combinations(langs, 2):
        p1 = retrieval_p1(reps[la], reps[lb])
        row = {'pair': f'{la}-{lb}', 'p1': round(p1, 4)}
        if args.low_band:
            row['low_band_adv'] = round(low_band_alignment(adapted[la], adapted[lb]), 4)
        rows.append(row)
        p1_all.append(p1)

    mean_p1 = float(np.mean(p1_all))
    print(f'\nmean P@1 = {mean_p1:.4f}')

    # Print table
    header = f"{'pair':<10} {'P@1':>7}"
    if args.low_band:
        header += f" {'Low-adv':>10}"
    print(header)
    for r in rows:
        line = f"{r['pair']:<10} {r['p1']:>7.4f}"
        if args.low_band:
            line += f" {r.get('low_band_adv', float('nan')):>10.4f}"
        print(line)
    print(f"{'mean':<10} {mean_p1:>7.4f}")

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_stem = Path(args.test_corpus).stem
    out_path = out_dir / f'{run_label}_{corpus_stem}.csv'
    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        w.writerow({'pair': 'mean', 'p1': round(mean_p1, 4)})
    print(f'\nsaved → {out_path}')


if __name__ == '__main__':
    main()
