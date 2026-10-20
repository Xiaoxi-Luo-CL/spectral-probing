"""
eval_full.py — Evaluate a finetune_full LoRA checkpoint on a test corpus.

Loads a saved PeftModel, encodes all languages with final-layer mean-pool,
and reports bidirectional P@1 per language pair and their mean.

Usage (from spectral-probing/):
    python -m direction2.eval_full \
        --run_dir direction2/results/finetune_full/mbert/L8_none_k1-2-3-4_r16_s42 \
        --test_corpus direction2/data/NTREX/aligned/ntrex_25way_transfer_min100.json

    # Frozen baseline (no LoRA):
    python -m direction2.eval_full \
        --frozen --model mbert \
        --test_corpus direction2/data/NTREX/aligned/ntrex_25way_transfer_min100.json
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import transformers
from peft import PeftModel

from direction2.adapter import filter_token_safe
from direction2.alignment import content_masks, sentence_pool, spectral_coefficients

MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr':  'xlm-roberta-base',
}


def encode_lang(model, tok, texts, device, batch_size, max_length, main_layer=None,
                readout='pooled', target_layer=None, k_indices=None, pooling='mean'):
    """Encode all texts into L2-normalised retrieval vectors.

    ``pooled`` mean-pools main_layer (or the final layer) -- the model's ordinary
    sentence representation. ``spectral`` instead concatenates the DCT
    coefficients ``k_indices`` at ``target_layer``; cosine on that vector is the
    rigid natural-frequency distance, i.e. the geometry the spectral auxiliary
    trains on. This lets a checkpoint be scored in the space it was supervised
    in rather than only through the pooled readout.
    """
    spectral = readout == 'spectral'
    need_hidden = main_layer is not None or spectral
    parts = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], return_tensors='pt', padding=True,
                      truncation=True, max_length=max_length).to(device)
            masks = content_masks(enc, device)
            out = model(**enc, output_hidden_states=need_hidden)
            if spectral:
                H = out.hidden_states[target_layer]
                reps = spectral_coefficients(H, torch.stack(masks), k_indices).flatten(1)
            else:
                H = out.hidden_states[main_layer] if main_layer is not None else out.last_hidden_state
                reps = sentence_pool(H, masks, pooling)
            parts.append(reps.cpu().float().numpy())
    reps = np.concatenate(parts, 0)
    reps /= np.linalg.norm(reps, axis=1, keepdims=True).clip(1e-9)
    return reps


def retrieval_p1(reps_a, reps_b):
    """Bidirectional P@1 (diagonal positives)."""
    S = reps_a @ reps_b.T
    n = min(reps_a.shape[0], reps_b.shape[0])
    p1_ab = float(np.mean(np.argmax(S[:n], axis=1) == np.arange(n)))
    p1_ba = float(np.mean(np.argmax(S[:, :n], axis=0) == np.arange(n)))
    return (p1_ab + p1_ba) / 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', default=None,
                   help='path to finetune_full run dir (contains config.json + lora/)')
    p.add_argument('--frozen', action='store_true',
                   help='evaluate frozen baseline (no LoRA)')
    p.add_argument('--model', default=None,
                   help='model key (mbert/xlmr); read from config.json if not set')
    p.add_argument('--test_corpus', required=True)
    p.add_argument('--out_dir', default='direction2/results/eval_full')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--checkpoint', choices=('best', 'last'), default='best',
                   help='evaluate lora/ (best dev) or lora_last/ (final step)')
    p.add_argument('--main_layer', type=int, default=None,
                   help='override the pooled-readout layer; default comes from config.json '
                        '(None there means the final layer). Use it to layer-match a pooled '
                        'readout against a spectral one.')
    p.add_argument('--target_layer', type=int, default=8,
                   help='spectral readout layer when --frozen (else read from config.json)')
    p.add_argument('--k_indices', nargs='*', type=int, default=[1, 2],
                   help='spectral readout coefficients; overrides config.json when set. '
                        'Use a single k to dimension-match a spectral readout against pooled.')
    p.add_argument('--readout', choices=('pooled', 'spectral'), default='pooled',
                   help="'pooled' scores the ordinary sentence representation; "
                        "'spectral' scores concatenated DCT coefficients at the "
                        'auxiliary layer, the space the spectral loss trains on')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    if args.run_dir and not args.frozen:
        rdir = Path(args.run_dir)
        cfg = json.load(open(rdir / 'config.json'))
        model_key = args.model or cfg['model']
        main_layer = args.main_layer if args.main_layer is not None else cfg.get('main_layer')
        pooling = cfg.get('pooling') or 'mean'   # score a run with the vector it trained
        checkpoint_dir = 'lora_last' if args.checkpoint == 'last' else 'lora'
        checkpoint_path = rdir / checkpoint_dir
        if not checkpoint_path.exists():
            raise SystemExit(f'Missing {args.checkpoint} checkpoint: {checkpoint_path}')
        checkpoint_suffix = '_last' if args.checkpoint == 'last' else ''
        kspec = ('k' + '-'.join(map(str, args.k_indices))) if args.k_indices else ''
        readout_suffix = (f'_specread{kspec}' if args.readout == 'spectral'
                          else (f'_poolL{args.main_layer}' if args.main_layer is not None else ''))
        run_label = f'{model_key}_{rdir.name}{checkpoint_suffix}{readout_suffix}'
        target_layer = cfg.get('target_layer')
        k_indices = args.k_indices or cfg.get('k_indices') or [1, 2, 3, 4]
    else:
        if not args.model:
            raise SystemExit('--model required with --frozen')
        model_key = args.model
        cfg = {}
        main_layer = args.main_layer
        pooling = 'mean'
        target_layer = args.target_layer
        k_indices = args.k_indices
        if args.readout == 'spectral':
            kspec = ('k' + '-'.join(map(str, k_indices))) if args.k_indices else ''
            suffix = f'_specread{kspec}'
        else:
            suffix = f'_poolL{main_layer}' if main_layer is not None else ''
        run_label = f'{model_key}_frozen{suffix}'

    model_name = MODELS[model_key]
    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base = transformers.AutoModel.from_pretrained(model_name)

    if args.run_dir and not args.frozen:
        model = PeftModel.from_pretrained(base, str(checkpoint_path)).to(device)
    else:
        model = base.to(device)
    model.eval()

    with open(args.test_corpus, encoding='utf-8') as f:
        blocks = json.load(f)
    langs = list(blocks[0]['texts'].keys())

    n_raw = len(blocks)
    blocks = filter_token_safe(blocks, tok, langs, args.max_length)
    if len(blocks) < n_raw:
        print(f'token-safe filter: {n_raw} → {len(blocks)} blocks')

    print(f'corpus: {args.test_corpus}  n={len(blocks)}  langs={len(langs)}')
    print(f'evaluating: {run_label}  readout={args.readout}'
          + (f'  target=L{target_layer} k={k_indices}' if args.readout == 'spectral' else ''))

    reps = {}
    for lang in langs:
        print(f'  {lang} ...', end=' ', flush=True)
        texts = [' '.join(b['texts'][lang]) for b in blocks]
        reps[lang] = encode_lang(model, tok, texts, device, args.batch_size, args.max_length,
                                 main_layer=main_layer, readout=args.readout,
                                 target_layer=target_layer, k_indices=k_indices, pooling=pooling)
        print('done')

    rows, p1_all = [], []
    for la, lb in combinations(langs, 2):
        p1 = retrieval_p1(reps[la], reps[lb])
        rows.append({'pair': f'{la}-{lb}', 'lang_a': la, 'lang_b': lb, 'p1': round(p1, 4)})
        p1_all.append(p1)

    mean_p1 = float(np.mean(p1_all))
    print(f'\nmean P@1 = {mean_p1:.4f}  ({len(rows)} pairs)')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_stem = Path(args.test_corpus).stem
    out_path = out_dir / f'{run_label}_{corpus_stem}.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair', 'lang_a', 'lang_b', 'p1'])
        w.writeheader()
        w.writerows(rows)
        w.writerow({'pair': 'mean', 'lang_a': '', 'lang_b': '', 'p1': round(mean_p1, 4)})
    print(f'saved → {out_path}')


if __name__ == '__main__':
    main()
