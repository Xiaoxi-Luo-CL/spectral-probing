r"""
direction3/eval.py — Evaluate a direction3 LoRA checkpoint on a test corpus.

Reads main_layer from config.json; uses last-token at that layer.

Usage (from spectral-probing/):
    python -m direction3.eval \\
        --run_dir direction3/results/finetune/qwen3-1.7b/L24_low_fk_k1-2-3-4_r16_m16_s42 \\
        --test_corpus direction2/data/NTREX/aligned/ntrex_25way_transfer_min100.json

    # Frozen baseline:
    python -m direction3.eval \\
        --frozen --model qwen3-1.7b --main_layer 16 \\
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
from transformers import AutoModel, AutoTokenizer

from direction2.alignment import (content_masks, get_hidden_state, last_pool,
                                  spectral_coefficients)
from direction2.eval_full import retrieval_p1
from direction3.finetune import MODELS


def encode_lang(model, tok, texts, device, batch_size, main_layer, num_layers,
                readout='pooled', target_layer=None, k_indices=None):
    """Encode texts into L2-normalised retrieval vectors.

    ``pooled`` uses the last content token at main_layer -- the decoder's ordinary
    sentence representation. ``spectral`` instead concatenates the DCT
    coefficients ``k_indices`` at ``target_layer``, which is the space the
    spectral auxiliary trains on. No mean-pooling is used in either case.
    """
    spectral = readout == 'spectral'
    need_hidden = main_layer != num_layers or spectral
    parts = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], return_tensors='pt',
                      padding=True).to(device)
            masks = content_masks(enc, device)
            out = model(**enc, output_hidden_states=need_hidden)
            if spectral:
                H = out.hidden_states[target_layer]
                reps = spectral_coefficients(
                    H, torch.stack(masks), k_indices).flatten(1).cpu().float().numpy()
            else:
                H = get_hidden_state(out, main_layer, num_layers)
                reps = last_pool(H, masks).cpu().float().numpy()
            parts.append(reps)
    reps = np.concatenate(parts, 0)
    reps /= np.linalg.norm(reps, axis=1, keepdims=True).clip(1e-9)
    return reps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', default=None)
    p.add_argument('--frozen', action='store_true')
    p.add_argument('--model', default=None)
    p.add_argument('--main_layer', type=int, default=None,
                   help='required when --frozen; otherwise overrides config.json. Use it '
                        'to read the pooled vector at the spectral layer, so the two '
                        'readouts differ only in representation.')
    p.add_argument('--target_layer', type=int, default=None,
                   help='spectral readout layer; defaults to config.json (main_layer when frozen)')
    p.add_argument('--k_indices', type=int, nargs='*', default=None,
                   help='spectral readout coefficients; overrides config.json. Pass a single '
                        'k to dimension-match spectral (k*d dims) against pooled (d dims).')
    p.add_argument('--test_corpus', required=True)
    p.add_argument('--out_dir', default='direction3/results/eval')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--checkpoint', choices=('best', 'last'), default='best',
                   help='evaluate lora/ (best dev) or lora_last/ (final step)')
    p.add_argument('--readout', choices=('pooled', 'spectral'), default='pooled',
                   help="'pooled' scores last-token at main_layer; 'spectral' scores "
                        'concatenated DCT coefficients at the auxiliary layer')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    if args.frozen:
        if not args.model or args.main_layer is None:
            raise SystemExit('--model and --main_layer required with --frozen')
        model_key = args.model
        main_layer = args.main_layer
        cfg = {}
        target_layer = args.target_layer if args.target_layer is not None else main_layer
        k_indices = args.k_indices or [1, 2, 3, 4]
        if args.readout == 'spectral':
            suffix = f'_specreadL{target_layer}k' + '-'.join(map(str, k_indices))
        else:
            suffix = ''
        run_label = f'{model_key}_frozen_m{main_layer}{suffix}'
    else:
        rdir = Path(args.run_dir)
        cfg = json.load(open(rdir / 'config.json'))
        model_key = args.model or cfg['model']
        checkpoint_dir = 'lora_last' if args.checkpoint == 'last' else 'lora'
        checkpoint_path = rdir / checkpoint_dir
        if not checkpoint_path.exists():
            raise SystemExit(f'Missing {args.checkpoint} checkpoint: {checkpoint_path}')
        checkpoint_suffix = '_last' if args.checkpoint == 'last' else ''
        main_layer = args.main_layer if args.main_layer is not None else cfg['main_layer']
        target_layer = args.target_layer if args.target_layer is not None else cfg.get('target_layer')
        k_indices = args.k_indices or cfg.get('k_indices') or [1, 2, 3, 4]
        if args.readout == 'spectral':
            kspec = ('k' + '-'.join(map(str, k_indices))) if args.k_indices else ''
            readout_suffix = f'_specread{kspec}'
        else:
            readout_suffix = f'_poolL{main_layer}' if args.main_layer is not None else ''
        run_label = f'{model_key}_{rdir.name}{checkpoint_suffix}{readout_suffix}'

    model_name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    num_layers = base.config.num_hidden_layers

    if args.frozen:
        model = base.to(device)
    else:
        model = PeftModel.from_pretrained(base, str(checkpoint_path)).to(device)
    model.eval()

    with open(args.test_corpus, encoding='utf-8') as f:
        blocks = json.load(f)
    langs = list(blocks[0]['texts'].keys())

    print(f'corpus: {args.test_corpus}  n={len(blocks)}  langs={len(langs)}')
    print(f'evaluating: {run_label}  main_layer={main_layer}')

    reps = {}
    for lang in langs:
        print(f'  {lang} ...', end=' ', flush=True)
        texts = [' '.join(b['texts'][lang]) for b in blocks]
        reps[lang] = encode_lang(model, tok, texts, device, args.batch_size,
                                 main_layer, num_layers, args.readout,
                                 target_layer, k_indices)
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
