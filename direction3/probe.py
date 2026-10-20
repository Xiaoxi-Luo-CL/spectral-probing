"""
direction3/probe.py — frozen spectral probing for decoder-only models.

Extracts hidden states at specified layers, computes mean-pool, last-token,
and spectral (k=1..4 DCT) representations, then reports bidirectional P@1
per layer and representation type.

Usage (from spectral-probing/):
    python -m direction3.probe \
        --model qwen3-1.7b \
        --corpus direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json \
        --layers 4 8 12 16 20 24 28

    python -m direction3.probe --model qwen3-1.7b  # uses default layers + UN eval
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from direction2.adapter import filter_token_safe
from direction2.alignment import content_masks
from direction2.eval_full import retrieval_p1
from models.encoders import PrismEncoder


MODELS = {
    'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
    'qwen3-4b':   'Qwen/Qwen3-4B',
}

DEFAULT_LAYERS = {
    'qwen3-1.7b': [4, 8, 12, 16, 20, 24, 28],
    'qwen3-4b':   [5, 10, 15, 20, 25, 30, 36],
}

DEFAULT_CORPUS = 'direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json'
REPR_TYPES = ('mean', 'last', 'spectral')


def spectral_rep(h, ks=(1, 2, 3, 4)):
    """Compute concat of DCT coefficients at indices ks.

    h: (T, d) float tensor, content tokens only.
    Returns: (len(ks)*d,) tensor.
    """
    F = PrismEncoder.dct(h.T, norm='ortho')  # (d, T)
    parts = []
    for k in ks:
        parts.append(F[:, k] if k < F.shape[1] else torch.zeros(F.shape[0], device=h.device))
    return torch.cat(parts, dim=0)


def encode_lang(model, tok, texts, device, batch_size, layers, ks=(1, 2, 3, 4)):
    """Encode all texts; return {layer: {'mean': (N,d), 'last': (N,d), 'spectral': (N,len(ks)*d)}}."""
    accum = {layer: {rep: [] for rep in REPR_TYPES} for layer in layers}

    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(
                texts[s:s + batch_size], return_tensors='pt', padding=True,
            ).to(device)
            masks = content_masks(enc, device)
            out = model(**enc, output_hidden_states=True)

            for layer in layers:
                H = out.hidden_states[layer]  # (B, T, d)
                for i, mask in enumerate(masks):
                    h = H[i, mask].float()  # (T_content, d)
                    accum[layer]['mean'].append(h.mean(0).cpu().numpy())
                    accum[layer]['last'].append(h[-1].cpu().numpy())
                    accum[layer]['spectral'].append(spectral_rep(h, ks).cpu().numpy())

    result = {}
    for layer in layers:
        result[layer] = {}
        for r in REPR_TYPES:
            reps = np.stack(accum[layer][r])  # (N, *)
            reps /= np.linalg.norm(reps, axis=1, keepdims=True).clip(1e-9)
            result[layer][r] = reps
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3-1.7b', choices=list(MODELS))
    p.add_argument('--corpus', default=DEFAULT_CORPUS)
    p.add_argument('--layers', type=int, nargs='+', default=None,
                   help='hidden state indices to probe (default: model-specific list)')
    p.add_argument('--out_dir', default='direction3/results/probe')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--k_indices', type=int, nargs='+', default=[1, 2, 3, 4],
                   help='DCT coefficients for the spectral representation')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    layers = args.layers or DEFAULT_LAYERS[args.model]
    model_name = MODELS[args.model]

    print(f'model: {model_name}  device: {device}')
    print(f'layers: {layers}')

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    with open(args.corpus, encoding='utf-8') as f:
        blocks = json.load(f)
    langs = list(blocks[0]['texts'].keys())

    blocks = filter_token_safe(blocks, tok, langs, args.max_length)
    print(f'corpus: {args.corpus}  n={len(blocks)}  langs={len(langs)}')

    # Encode all languages
    reps = {}
    for lang in langs:
        print(f'  {lang} ...', end=' ', flush=True)
        texts = [' '.join(b['texts'][lang]) for b in blocks]
        reps[lang] = encode_lang(
            model, tok, texts, device, args.batch_size, layers, args.k_indices)
        print('done')

    # Evaluate retrieval P@1 per layer and representation type
    rows = []
    for layer in layers:
        for repr_type in REPR_TYPES:
            p1_vals = []
            for la, lb in combinations(langs, 2):
                p1 = retrieval_p1(reps[la][layer][repr_type], reps[lb][layer][repr_type])
                rows.append({'layer': layer, 'repr': repr_type,
                             'pair': f'{la}-{lb}', 'p1': round(p1, 4)})
                p1_vals.append(p1)
            mean_p1 = float(np.mean(p1_vals))
            rows.append({'layer': layer, 'repr': repr_type,
                         'pair': 'mean', 'p1': round(mean_p1, 4)})
            print(f'  L{layer:>2} {repr_type:<8}  mean P@1 = {mean_p1:.4f}')

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_stem = Path(args.corpus).stem
    ksfx = '' if args.k_indices == [1, 2, 3, 4] else '_k' + '-'.join(map(str, args.k_indices))
    out_path = out_dir / f'{corpus_stem}{ksfx}.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['layer', 'repr', 'pair', 'p1'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nsaved → {out_path}')


if __name__ == '__main__':
    main()
