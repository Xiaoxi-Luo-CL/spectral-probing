"""
embed.py — Extract multi-layer hidden states and compute DCT coarse-band vectors.

For each (model, pool, language), saves an npz at:
    <out_dir>/<model>/<pool>/<lang>.npz

Contents:
    bands      float16  (n_blocks, n_layers, N_BANDS, d)  RMS band-power vectors
    mu         float16  (n_blocks, n_layers, d)            mean of row-normalised H
    block_ids  int32    (n_blocks,)
    doc_ids    U64      (n_blocks,)                        document ID for each block
    seq_lens   int32    (n_blocks,)                        position count after pooling

Pool modes:
    subword  raw content tokens (excluding CLS/SEP)
    word     average subword embeddings within each whitespace word span

DCT uses norm='ortho' (orthonormal) so band energies satisfy Parseval exactly.

Usage (from spectral-probing/):
    python -m direction2.embed \
        --model mbert --pool subword --langs en de ar zh hi sw \
        --blocks direction2/data/NTREX/aligned/pilot_blocks.json \
        --out_dir direction2/results/embed
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from direction2.bands import N_BANDS, band_rms
from models.encoders import PrismEncoder

POOL_MODES = ('subword', 'word')


def _pool_words(H, word_ids):
    """Average subword embeddings within each whitespace word (scatter mean).

    Args:
        H: (n_all_tokens, d) tensor including special tokens.
        word_ids: list[int|None] from enc.word_ids(); None for special tokens.

    Returns:
        (n_words, d) tensor.
    """
    wids = torch.tensor(
        [w if w is not None else -1 for w in word_ids],
        device=H.device, dtype=torch.long,
    )
    mask = wids >= 0
    H_c, wids_c = H[mask], wids[mask]
    n_words = int(wids_c.max().item()) + 1
    out = torch.zeros(n_words, H.shape[1], device=H.device, dtype=H.dtype)
    cnt = torch.zeros(n_words, device=H.device, dtype=H.dtype)
    out.scatter_add_(0, wids_c.unsqueeze(1).expand_as(H_c), H_c)
    cnt.scatter_add_(0, wids_c, torch.ones(len(wids_c), device=H.device, dtype=H.dtype))
    return out / cnt.unsqueeze(1).clamp(min=1)

MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr': 'xlm-roberta-base',
    'labse': 'sentence-transformers/LaBSE',
}
DEFAULT_LAYERS = [0, 2, 4, 6, 8, 10, 12]


@torch.no_grad()
def embed_block(model, tok, text, layers, pool='subword'):
    """Extract band vectors and mu for a single text block.

    Args:
        model: AutoModel in eval mode.
        tok: fast AutoTokenizer.
        text: str — block sentences joined with spaces.
        layers: list[int] — hidden state indices (0 = embedding layer).
        pool: 'subword' uses raw content tokens; 'word' averages within whitespace words.

    Returns:
        bands: (n_layers, N_BANDS, d) float32.
        mu: (n_layers, d) float32.
        n_pos: int — number of positions after pooling.
    """
    enc = tok(text, return_tensors='pt', truncation=True, max_length=512)
    device = next(model.parameters()).device
    inp = {k: enc[k].to(device) for k in ('input_ids', 'attention_mask') if k in enc}
    out = model(**inp, output_hidden_states=True)

    word_ids = enc.word_ids(batch_index=0)
    if pool == 'word':
        n_pos = len(set(w for w in word_ids if w is not None))
        content_mask = None
    else:
        content_mask = torch.tensor([w is not None for w in word_ids], device=device)
        n_pos = int(content_mask.sum().item())

    d = out.hidden_states[0].shape[-1]
    bands = np.zeros((len(layers), N_BANDS, d), dtype=np.float32)
    mu = np.zeros((len(layers), d), dtype=np.float32)

    for li, layer_idx in enumerate(layers):
        H_all = out.hidden_states[layer_idx][0]  # (n_all_tokens, d)
        H = _pool_words(H_all, word_ids) if pool == 'word' else H_all[content_mask]
        if H.shape[0] == 0:
            continue

        # mean of row-normalised H; dot product gives average pairwise cosine
        H_n = H / H.norm(dim=1, keepdim=True).clamp(min=1e-9)
        mu[li] = H_n.mean(dim=0).cpu().float().numpy()

        # norm='ortho' ensures Parseval: sum_k ||F[:,k]||^2 == sum_t ||H[t,:]||^2
        F = PrismEncoder.dct(H.float().T, norm='ortho').cpu().numpy()  # (d, n_pos)
        for b in range(N_BANDS):
            bands[li, b] = band_rms(F, b)

    return bands, mu, n_pos


def embed_lang(model, tok, blocks, lang, layers, pool='subword'):
    """Embed all blocks for one language.

    Args:
        model: AutoModel.
        tok: fast AutoTokenizer.
        blocks: list of block dicts (with 'texts' key).
        lang: str — key into block['texts'].
        layers: list[int].
        pool: 'subword' or 'word'.

    Returns:
        bands: (n_blocks, n_layers, N_BANDS, d) float16.
        mu: (n_blocks, n_layers, d) float16.
        block_ids: (n_blocks,) int32.
        doc_ids: (n_blocks,) str — document ID for each block (for document-level bootstrap).
        seq_lens: (n_blocks,) int32 — position count after pooling.
    """
    n = len(blocks)
    d = model.config.hidden_size
    bands = np.zeros((n, len(layers), N_BANDS, d), dtype=np.float16)
    mu_arr = np.zeros((n, len(layers), d), dtype=np.float16)
    seq_lens = np.zeros(n, dtype=np.int32)
    ids = np.array([b['block_id'] for b in blocks], dtype=np.int32)
    dids = np.array([b['doc_id'] for b in blocks])

    for i, block in enumerate(blocks):
        text = ' '.join(block['texts'].get(lang, []))
        if not text:
            continue
        b_out, m_out, n_pos = embed_block(model, tok, text, layers, pool=pool)
        bands[i] = b_out.astype(np.float16)
        mu_arr[i] = m_out.astype(np.float16)
        seq_lens[i] = n_pos
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f'  [{lang}] {i + 1}/{n}')

    return bands, mu_arr, ids, dids, seq_lens


def save_embed(path, bands, mu, block_ids, doc_ids, seq_lens):
    """Save embedding arrays to a compressed npz."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, bands=bands, mu=mu, block_ids=block_ids, doc_ids=doc_ids, seq_lens=seq_lens)


def load_embed(path):
    """Load embedding arrays. Returns (bands, mu, block_ids, doc_ids, seq_lens)."""
    d = np.load(path, allow_pickle=True)
    return d['bands'], d['mu'], d['block_ids'], d['doc_ids'], d['seq_lens']


def parse_args():
    p = argparse.ArgumentParser(description='Extract NTREX band vectors per model/language')
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--pool', default='subword', choices=POOL_MODES)
    p.add_argument('--langs', nargs='+', required=True, help='short language codes')
    p.add_argument('--blocks', required=True, help='blocks JSON from ntrex.py')
    p.add_argument('--out_dir', default='direction2/results/embed')
    p.add_argument('--layers', nargs='+', type=int, default=DEFAULT_LAYERS)
    p.add_argument('--no_skip', action='store_true', help='overwrite existing npz files')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    with open(args.blocks, encoding='utf-8') as f:
        blocks = json.load(f)

    model_name = MODELS[args.model]
    print(f'Loading {model_name} on {device} ...')
    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = transformers.AutoModel.from_pretrained(model_name, return_dict=True).to(device).eval()

    for lang in args.langs:
        out_path = Path(args.out_dir) / args.model / args.pool / f'{lang}.npz'
        if not args.no_skip and out_path.exists():
            print(f'  [{lang}] exists — skip')
            continue
        print(f'  [{lang}] extracting ...')
        bds, mu, ids, dids, seq_lens = embed_lang(model, tok, blocks, lang, args.layers, pool=args.pool)
        save_embed(out_path, bds, mu, ids, dids, seq_lens)
        print(f'  [{lang}] saved  shape={bds.shape}')


if __name__ == '__main__':
    main()
