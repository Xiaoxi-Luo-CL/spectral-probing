"""
metrics.py — Per-band cross-lingual alignment metrics for Direction 2.

Reads pre-computed band vectors from embed.py and computes, for each language pair:

    band_adv      (n_layers, N_BANDS)  translation advantage per band (primary diagnostic)
    band_p1       (n_layers, N_BANDS)  bidirectional P@1 per band
    band_energy_A (n_layers, N_BANDS)  relative band energy (language A, Parseval-correct)
    band_energy_B (n_layers, N_BANDS)  relative band energy (language B)
    mu_adv        (n_layers,)          full-sequence pairwise-cosine advantage (from mu)
    cum_band_adv  (n_layers, N_BANDS)  cumulative band-summary alignment A_{≤b} (raw, not normalised)

cum_band_adv uses width-scaled band vectors q_b = sqrt(width_b) * p_b so each band contributes
its total energy to the concatenated norm (Parseval-correct). Values are raw translation
advantages — not normalised — so non-monotonicity is visible when adding bands hurts retrieval.
mu_adv is the translation advantage of the dot-product matrix mu_A @ mu_B.T, which
equals the average pairwise cosine (sum over all position pairs, normalised per token).
band_energy uses width-corrected total band energy (mean_k → sum_k via Parseval bandwidth factor).

Usage (from spectral-probing/):
    python -m direction2.metrics \
        --model mbert \
        --lang_a en --lang_b de ar zh hi sw \
        --embed_dir direction2/results/embed \
        --out_dir direction2/results/metrics
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from direction2.bands import SCHEMES, band_bounds
from direction2.embed import load_embed, pool_dir, WORD_POOL_EXCLUDED
from models.encoders import PrismEncoder


def _norm_rows(V):
    """L2-normalise rows of a (n, d) float32 array."""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(norms, 1e-9)


def cos_matrix(band_A, band_B):
    """(n_A, n_B) cosine similarity matrix from (n_A, d) and (n_B, d) arrays."""
    return _norm_rows(band_A.astype(np.float32)) @ _norm_rows(band_B.astype(np.float32)).T


def translation_advantage(cos_mat):
    """Mean diagonal minus mean off-diagonal of cosine matrix.

    Args:
        cos_mat: (n, n) float array.

    Returns:
        float.
    """
    n = cos_mat.shape[0]
    pos = np.mean(np.diag(cos_mat))
    neg = (cos_mat.sum() - np.trace(cos_mat)) / max(n * (n - 1), 1)
    return float(pos - neg)


def retrieval_p1(cos_mat):
    """Bidirectional P@1 from (n_A, n_B) cosine matrix.

    Assumes block i in A corresponds to block i in B (diagonal positives).

    Returns:
        (p1_ab, p1_ba) — A→B and B→A accuracy separately.
    """
    n = min(cos_mat.shape)
    p1_ab = float(np.mean(np.argmax(cos_mat[:n, :], axis=1) == np.arange(n)))
    p1_ba = float(np.mean(np.argmax(cos_mat[:, :n], axis=0) == np.arange(n)))
    return p1_ab, p1_ba


def compute_pair_metrics(bands_A, mu_A, ids_A, seq_lens_A, bands_B, mu_B, ids_B, seq_lens_B,
                         scheme='abs'):
    """Compute all alignment metrics for one language pair.

    Joins on block_id so row order in the two npz files does not need to match.

    Args:
        bands_A: (n, n_layers, N_BANDS, d) float16.
        mu_A:    (n, n_layers, d) float16.
        ids_A:   (n,) int32 — block IDs for language A.
        seq_lens_A: (n,) int32 — position count per block for language A.
        bands_B, mu_B, ids_B, seq_lens_B: same for language B.

    Returns:
        dict with keys 'band_adv', 'band_p1', 'p1_ab', 'p1_ba',
        'band_energy_A', 'band_energy_B', 'mu_adv', 'cum_band_adv', 'block_ids'.
    """
    common, idx_A, idx_B = np.intersect1d(ids_A, ids_B, return_indices=True)
    if len(common) == 0:
        raise ValueError('No common block IDs between the two languages')

    bands_A, mu_A, seq_lens_A = bands_A[idx_A], mu_A[idx_A], seq_lens_A[idx_A]
    bands_B, mu_B, seq_lens_B = bands_B[idx_B], mu_B[idx_B], seq_lens_B[idx_B]

    n, n_layers, n_bands, d = bands_A.shape
    band_adv     = np.zeros((n_layers, n_bands), dtype=np.float32)
    p1_ab        = np.zeros((n_layers, n_bands), dtype=np.float32)   # A→B
    p1_ba        = np.zeros((n_layers, n_bands), dtype=np.float32)   # B→A
    band_energy_A = np.zeros((n_layers, n_bands), dtype=np.float32)
    band_energy_B = np.zeros((n_layers, n_bands), dtype=np.float32)
    mu_adv       = np.zeros(n_layers, dtype=np.float32)
    cum_band_adv = np.zeros((n_layers, n_bands), dtype=np.float32)

    # Parseval bandwidth factor: width_b(T) = k_hi - k_lo for each block
    widths_A = np.array(
        [[bb[1] - bb[0] for bb in band_bounds(max(int(t), 1), scheme)] for t in seq_lens_A],
        dtype=np.float32,
    )  # (n, n_bands)
    widths_B = np.array(
        [[bb[1] - bb[0] for bb in band_bounds(max(int(t), 1), scheme)] for t in seq_lens_B],
        dtype=np.float32,
    )  # (n, n_bands)

    for li in range(n_layers):
        # full-sequence advantage: mu dot-product gives average pairwise cosine
        cos_full = mu_A[:, li, :].astype(np.float32) @ mu_B[:, li, :].astype(np.float32).T
        mu_adv[li] = translation_advantage(cos_full)

        # cumulative state: accumulate squared norms and cross dot-products across bands
        cum_cross_dot = np.zeros((n, n), dtype=np.float32)
        cum_sq_A = np.zeros(n, dtype=np.float32)
        cum_sq_B = np.zeros(n, dtype=np.float32)

        for b in range(n_bands):
            band_A = bands_A[:, li, b, :].astype(np.float32)
            band_B = bands_B[:, li, b, :].astype(np.float32)

            # bandwidth-corrected total energy: width × mean_k(sum_j F[j,k]^2)
            band_energy_A[li, b] = float(np.mean(np.sum(band_A ** 2, axis=1) * widths_A[:, b]))
            band_energy_B[li, b] = float(np.mean(np.sum(band_B ** 2, axis=1) * widths_B[:, b]))

            cos_mat = cos_matrix(band_A, band_B)
            band_adv[li, b] = translation_advantage(cos_mat)
            p1_ab[li, b], p1_ba[li, b] = retrieval_p1(cos_mat)

            # scaled_b = sqrt(width_b) * band_b so each band contributes its total energy
            # to the concatenated norm (Parseval-correct)
            scaled_A = band_A * np.sqrt(widths_A[:, b])[:, None]
            scaled_B = band_B * np.sqrt(widths_B[:, b])[:, None]
            cum_cross_dot += scaled_A @ scaled_B.T
            cum_sq_A += np.sum(scaled_A ** 2, axis=1)
            cum_sq_B += np.sum(scaled_B ** 2, axis=1)
            denom = np.outer(np.sqrt(cum_sq_A), np.sqrt(cum_sq_B))
            cos_cum = np.where(denom > 1e-9, cum_cross_dot / denom, 0.0)
            cum_band_adv[li, b] = translation_advantage(cos_cum)

    # relative band energy (sum to 1 over bands per layer)
    for energy in (band_energy_A, band_energy_B):
        energy_total = energy.sum(axis=1, keepdims=True)
        energy[:] = np.where(energy_total > 1e-9, energy / energy_total, 0.0)

    band_p1 = (p1_ab + p1_ba) / 2
    return {
        'band_adv': band_adv, 'band_p1': band_p1, 'p1_ab': p1_ab, 'p1_ba': p1_ba,
        'band_energy_A': band_energy_A, 'band_energy_B': band_energy_B,
        'mu_adv': mu_adv, 'cum_band_adv': cum_band_adv, 'block_ids': common,
    }


def verify_dc_identity(n_trials=5, rtol=1e-4):
    """Verify DC band vector equals sqrt(n) * mean(H) and cosine(DC_A, DC_B) == cosine(mean_A, mean_B).

    Uses the same PrismEncoder.dct(norm='ortho') as embed.py.

    Args:
        n_trials: number of random (n, d) matrices to test.
        rtol: relative tolerance for np.allclose checks.
    """
    rng = np.random.default_rng(0)
    print('Verifying DC band identity  (DC = sqrt(n) * mean(H))')
    print()

    all_ok = True
    for trial in range(n_trials):
        n = rng.integers(20, 200)
        d = rng.integers(64, 512)
        H = rng.standard_normal((n, d)).astype(np.float32)

        # DCT path used in embed.py: F = dct(H.T, norm='ortho')  →  (d, n)
        F = PrismEncoder.dct(torch.from_numpy(H).T, norm='ortho').numpy()

        dc_vec = F[:, 0]
        mean_vec = H.mean(axis=0)
        expected = np.sqrt(n) * mean_vec

        ok1 = np.allclose(dc_vec, expected, rtol=rtol, atol=1e-5)  # float32 noise ~1e-6

        H2 = rng.standard_normal((n, d)).astype(np.float32)
        F2 = PrismEncoder.dct(torch.from_numpy(H2).T, norm='ortho').numpy()
        dc2 = F2[:, 0]
        mean2 = H2.mean(axis=0)

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        cos_dc = cos(dc_vec, dc2)
        cos_mean = cos(mean_vec, mean2)
        ok2 = abs(cos_dc - cos_mean) < rtol

        status = 'PASS' if (ok1 and ok2) else 'FAIL'
        if not (ok1 and ok2):
            all_ok = False
        max_err = float(np.max(np.abs(dc_vec - expected)))
        print(
            f'  trial {trial+1}: n={n:3d}  d={d:3d}'
            f'  DC==sqrt(n)*mean: {"ok" if ok1 else "FAIL"}  max_err={max_err:.2e}'
            f'  cos(DC_A,DC_B)==cos(mean_A,mean_B): {"ok" if ok2 else "FAIL"}'
            f'  |Δcos|={abs(cos_dc-cos_mean):.2e}'
            f'  [{status}]'
        )

    print()
    if all_ok:
        print('All trials PASSED.')
        print()
        print('Conclusion:')
        print('  With norm="ortho", F[:,0] = sqrt(n) * mean_t(H[t,:])')
        print('  => cosine(DC_A, DC_B)  ==  cosine(mean(H_A), mean(H_B))')
        print('  => band_adv[DC] is exactly the mean-pool translation advantage.')
    else:
        print('Some trials FAILED — check DCT norm setting.')


def parse_args():
    p = argparse.ArgumentParser(description='Compute per-band alignment metrics for language pairs')
    p.add_argument('--verify', action='store_true', help='run DC identity check and exit')
    p.add_argument('--model', default=None)
    p.add_argument('--pool', default='subword', choices=('subword', 'word'))
    p.add_argument('--bands', default='abs', choices=SCHEMES, dest='bands_scheme',
                   help='abs: T-relative k boundaries; rel: fixed k=4,16,64')
    p.add_argument('--corpus', default='',
                   help='corpus identifier; must match the one used in embed.py')
    p.add_argument('--lang_a', nargs='+', default=['en'])
    p.add_argument('--lang_b', nargs='+', default=[])
    p.add_argument('--embed_dir', default='direction2/results/embed')
    p.add_argument('--out_dir', default='direction2/results/metrics')
    p.add_argument('--no_skip', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.verify:
        verify_dc_identity()
        return
    if not args.model or not args.lang_b:
        raise SystemExit('--model and --lang_b are required unless --verify is set')
    pdir = pool_dir(args.pool, args.bands_scheme)
    def _base(root): return Path(root) / args.corpus / args.model if args.corpus else Path(root) / args.model
    embed_root = _base(args.embed_dir) / pdir
    out_dir = _base(args.out_dir) / pdir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {}

    def _load(lang):
        if lang not in cache:
            cache[lang] = load_embed(embed_root / f'{lang}.npz')
        return cache[lang]

    for la in args.lang_a:
        for lb in args.lang_b:
            if la == lb:
                continue
            if args.pool == 'word':
                excluded = [l for l in (la, lb) if l in WORD_POOL_EXCLUDED]
                if excluded:
                    print(f'  [{la}-{lb}] skipped — word pooling excluded for: {", ".join(excluded)}')
                    continue
            out_path = out_dir / f'{la}-{lb}.npz'
            if not args.no_skip and out_path.exists():
                print(f'  [{la}-{lb}] exists — skip')
                continue
            print(f'  [{la}-{lb}] computing ...')
            bands_A, mu_A, ids_A, _, seq_lens_A = _load(la)
            bands_B, mu_B, ids_B, _, seq_lens_B = _load(lb)
            m = compute_pair_metrics(bands_A, mu_A, ids_A, seq_lens_A, bands_B, mu_B, ids_B,
                                     seq_lens_B, scheme=args.bands_scheme)
            np.savez_compressed(out_path, **m)
            last = -1
            print(f'    band_adv[last]    : {[f"{v:.3f}" for v in m["band_adv"][last]]}')
            print(f'    band_p1[last]     : {[f"{v:.3f}" for v in m["band_p1"][last]]}')
            print(f'    cum_band_adv[last]: {[f"{v:.3f}" for v in m["cum_band_adv"][last]]}')
            print(f'    mu_adv[last]      : {m["mu_adv"][last]:.4f}')


if __name__ == '__main__':
    main()
