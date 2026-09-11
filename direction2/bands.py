"""bands.py — Coarse spectral band definitions for Direction 2.

Five T-relative bands with boundaries at T//64, T//16, T//4 (factor-of-4
geometric progression in k). Boundaries scale with sequence length so each
band covers the same fraction of the k-range regardless of tokenisation or
language fertility.

    DC:       k = 0
    Low:      k = 1  ..  T//64    (λ ≥ 64 tokens; longest-range structure)
    Mid:      k = T//64+1..T//16
    High:     k = T//16+1..T//4
    VeryHigh: k > T//4            (λ < 4 tokens; fine-grained / sub-word)
"""
import numpy as np

BAND_NAMES = ['DC', 'Low', 'Mid', 'High', 'VeryHigh']
N_BANDS = len(BAND_NAMES)
_DIVS = [64, 16, 4]  # upper-boundary divisors for Low, Mid, High


def band_bounds(n):
    """Compute (k_lo, k_hi) pairs for sequence length n (k_hi exclusive).

    Boundaries n//64, n//16, n//4 scale with n; strictly increasing is
    enforced so every band gets at least one coefficient.
    """
    cuts = [0, 1]
    for div in _DIVS:
        cuts.append(max(cuts[-1] + 1, n // div + 1))
    cuts.append(max(cuts[-1], n))
    return [(cuts[i], cuts[i + 1]) for i in range(N_BANDS)]


def band_rms(F, b):
    """RMS power vector for band b from DCT matrix F of shape (d, n).

    Returns (d,) float32; zero vector if band has no coefficients.
    DC (b=0) returns the signed coefficient F[:,0] so that cosine similarity
    equals mean-pooled cosine rather than |mean|-pooled cosine.
    """
    k_lo, k_hi = band_bounds(F.shape[1])[b]
    k_hi = min(k_hi, F.shape[1])
    if k_hi <= k_lo:
        return np.zeros(F.shape[0], dtype=np.float32)
    if b == 0:  # singleton DC: preserve sign
        return F[:, 0].astype(np.float32)
    return np.sqrt(np.mean(F[:, k_lo:k_hi] ** 2, axis=1))  # avg over frequencies within this band
