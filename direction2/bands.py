"""bands.py — Coarse spectral band definitions for Direction 2.

Two schemes (both have 5 bands: DC, Low, Mid, High, VeryHigh):

  'abs'  T-relative k-index: boundaries at T//64, T//16, T//4.
         Low covers k=1..T//64, i.e. DCT-II components with half-period >= 64
         tokens (full wavelength >= 128 tokens), regardless of sequence length.
  'rel'  Fixed k-index: boundaries at k=4, 16, 64.
         Low covers k=1..4, i.e. components with half-period >= T//4 (25% of
         the context window), regardless of sequence length.

Half-period of DCT-II mode k: T/k.  Full wavelength: 2T/k.
Band vectors are RMS-aggregated (spectral magnitude/power); sign/phase is
discarded. DC (k=0) is the exception — its sign is preserved so that
cos(DC_A, DC_B) == cos(mean(H_A), mean(H_B)).
"""
import numpy as np

BAND_NAMES = ['DC', 'Low', 'Mid', 'High', 'VeryHigh']
N_BANDS = len(BAND_NAMES)
SCHEMES = ('abs', 'rel')

_ABS_DIVS = [64, 16, 4]   # T//div gives upper k for Low, Mid, High
_REL_CUTS = [4, 16, 64]   # fixed upper k for Low, Mid, High


def band_bounds(n, scheme='abs'):
    """(k_lo, k_hi) pairs for each of the 5 bands (k_hi exclusive).

    Args:
        n: sequence length (number of positions after pooling).
        scheme: 'abs' uses T-scaled k boundaries; 'rel' uses fixed k=4,16,64.
    """
    cuts = [0, 1]
    thresholds = [n // d for d in _ABS_DIVS] if scheme == 'abs' else _REL_CUTS
    for t in thresholds:
        cuts.append(max(cuts[-1] + 1, t + 1))
    cuts.append(max(cuts[-1], n))
    # Clamp at n so widths never exceed the actual DCT length.
    return [(min(cuts[i], n), min(cuts[i + 1], n)) for i in range(N_BANDS)]


def band_rms(F, b, scheme='abs'):
    """RMS power vector for band b from DCT matrix F of shape (d, n).

    Returns (d,) float32. DC (b=0) returns signed F[:,0] to preserve cosine
    equality with mean-pooled cosine.
    """
    k_lo, k_hi = band_bounds(F.shape[1], scheme)[b]
    k_hi = min(k_hi, F.shape[1])
    if k_hi <= k_lo:
        return np.zeros(F.shape[0], dtype=np.float32)
    if b == 0:
        return F[:, 0].astype(np.float32)
    return np.sqrt(np.mean(F[:, k_lo:k_hi] ** 2, axis=1))
