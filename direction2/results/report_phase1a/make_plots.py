# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "seaborn", "pandas", "numpy"]
# ///
"""
Generate Phase 1a analysis plots for mBERT, XLM-R, and LaBSE.

python direction2/results/report_phase1a/make_plots.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path('/u501/x25luo/.claude/skills/generate-report/scripts')))
from plots import save_svg, PALETTE, CATEGORICAL

OUT = Path('/u501/x25luo/codebase/spectral-probing/direction2/results/report_phase1a')
METRICS = Path('/u501/x25luo/codebase/spectral-probing/direction2/results/metrics')

LAYERS = [0, 2, 4, 6, 8, 10, 12]
BANDS = ['DC', 'Low', 'Mid', 'High', 'VHigh']
PAIRS = ['en-de', 'en-ar', 'en-zh', 'en-hi', 'en-sw']
PAIR_LABELS = {'en-de': 'de', 'en-ar': 'ar', 'en-zh': 'zh', 'en-hi': 'hi', 'en-sw': 'sw'}
MODELS = ['mbert', 'xlmr', 'labse']
MODEL_LABELS = {'mbert': 'mBERT', 'xlmr': 'XLM-R', 'labse': 'LaBSE'}
N_BANDS, N_LAYERS = 5, 7

BAND_COLORS = CATEGORICAL[:5]
MODEL_COLORS = {'mbert': CATEGORICAL[0], 'xlmr': CATEGORICAL[1], 'labse': CATEGORICAL[2]}
PAIR_COLORS = {p: CATEGORICAL[i] for i, p in enumerate(PAIRS)}


def load_model(model):
    files = {p: np.load(METRICS / model / 'subword' / f'{p}.npz') for p in PAIRS}
    band_adv = np.mean([files[p]['band_adv'] for p in PAIRS], 0)
    band_p1 = np.mean([files[p]['band_p1'] for p in PAIRS], 0)
    p1_ab = np.mean([files[p]['p1_ab'] for p in PAIRS], 0)
    p1_ba = np.mean([files[p]['p1_ba'] for p in PAIRS], 0)
    mu_adv = np.mean([files[p]['mu_adv'] for p in PAIRS], 0)
    band_energy_A = np.mean([files[p]['band_energy_A'] for p in PAIRS], 0)
    band_energy_B = np.mean([files[p]['band_energy_B'] for p in PAIRS], 0)
    cum_band_adv = np.mean([files[p]['cum_band_adv'] for p in PAIRS], 0)
    return {'band_adv': band_adv, 'band_p1': band_p1, 'p1_ab': p1_ab, 'p1_ba': p1_ba,
            'mu_adv': mu_adv, 'band_energy_A': band_energy_A, 'band_energy_B': band_energy_B,
            'cum_band_adv': cum_band_adv, 'files': files}


data = {m: load_model(m) for m in MODELS}


# Fig 1: mBERT DC vs Low P@1 by layer, one panel per language pair
fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
for ax, pair in zip(axes, PAIRS):
    f = data['mbert']['files'][pair]
    dc = f['band_p1'][:, 0]
    lo = f['band_p1'][:, 1]
    ax.plot(LAYERS, dc, color=CATEGORICAL[0], marker='o', ms=5, lw=2, label='DC (=mean-pool)')
    ax.plot(LAYERS, lo, color=CATEGORICAL[1], marker='s', ms=5, lw=2, label='Low')
    for li in range(N_LAYERS):
        if lo[li] > dc[li]:
            ax.axvline(LAYERS[li], color='#d4880f', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(f'en-{PAIR_LABELS[pair]}', fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel('Bidirectional P@1')
axes[0].legend(fontsize=8)
fig.suptitle('mBERT: DC vs Low band P@1 by layer (orange dashed = Low > DC)', y=1.02)
fig.tight_layout()
save_svg(fig, str(OUT / 'fig1_mbert_dc_vs_low_per_pair.svg'))
plt.close(fig)


# Fig 2: DC P@1 by layer, one panel per model, 5 lines (one per pair)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, model in zip(axes, MODELS):
    for pair in PAIRS:
        f = data[model]['files'][pair]
        dc = f['band_p1'][:, 0]
        ax.plot(LAYERS, dc, color=PAIR_COLORS[pair], marker='o', ms=4, lw=1.8,
                label=f'en-{PAIR_LABELS[pair]}')
    ax.set_title(MODEL_LABELS[model], fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
axes[0].set_ylabel('DC P@1 (= mean-pool retrieval)')
fig.suptitle('DC band P@1 by layer and language pair — all 3 models', y=1.02)
fig.tight_layout()
save_svg(fig, str(OUT / 'fig2_dc_p1_per_pair_3models.svg'))
plt.close(fig)


# Fig 3: P@1 by band at mBERT L8, one line per language pair
L8_IDX = LAYERS.index(8)
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(N_BANDS)
for pair in PAIRS:
    f = data['mbert']['files'][pair]
    vals = f['band_p1'][L8_IDX, :]
    ax.plot(x, vals, color=PAIR_COLORS[pair], marker='o', ms=5, lw=2,
            label=f'en-{PAIR_LABELS[pair]}')
ax.set_xticks(x)
ax.set_xticklabels(BANDS)
ax.set_ylabel('Bidirectional P@1')
ax.set_title('mBERT L8: P@1 by band per language pair')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
fig.tight_layout()
save_svg(fig, str(OUT / 'fig3_mbert_L8_band_p1_per_pair.svg'))
plt.close(fig)


# Fig 4: Directional P@1 asymmetry — mBERT Low band, per pair
fig, ax = plt.subplots(figsize=(8, 4))
for pair in PAIRS:
    f = data['mbert']['files'][pair]
    ab = f['p1_ab'][:, 1]   # en -> tgt, Low band
    ba = f['p1_ba'][:, 1]   # tgt -> en, Low band
    col = PAIR_COLORS[pair]
    ax.plot(LAYERS, ab, color=col, lw=1.8, marker='o', ms=4,
            label=f'en-{PAIR_LABELS[pair]} en→tgt')
    ax.plot(LAYERS, ba, color=col, lw=1.2, marker='s', ms=4, ls='--',
            label=f'en-{PAIR_LABELS[pair]} tgt→en')
ax.set_xlabel('Layer')
ax.set_ylabel('P@1')
ax.set_title('mBERT Low band: directional P@1 per pair (solid=en→tgt, dashed=tgt→en)')
ax.set_ylim(0, 1.05)
ax.set_xticks(LAYERS)
ax.legend(fontsize=6.5, ncol=2)
fig.tight_layout()
save_svg(fig, str(OUT / 'fig4_mbert_low_directional_p1.svg'))
plt.close(fig)


# JSON summary
summary = {'models': {}, 'pairs': PAIRS, 'layers': LAYERS, 'bands': BANDS}
for m in MODELS:
    d = data[m]
    m_pairs = data[m]['files']
    summary['models'][m] = {
        'band_adv_final': {b: float(d['band_adv'][-1, i]) for i, b in enumerate(BANDS)},
        'band_p1_final':  {b: float(d['band_p1'][-1, i])  for i, b in enumerate(BANDS)},
        'mu_adv_final':   float(d['mu_adv'][-1]),
        'band_energy_A_final': {b: float(d['band_energy_A'][-1, i]) for i, b in enumerate(BANDS)},
        'cum_band_adv_final': {b: float(d['cum_band_adv'][-1, i]) for i, b in enumerate(BANDS)},
        'band_adv_all': {b: [float(v) for v in d['band_adv'][:, i]] for i, b in enumerate(BANDS)},
        'band_p1_all':  {b: [float(v) for v in d['band_p1'][:, i]] for i, b in enumerate(BANDS)},
        'mu_adv_all':   [float(v) for v in d['mu_adv']],
        'per_pair_band_adv_final': {
            p: {b: float(m_pairs[p]['band_adv'][-1, i]) for i, b in enumerate(BANDS)} for p in PAIRS},
        'per_pair_band_p1_final': {
            p: {b: float(m_pairs[p]['band_p1'][-1, i]) for i, b in enumerate(BANDS)} for p in PAIRS},
        'per_pair_mu_adv_final': {p: float(m_pairs[p]['mu_adv'][-1]) for p in PAIRS},
    }

with open(OUT / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('All plots saved to', OUT)
print()
for m in MODELS:
    d = data[m]
    print(f'--- {MODEL_LABELS[m]} ---')
    ba = d['band_adv']
    bp1 = d['band_p1']
    print('  band_adv final: ' + '  '.join('%s=%.4f' % (b, ba[-1, i]) for i, b in enumerate(BANDS)))
    print('  band_p1 final:  ' + '  '.join('%s=%.3f' % (b, bp1[-1, i]) for i, b in enumerate(BANDS)))
    print('  mu_adv L12: %.4f' % d['mu_adv'][-1])
    print('  Low>DC layers (band_p1):', [LAYERS[li] for li in range(N_LAYERS) if bp1[li, 1] > bp1[li, 0]])
    print()
