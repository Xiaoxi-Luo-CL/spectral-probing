'''Visualize original frequencies of the dataset.'''
from analysis.raw_encoder import dct, idct
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def freq(embeddings, attentions):
    for sidx in range(embeddings.shape[0]):
        seq_len = torch.sum(attentions[sidx])
        # seq_frequencies: (hidden_dim, seq_length)
        seq_frequencies = dct(embeddings[sidx, :seq_len].T)
        frq_pooled = F.adaptive_avg_pool1d(seq_frequencies, 512)


def interpolated_dct(embeddings_list, target_len=512):
    # Calculate DCT for each sentence and interpolate to target_len
    # n_dims = embeddings_list[0].shape[1]
    all_dct = []

    for emb in embeddings_list:
        seq_len = emb.shape[0]
        if seq_len < 5:
            continue
        emb_dct = dct(torch.Tensor(emb).T)
        # (1, 768, 512)
        pooled = F.interpolate(emb_dct.unsqueeze(
            0), size=target_len, mode='linear', align_corners=False)
        # Revert to shape: (target_len, 768)
        all_dct.append(pooled)

    return torch.concat(all_dct)  # (num_sentences, target_len, 768)


def analyze_and_plot(dct_matrix):
    dct_abs = np.abs(dct_matrix)

    # Mean across sentences
    mean_raw = np.mean(dct_matrix, axis=0)  # Shape: (768, 512)
    mean_abs = np.mean(dct_abs, axis=0)

    # Plot 1: Heatmaps (Raw vs Absolute)
    fig_hm, axes_hm = plt.subplots(1, 2, figsize=(16, 6))

    # Raw values with SymLogNorm to handle negative values
    im0 = axes_hm[0].imshow(mean_raw, aspect='auto', cmap='viridis',
                            norm=colors.SymLogNorm(linthresh=1e-2, linscale=1, base=10))
    axes_hm[0].set_title('Mean DCT (Raw, SymLog Scale)')
    axes_hm[0].set_xlabel('Frequency Bin (Interpolated)')
    axes_hm[0].set_ylabel('Embedding Dimension (0-767)')
    fig_hm.colorbar(im0, ax=axes_hm[0])

    # Absolute values with standard LogNorm
    im1 = axes_hm[1].imshow(mean_abs, aspect='auto', cmap='viridis',
                            norm=colors.LogNorm(vmin=max(1e-5, mean_abs.min())))
    axes_hm[1].set_title('Mean DCT (Absolute, Log Scale)')
    axes_hm[1].set_xlabel('Frequency Bin (Interpolated)')
    fig_hm.colorbar(im1, ax=axes_hm[1])

    fig_hm.tight_layout()
    fig_hm.savefig('dct_heatmaps.png', dpi=300)
    plt.close(fig_hm)

    # Plot 2: Line Plots (4 Subplots)
    fig_line, axes_line = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(512)

    # 1D aggregates across dimensions
    mean_raw_1d = np.mean(mean_raw, axis=0)
    std_raw_1d = np.std(mean_raw, axis=0)
    mean_abs_1d = np.mean(mean_abs, axis=0)
    std_abs_1d = np.std(mean_abs, axis=0)

    # [0, 0]: Raw without variance
    axes_line[0, 0].plot(x, mean_raw_1d, color='b')
    axes_line[0, 0].set_title('Raw Mean Spectrum (No Var)')

    # [0, 1]: Raw with variance
    axes_line[0, 1].plot(x, mean_raw_1d, color='b')
    axes_line[0, 1].fill_between(
        x, mean_raw_1d - std_raw_1d, mean_raw_1d + std_raw_1d, color='b', alpha=0.2)
    axes_line[0, 1].set_title('Raw Mean Spectrum (With Var)')

    # [1, 0]: Absolute without variance
    axes_line[1, 0].plot(x, mean_abs_1d, color='r')
    axes_line[1, 0].set_title('Absolute Mean Spectrum (No Var)')

    # [1, 1]: Absolute with variance
    axes_line[1, 1].plot(x, mean_abs_1d, color='r')
    axes_line[1, 1].fill_between(
        x, mean_abs_1d - std_abs_1d, mean_abs_1d + std_abs_1d, color='r', alpha=0.2)
    axes_line[1, 1].set_title('Absolute Mean Spectrum (With Var)')

    for ax in axes_line.flat:
        ax.set_xlabel('Frequency Bin')
        ax.set_ylabel('Magnitude')

    fig_line.tight_layout()
    fig_line.savefig('dct_line_plots.png', dpi=300)
    plt.close(fig_line)


if __name__ == '__main__':
    data = torch.load(
        '/u501/x25luo/codebase/spectral-probing/analysis/en-ewt-train-pos_bert-base-cased_embedding.pt', weights_only=False)
    embeddings_list = data['embeddings']

    pooled_dct = interpolated_dct(embeddings_list, target_len=512)
    analyze_and_plot(pooled_dct.numpy())
