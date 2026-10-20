"""Shared multilingual alignment losses for encoder and decoder models.

The module contains pooling, spectral representations, and direct frequency-axis
soft-DTW. Model-specific training entry points remain separate so existing
checkpoint paths and commands continue to work.
"""
from itertools import combinations, permutations

import torch
import torch.nn.functional as F

from models.encoders import PrismEncoder


POINT_SPECTRAL_MODES = ('low_rms', 'low_fk')
DISTANCE_SPECTRAL_MODES = ('rigid_freq', 'freq_dtw')
# Auxiliary on the ordinary pooled state at target_layer, with no DCT. Isolates
# "supervision at this layer" from "supervision in frequency space". Each
# trainer pools the way its own main loss does: mean for encoders, last token
# for decoders. Note the mean-pool case is already reachable as low_fk k=0 --
# the DC coefficient is sqrt(T) times the mean, and that scale cancels once
# infonece_loss L2-normalises -- so it is only a distinct control for decoders.
SPATIAL_MODES = ('spatial',)
SPECTRAL_MODES = (('none',) + POINT_SPECTRAL_MODES + SPATIAL_MODES
                  + DISTANCE_SPECTRAL_MODES)
DTW_COST_MODES = ('natural', 'equal_k')

# Finite stand-in for an unreachable soft-DTW cell. Large enough that its
# soft-min weight underflows to zero, small enough to stay finite in float32.
DTW_UNREACHABLE = 1e9


def content_masks(enc, device):
    """Return masks selecting content tokens and excluding special tokens."""
    return [
        torch.tensor(
            [word_id is not None for word_id in enc.word_ids(batch_index=i)],
            device=device,
        )
        for i in range(enc['input_ids'].shape[0])
    ]


def mean_pool(hidden, masks):
    """Mean-pool content-token states for each sequence."""
    return torch.stack([hidden[i, mask].mean(0) for i, mask in enumerate(masks)])


def sentence_pool(hidden, masks, pooling='mean'):
    """Pool an encoder's sentence vector: mean of content tokens, or position 0.

    Position 0 is [CLS] for BERT and <s> for XLM-R; both tokenizers pad on the
    right, so it is that token in every row.
    """
    if pooling == 'cls':
        return hidden[:, 0]
    return mean_pool(hidden, masks)


def last_pool(hidden, masks):
    """Select the last content-token state for each sequence."""
    return torch.stack([hidden[i, mask][-1] for i, mask in enumerate(masks)])


def get_hidden_state(outputs, layer, num_layers):
    """Return a requested hidden layer, using the final output when possible."""
    if layer == num_layers:
        return outputs.last_hidden_state
    return outputs.hidden_states[layer]


def infonece_loss(reps, n_blocks, n_langs, temperature):
    """Compute symmetric multilingual InfoNCE over all directed language pairs."""
    reps = reps.view(n_blocks, n_langs, -1)
    targets = torch.arange(n_blocks, device=reps.device)
    total = reps.new_zeros(())
    n_pairs = 0
    for lang_a, lang_b in permutations(range(n_langs), 2):
        scores = reps[:, lang_a] @ reps[:, lang_b].T
        total = total + F.cross_entropy(scores / temperature, targets)
        n_pairs += 1
    return total / n_pairs


def spectral_coefficients(hidden, masks, k_indices, strict=False):
    """Extract ordered signed DCT coefficients from variable-length sequences.

    Args:
        hidden: Tensor with shape ``(N, padded_length, hidden_size)``.
        masks: Boolean tensor with shape ``(N, padded_length)``.
        k_indices: Ordered frequency indices to extract.
        strict: Raise when any sequence is too short. Otherwise missing
            coefficients are represented by zeros for legacy compatibility.

    Returns:
        Tensor with shape ``(N, len(k_indices), hidden_size)``.
    """
    if not k_indices:
        raise ValueError('k_indices must not be empty')
    if list(k_indices) != sorted(set(k_indices)):
        raise ValueError('k_indices must be sorted and unique')

    all_coeffs = []
    for i in range(hidden.shape[0]):
        sequence = hidden[i][masks[i].bool()].float()
        freq = PrismEncoder.dct(sequence.T, norm='ortho').T
        coeffs = []
        for k in k_indices:
            if k < freq.shape[0]:
                coeffs.append(freq[k])
            elif strict:
                raise ValueError(
                    f'Frequency k={k} is unavailable for a sequence with '
                    f'{freq.shape[0]} content tokens'
                )
            else:
                coeffs.append(freq.new_zeros(freq.shape[1]))
        all_coeffs.append(torch.stack(coeffs))
    return torch.stack(all_coeffs)


def spectral_rep(hidden, masks, k_indices, mode):
    """Build the legacy RMS or flattened signed-coefficient representation."""
    if mode not in POINT_SPECTRAL_MODES:
        raise ValueError(f'Unsupported point spectral mode: {mode}')
    coeffs = spectral_coefficients(hidden, masks, k_indices)
    if mode == 'low_rms':
        return torch.sqrt(coeffs.square().mean(dim=1) + 1e-6)
    return coeffs.flatten(start_dim=1)


def soft_dtw(cost, gamma=0.1, band_radius=None):
    """Compute differentiable soft-DTW for a batch of local-cost matrices.

    The recursion is evaluated one anti-diagonal at a time. Cells on a shared
    anti-diagonal are independent, so the number of sequential steps is
    ``n + m - 1`` rather than ``n * m``.

    Unreachable cells carry a large finite sentinel instead of ``inf``. A cell
    whose predecessors are all unreachable would otherwise soft-min ``-inf``
    against itself and make the backward pass non-finite.

    Args:
        cost: Tensor with shape ``(..., n, m)``.
        gamma: Positive soft-min temperature.
        band_radius: Optional Sakoe-Chiba radius in index units.

    Returns:
        Tensor with shape ``cost.shape[:-2]``.
    """
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    if band_radius is not None and band_radius < 0:
        raise ValueError('band_radius must be non-negative or None')

    original_shape = cost.shape[:-2]
    n_rows, n_cols = cost.shape[-2:]
    flat = cost.reshape(-1, n_rows, n_cols)
    n_batch = flat.shape[0]

    # Anti-diagonal d holds R[i, d - i]; indexing by i makes both predecessor
    # diagonals plain one-step shifts of the stored vectors. Skew the cost
    # matrix into that layout once rather than gathering inside the loop.
    rows = torch.arange(n_rows + 1, device=flat.device)
    diags = torch.arange(2, n_rows + n_cols + 1, device=flat.device)
    cols = diags[:, None] - rows[None, :]
    valid = (
        (rows[None, :] >= 1) & (rows[None, :] <= n_rows)
        & (cols >= 1) & (cols <= n_cols)
    )
    if band_radius is not None:
        valid = valid & ((rows[None, :] - cols).abs() <= band_radius)
    skewed = flat[
        :,
        (rows - 1).clamp(0, n_rows - 1).expand(len(diags), -1),
        (cols - 1).clamp(0, n_cols - 1),
    ].permute(1, 0, 2)

    unreachable = flat.new_full((), DTW_UNREACHABLE)
    prev_two = flat.new_full((n_batch, n_rows + 1), DTW_UNREACHABLE)
    prev_two[:, 0] = 0.0  # R[0, 0]
    prev_one = flat.new_full((n_batch, n_rows + 1), DTW_UNREACHABLE)  # R[i,j-1]    the (0,1) move

    for step in range(len(diags)):
        diagonal_step = F.pad(prev_two[:, :-1], (1, 0), value=DTW_UNREACHABLE)  # R[i-1,j-1]  the (1,1) move
        row_step = F.pad(prev_one[:, :-1], (1, 0), value=DTW_UNREACHABLE)  # R[i-1,j]    the (1,0) move
        predecessors = torch.stack([diagonal_step, row_step, prev_one], dim=-1)
        soft_min = -gamma * torch.logsumexp(-predecessors / gamma, dim=-1)
        prev_two, prev_one = prev_one, torch.where(
            valid[step], skewed[step] + soft_min, unreachable,
        )

    return prev_one[:, n_rows].reshape(original_shape)


def expected_path(cost, gamma=0.1, band_radius=None):
    """Return the soft alignment matrix for a batch of local-cost matrices.

    The gradient of soft-DTW with respect to its costs is the path posterior:
    entry ``[a, b]`` is how often a path visits ``(a, b)`` in expectation, so
    row ``a`` describes where source frequency ``a`` is matched.

    Args:
        cost: Tensor with shape ``(..., n, m)``.
        gamma: Positive soft-min temperature.
        band_radius: Optional Sakoe-Chiba radius in index units.

    Returns:
        Tensor with shape ``cost.shape``.
    """
    cost = cost.detach().requires_grad_(True)
    with torch.enable_grad():
        value = soft_dtw(cost, gamma, band_radius)
        alignment, = torch.autograd.grad(value.sum(), cost)
    return alignment


def path_centroids(alignment):
    """Return the mean matched target index for every source index.

    Args:
        alignment: Soft alignment with shape ``(..., n, m)``.

    Returns:
        Tensor with shape ``(..., n)``.
    """
    index = torch.arange(
        alignment.shape[-1], device=alignment.device, dtype=alignment.dtype,
    )
    weight = alignment.sum(dim=-1).clamp_min(1e-9)
    return (alignment * index).sum(dim=-1) / weight


def path_entropy(alignment):
    """Return the mean per-source-row entropy of a soft alignment.

    A diagonal path has near-zero entropy; a diffuse one spreads its mass over
    many target frequencies.
    """
    weight = alignment.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    probability = (alignment / weight).clamp_min(1e-9)
    return -(probability * probability.log()).sum(dim=-1).mean(dim=-1)


def boundary_cost_fraction(alignment, cost):
    """Return the share of expected path cost sitting on the two forced corners.

    Global DTW pins ``(1, 1)`` and ``(K, K)`` into every path, so a large share
    here means the metric is dominated by its endpoints rather than the interior
    spectrum.
    """
    weighted = alignment * cost
    total = weighted.flatten(start_dim=-2).sum(dim=-1).clamp_min(1e-9)
    corners = weighted[..., 0, 0] + weighted[..., -1, -1]
    return corners / total


def cosine_cost(x, y, position_weight=0.0):
    """Construct pairwise signed-cosine costs between frequency sequences."""
    if position_weight < 0:
        raise ValueError('position_weight must be non-negative')
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    cost = 1.0 - torch.einsum('akd,bld->abkl', x, y)
    if position_weight:
        n_x, n_y = x.shape[1], y.shape[1]
        pos_x = torch.arange(n_x, device=x.device, dtype=x.dtype) / max(n_x - 1, 1)
        pos_y = torch.arange(n_y, device=y.device, dtype=y.dtype) / max(n_y - 1, 1)
        penalty = (pos_x[:, None] - pos_y[None, :]).square()
        cost = cost + position_weight * penalty
    return cost


def normalize_spectrum(x):
    """Globally L2-normalize every full frequency representation."""
    x = x.float()
    norm = torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True)
    return x / norm.clamp_min(torch.finfo(x.dtype).eps)


def natural_cost(x, y, position_weight=0.0):
    """Construct pairwise costs that preserve relative coefficient energies."""
    if position_weight < 0:
        raise ValueError('position_weight must be non-negative')
    x = normalize_spectrum(x)
    y = normalize_spectrum(y)
    x_sq = x.square().sum(dim=-1)
    y_sq = y.square().sum(dim=-1)
    dot = torch.einsum('akd,bld->abkl', x, y)
    cost = 0.5 * (x_sq[:, None, :, None] + y_sq[None, :, None, :] - 2.0 * dot)
    cost = cost.clamp_min(0.0)
    if position_weight:
        n_x, n_y = x.shape[1], y.shape[1]
        pos_x = torch.arange(n_x, device=x.device, dtype=x.dtype) / max(n_x - 1, 1)
        pos_y = torch.arange(n_y, device=y.device, dtype=y.dtype) / max(n_y - 1, 1)
        cost = cost + position_weight * (pos_x[:, None] - pos_y[None, :]).square()
    return cost


def paired_cosine_cost(x, y, position_weight=0.0):
    """Construct signed-cosine costs for corresponding rows of two batches.

    Building the full cross-batch tensor and keeping its diagonal would cost a
    factor of the batch size in both time and memory.
    """
    if position_weight < 0:
        raise ValueError('position_weight must be non-negative')
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    cost = 1.0 - torch.einsum('akd,ald->akl', x, y)
    if position_weight:
        n_steps = x.shape[1]
        positions = torch.arange(
            n_steps, device=x.device, dtype=x.dtype,
        ) / max(n_steps - 1, 1)
        cost = cost + position_weight * (
            positions[:, None] - positions[None, :]
        ).square()
    return cost


def paired_natural_cost(x, y, position_weight=0.0):
    """Construct energy-preserving costs for corresponding batch rows."""
    if position_weight < 0:
        raise ValueError('position_weight must be non-negative')
    x = normalize_spectrum(x)
    y = normalize_spectrum(y)
    x_sq = x.square().sum(dim=-1)
    y_sq = y.square().sum(dim=-1)
    dot = torch.einsum('akd,ald->akl', x, y)
    cost = 0.5 * (x_sq[:, :, None] + y_sq[:, None, :] - 2.0 * dot)
    cost = cost.clamp_min(0.0)
    if position_weight:
        n_x, n_y = x.shape[1], y.shape[1]
        pos_x = torch.arange(n_x, device=x.device, dtype=x.dtype) / max(n_x - 1, 1)
        pos_y = torch.arange(n_y, device=y.device, dtype=y.dtype) / max(n_y - 1, 1)
        cost = cost + position_weight * (pos_x[:, None] - pos_y[None, :]).square()
    return cost


def self_cosine_cost(x, position_weight=0.0):
    """Construct within-sequence costs used to debias soft-DTW."""
    x = F.normalize(x.float(), dim=-1)
    cost = 1.0 - torch.einsum('akd,ald->akl', x, x)
    if position_weight:
        n_steps = x.shape[1]
        positions = torch.arange(
            n_steps, device=x.device, dtype=x.dtype,
        ) / max(n_steps - 1, 1)
        cost = cost + position_weight * (
            positions[:, None] - positions[None, :]
        ).square()
    return cost


def frequency_cost(x, y, cost_mode='equal_k', position_weight=0.0):
    """Construct all-pairs local costs under the selected spectral geometry."""
    if cost_mode == 'natural':
        return natural_cost(x, y, position_weight)
    if cost_mode == 'equal_k':
        return cosine_cost(x, y, position_weight)
    raise ValueError(f'Unsupported DTW cost mode: {cost_mode}')


def paired_frequency_cost(x, y, cost_mode='equal_k', position_weight=0.0):
    """Construct paired local costs under the selected spectral geometry."""
    if cost_mode == 'natural':
        return paired_natural_cost(x, y, position_weight)
    if cost_mode == 'equal_k':
        return paired_cosine_cost(x, y, position_weight)
    raise ValueError(f'Unsupported DTW cost mode: {cost_mode}')


def self_frequency_cost(x, cost_mode='equal_k', position_weight=0.0):
    """Construct self-alignment costs under the selected spectral geometry."""
    if cost_mode == 'natural':
        return paired_natural_cost(x, x, position_weight)
    if cost_mode == 'equal_k':
        return self_cosine_cost(x, position_weight)
    raise ValueError(f'Unsupported DTW cost mode: {cost_mode}')


def self_soft_dtw(x, gamma=0.1, band_radius=None, position_weight=0.0, cost_mode='equal_k'):
    """Compute each sequence's own soft-DTW term used to debias the divergence.

    The term depends on one sequence alone, so callers that compare the same
    sequence against many candidates should compute it once and pass it back
    into the divergence helpers.
    """
    return soft_dtw(self_frequency_cost(x, cost_mode, position_weight), gamma, band_radius)


def pairwise_soft_dtw_divergence(x, y, gamma=0.1, band_radius=None,
                                 position_weight=0.0, self_x=None, self_y=None,
                                 cost_mode='equal_k'):
    """Compute normalized soft-DTW divergence between all sequence pairs."""
    if x.shape[1] != y.shape[1]:
        raise ValueError('Frequency soft-DTW requires equal frequency ranges')
    cross_cost = frequency_cost(x, y, cost_mode, position_weight)
    cross = soft_dtw(cross_cost, gamma, band_radius)

    if self_x is None:
        self_x = self_soft_dtw(x, gamma, band_radius, position_weight, cost_mode)
    if self_y is None:
        self_y = self_soft_dtw(y, gamma, band_radius, position_weight, cost_mode)
    divergence = cross - 0.5 * self_x[:, None] - 0.5 * self_y[None, :]
    return divergence if cost_mode == 'natural' else divergence / x.shape[1]


def paired_soft_dtw_divergence(x, y, gamma=0.1, band_radius=None,
                               position_weight=0.0, self_x=None, self_y=None,
                               cost_mode='equal_k'):
    """Compute normalized soft-DTW divergence for corresponding pairs."""
    if x.shape[:2] != y.shape[:2]:
        raise ValueError('Paired soft-DTW requires matching batch and frequency dimensions')
    # print('in paired_soft_dtw_divergence', x.shape, y.shape, gamma, band_radius, position_weight)
    # from IPython import embed
    # embed()  # BREAKPOINT
    cross = soft_dtw(paired_frequency_cost(x, y, cost_mode, position_weight), gamma, band_radius)
    if self_x is None:
        self_x = self_soft_dtw(x, gamma, band_radius, position_weight, cost_mode)
    if self_y is None:
        self_y = self_soft_dtw(y, gamma, band_radius, position_weight, cost_mode)
    divergence = cross - 0.5 * self_x - 0.5 * self_y
    return divergence if cost_mode == 'natural' else divergence / x.shape[1]


def pairwise_rigid_frequency_distance(x, y, cost_mode='equal_k'):
    """Compute same-index distance under the selected spectral geometry."""
    if x.shape[1] != y.shape[1]:
        raise ValueError('Rigid frequency alignment requires equal sequence lengths')
    if cost_mode == 'natural':
        x = normalize_spectrum(x)
        y = normalize_spectrum(y)
        return (1.0 - torch.einsum('akd,bkd->ab', x, y)).clamp_min(0.0)
    if cost_mode != 'equal_k':
        raise ValueError(f'Unsupported DTW cost mode: {cost_mode}')
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    similarity = torch.einsum('akd,bkd->abk', x, y).mean(dim=-1)
    return 1.0 - similarity


def paired_rigid_frequency_distance(x, y, cost_mode='equal_k'):
    """Compute paired same-index distance in the selected spectral geometry."""
    if x.shape[:2] != y.shape[:2]:
        raise ValueError('Paired rigid alignment requires matching dimensions')
    if cost_mode == 'natural':
        x = normalize_spectrum(x)
        y = normalize_spectrum(y)
        return (1.0 - (x * y).sum(dim=(-2, -1))).clamp_min(0.0)
    if cost_mode != 'equal_k':
        raise ValueError(f'Unsupported DTW cost mode: {cost_mode}')
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean(dim=-1)


def contrastive_distance_loss(distance, temperature):
    """Use diagonal parallel pairs as positives in a distance-based InfoNCE loss."""
    if distance.shape[0] != distance.shape[1]:
        raise ValueError('Contrastive distance matrix must be square')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    targets = torch.arange(distance.shape[0], device=distance.device)
    logits = -distance / temperature
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )


def rotating_language_pair(step, n_langs):
    """Select language pairs deterministically and approximately uniformly."""
    pairs = list(combinations(range(n_langs), 2))
    if not pairs:
        raise ValueError('At least two languages are required')
    return pairs[step % len(pairs)]


def frequency_distance_loss(hidden, masks, n_blocks, n_langs, k_indices,
                            mode, temperature, lang_pair, dtw_blocks=4,
                            gamma=0.1, band_radius=None, position_weight=0.0,
                            cost_mode='equal_k'):
    """Compute a matched rigid or soft-DTW contrastive frequency loss.

    Only ``dtw_blocks`` examples and one rotating language pair are used. This
    gives each positive all in-subset negatives while keeping the quadratic DTW
    cost bounded.
    """
    if mode not in DISTANCE_SPECTRAL_MODES:
        raise ValueError(f'Unsupported distance spectral mode: {mode}')
    if dtw_blocks < 2:
        raise ValueError('dtw_blocks must be at least two')
    if mode == 'freq_dtw' and len(k_indices) < 2:
        raise ValueError('freq_dtw requires at least two frequency indices')

    n_selected = min(n_blocks, dtw_blocks)
    lang_a, lang_b = lang_pair
    block_indices = torch.arange(n_selected, device=hidden.device)
    indices_a = block_indices * n_langs + lang_a
    indices_b = block_indices * n_langs + lang_b
    selected = torch.cat([indices_a, indices_b])
    coeffs = spectral_coefficients(
        hidden[selected], masks[selected], k_indices, strict=True,
    )
    coeffs_a = coeffs[:n_selected]
    coeffs_b = coeffs[n_selected:]

    if mode == 'freq_dtw':
        distance = pairwise_soft_dtw_divergence(
            coeffs_a, coeffs_b, gamma, band_radius, position_weight,
            cost_mode=cost_mode,
        )
    else:
        distance = pairwise_rigid_frequency_distance(coeffs_a, coeffs_b, cost_mode)
    return contrastive_distance_loss(distance, temperature)
