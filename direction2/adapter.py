"""
adapter.py — Frequency-domain (DCT) residual adapter for cross-lingual fine-tuning.

FreqAdapter applies a shared rank-r update to selected DCT modes of the hidden states,
or (when spectral=False) a direct hidden-state LoRA update.

Insertion point: between run_prefix (frozen, no_grad) and run_suffix (backbone frozen).
"""
import json
import time

import torch
import torch.nn as nn

from models.encoders import PrismEncoder


class FreqAdapter(nn.Module):
    """Residual DCT adapter for selected frequency modes.

    Modes (mutually exclusive, checked in order):
      spectral=False:  H += H @ U @ V.T  (hidden-state LoRA)
      all_nondc=True:  (U,V) applied to all k=1..T-1 jointly
      gate=True:       F'[:, k] = (1 + g_k) ⊙ F[:, k],  g_k ∈ R^d, init=0
      default:         F[:, k] += V @ (U.T @ F[:, k]) for k in k_indices

    Parameters: gate → d per mode; others → 2*d*rank shared.
    Init: V=0, U~N(0, init_scale), g_k=0  →  identity at start.
    """

    def __init__(self, d, k_indices, rank=16, spectral=True, all_nondc=False,
                 gate=False, init_scale=0.01):
        super().__init__()
        self.d = d
        self.k_indices = list(k_indices) if k_indices else []
        self.rank = rank
        self.spectral = spectral
        self.all_nondc = all_nondc
        self.gate = gate
        if gate:
            self.gates = nn.ParameterList(
                [nn.Parameter(torch.zeros(d)) for _ in self.k_indices])
        else:
            self.U = nn.Parameter(torch.randn(d, rank) * init_scale)
            self.V = nn.Parameter(torch.zeros(d, rank))

    def adapt_sequence(self, H):
        """Apply adapter to a single content-only sequence.

        H: (T, d) — content tokens, no padding.
        Returns H_adapted: (T, d), same dtype as H.
        """
        if not self.spectral:
            return H + H @ self.U @ self.V.T

        T = H.shape[0]
        Fd = PrismEncoder.dct(H.float().T, norm='ortho')  # (d, T)

        if self.all_nondc:
            fks = Fd[:, 1:]                          # (d, T-1)
            deltas = self.V @ (self.U.T @ fks)       # (d, T-1), has grad_fn
            Fd_new = torch.cat([Fd[:, :1], Fd[:, 1:] + deltas], dim=1)
        elif self.gate:
            k_valid = [(i, k) for i, k in enumerate(self.k_indices) if k < T]
            if not k_valid:
                return H
            Fd_new = Fd.clone()
            for i, k in k_valid:
                Fd_new[:, k] = (1.0 + self.gates[i]) * Fd[:, k]
        else:
            k_safe = [k for k in self.k_indices if k < T]
            if not k_safe:
                return H
            k_idx = torch.tensor(k_safe, device=H.device)
            fks = Fd[:, k_idx]                       # (d, len(k_safe))
            deltas = self.V @ (self.U.T @ fks)       # (d, len(k_safe)), has grad_fn
            one_hot = torch.zeros(len(k_safe), T, device=H.device)
            for i, k in enumerate(k_safe):
                one_hot[i, k] = 1.0
            Fd_new = Fd + deltas @ one_hot

        return PrismEncoder.idct(Fd_new, norm='ortho').T.to(H.dtype)

    def forward(self, H_full, content_masks):
        """Apply adapter to a padded batch.

        H_full: (B, T_pad, d) — no grad expected (output of run_prefix).
        content_masks: list of B bool tensors of shape (T_pad,).
        Returns H_adapted: (B, T_pad, d) with grad_fn through U, V.
        """
        rows = []
        for i, mask in enumerate(content_masks):
            H_i = H_full[i]
            idx = mask.nonzero(as_tuple=True)[0]
            H_c = H_i[idx]
            H_c_adapted = self.adapt_sequence(H_c)
            delta_i = torch.zeros_like(H_i)
            delta_i = delta_i.index_put((idx,), H_c_adapted - H_c.detach())
            rows.append(H_i + delta_i)
        return torch.stack(rows)


def run_prefix(model, input_ids, attention_mask, target_layer):
    """Run embedding + first target_layer encoder blocks under no_grad.

    Returns H: (B, T_pad, d) — hidden states after target_layer completed blocks.

    https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_utils.html#ModuleUtilsMixin.get_extended_attention_mask
    Provided a padding mask of dimensions [batch_size, seq_length]
    if the model is a decoder, apply a causal mask in addition to the padding mask
    if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
    """
    with torch.no_grad():
        ext_mask = model.get_extended_attention_mask(attention_mask, input_ids.shape)
        H = model.embeddings(input_ids)
        for layer in model.encoder.layer[:target_layer]:
            H = layer(H, ext_mask)[0]
    return H


def run_suffix(model, H, attention_mask, start_layer):
    """Run encoder blocks from start_layer onward (grad-tracked for adapter backprop).

    Returns H: (B, T_pad, d).
    """
    ext_mask = model.get_extended_attention_mask(attention_mask, H.shape[:2])
    for layer in model.encoder.layer[start_layer:]:
        H = layer(H, ext_mask)[0]
    return H


def mean_pool(H, content_masks):
    """Mean-pool content token positions per sequence.

    H: (B, T_pad, d)
    content_masks: list of B bool tensors of shape (T_pad,).
    Returns: (B, d)
    """
    return torch.stack([H[i, mask].mean(0) for i, mask in enumerate(content_masks)])


def pool(H, content_masks, pooling):
    """Dispatch to cls_pool or mean_pool based on pooling string."""
    if pooling == 'cls':
        return H[:, 0, :]  # cls pooling
    return mean_pool(H, content_masks)


def filter_token_safe(blocks, tok, langs, max_length):
    """Drop blocks where any language tokenises to more than max_length tokens.

    Batches tokenisation per language for speed; one pass per language over all blocks.
    Used by both finetune.py (training) and eval_adapter.py (evaluation).
    """
    keep = [True] * len(blocks)
    for lang in langs:
        texts = [' '.join(b['texts'][lang]) for b in blocks]
        encs = tok(texts, truncation=False, add_special_tokens=True)
        for i, ids in enumerate(encs['input_ids']):
            if len(ids) > max_length:
                keep[i] = False
    return [b for b, k in zip(blocks, keep) if k]


def filter_frequency_safe(blocks, tok, langs, max_k):
    """Keep blocks where every language contains frequency index ``max_k``."""
    if max_k < 0:
        return blocks
    keep = [True] * len(blocks)
    for lang in langs:
        texts = [' '.join(block['texts'][lang]) for block in blocks]
        encoded = tok(texts, truncation=False, add_special_tokens=True, return_special_tokens_mask=True)
        for i, special in enumerate(encoded['special_tokens_mask']):
            if len(special) - sum(special) <= max_k:
                keep[i] = False
    return [block for block, use in zip(blocks, keep) if use]


# Fields that define which experiment a run directory holds. A rerun that
# changes any of them is a different experiment and must not share a directory.
RUN_DEFINING = (
    'model', 'hf_model', 'target_layer', 'main_layer', 'k_indices',
    'spectral_mode', 'lambda_spectral', 'lambda_main', 'lora_r', 'lora_alpha',
    'lora_dropout', 'lora_targets',
    'rank', 'pooling', 'seed', 'max_steps', 'epochs', 'patience', 'max_length',
    'batch_size', 'lr', 'temperature', 'eval_every', 'dtw_gamma',
    'dtw_temperature', 'dtw_blocks', 'dtw_band_radius', 'dtw_position_weight',
    'dtw_cost',
    'manifest', 'train_corpus', 'eval_corpus', 'eval_corpus_hash', 'dev_size',
    'block_filter_k', 'sample_n', 'lambda_id',
)


def archive_run_dir(out_dir):
    """Move a run directory aside and return the archive path.

    Replacing a run in place is what produced mixed directories: the new config
    lands first, and if training then dies the old lora/ and results.json are
    still sitting there describing a different experiment. Moving the whole
    directory in one rename keeps every artefact of a run together.
    """
    stamp = time.strftime('%Y%m%d-%H%M%S')
    archived = out_dir.parent / f'{out_dir.name}.superseded-{stamp}'
    out_dir.rename(archived)
    out_dir.mkdir(parents=True)
    return archived


def guard_run_dir(out_dir, config, overwrite=False):
    """Refuse to write into a directory already holding another run.

    Two ways a directory gets corrupted. A rerun with different settings
    overwrites config.json while results.json and lora/ still describe the
    previous run. A rerun with identical settings is just as destructive,
    because neither trainer resumes -- it restarts from step zero and
    overwrites a finished checkpoint. Both are refused unless the caller opts
    in, and opting in archives rather than overwrites.

    A field named in the incoming config but absent from the stored one counts
    as a difference: the stored run predates that knob, so what it used is
    unknown.
    """
    config_path = out_dir / 'config.json'
    if not config_path.exists():
        return None
    with open(config_path, encoding='utf-8') as f:
        stored = json.load(f)

    changed = []
    for key in RUN_DEFINING:
        if key not in config:
            continue
        if key not in stored:
            changed.append(f'{key}: <absent> -> {config[key]!r}')
        elif stored[key] != config[key]:
            changed.append(f'{key}: {stored[key]!r} -> {config[key]!r}')
    finished = (out_dir / 'results.json').exists()

    if not changed and not finished:
        return None  # interrupted run, same settings: safe to retry in place

    if not overwrite:
        reason = (
            'holds a different experiment (' + ', '.join(changed) + ')'
            if changed else
            'holds a completed run; training restarts from step 0 and would '
            'overwrite results.json and lora/'
        )
        raise SystemExit(
            f'{out_dir} {reason}. Pick a new --out_dir, or pass '
            f'--overwrite_run to archive the existing run and start fresh.'
        )

    archived = archive_run_dir(out_dir)
    print(f'  [archive] previous run moved to {archived}')
    return archived


def split_pair(pair, langs):
    """Split a ``{lang_a}-{lang_b}`` key back into its two codes.

    Language codes may themselves contain hyphens (``aze-Latn``, ``zho-CN``),
    so a naive ``split('-')`` invents extra languages. Existing result CSVs
    store only the joined key, hence this reader; new runs also write
    ``lang_a``/``lang_b`` columns and should use those directly.
    """
    known = set(langs)
    for i, ch in enumerate(pair):
        if ch == '-' and pair[:i] in known and pair[i + 1:] in known:
            return pair[:i], pair[i + 1:]
    raise ValueError(f'cannot split {pair!r} into two known languages')
