"""Tests for shared rigid and soft-DTW alignment losses."""
import unittest

import torch

from direction2.alignment import (
    DTW_UNREACHABLE,
    contrastive_distance_loss,
    frequency_distance_loss,
    paired_rigid_frequency_distance,
    paired_soft_dtw_divergence,
    pairwise_rigid_frequency_distance,
    pairwise_soft_dtw_divergence,
    self_soft_dtw,
    soft_dtw,
)


def scalar_soft_dtw(cost, gamma, band_radius=None):
    """Evaluate soft-DTW cell by cell, as a reference for the vectorized form."""
    n_rows, n_cols = cost.shape[-2:]
    flat = cost.reshape(-1, n_rows, n_cols)
    infinite = flat.new_full((flat.shape[0],), float('inf'))
    previous = [flat.new_zeros(flat.shape[0])] + [infinite] * n_cols
    for row in range(1, n_rows + 1):
        current = [infinite]
        for col in range(1, n_cols + 1):
            if band_radius is not None and abs(row - col) > band_radius:
                current.append(infinite)
                continue
            predecessors = torch.stack(
                [previous[col - 1], previous[col], current[col - 1]], dim=-1,
            )
            current.append(
                flat[:, row - 1, col - 1]
                - gamma * torch.logsumexp(-predecessors / gamma, dim=-1)
            )
        previous = current
    return previous[-1].reshape(cost.shape[:-2])


class AlignmentTest(unittest.TestCase):
    """Check numerical behavior and differentiability of alignment losses."""

    def test_diagonal_band_matches_rigid_path_cost(self):
        cost = torch.tensor([
            [0.2, 0.7, 0.9],
            [0.8, 0.3, 0.6],
            [0.9, 0.5, 0.4],
        ])
        value = soft_dtw(cost, gamma=0.2, band_radius=0)
        self.assertTrue(torch.allclose(value, cost.diagonal().sum()))

    def test_soft_dtw_divergence_has_zero_self_diagonal(self):
        torch.manual_seed(7)
        sequence = torch.randn(3, 5, 8)
        distance = pairwise_soft_dtw_divergence(sequence, sequence, gamma=0.1)
        self.assertTrue(torch.allclose(distance.diagonal(), torch.zeros(3), atol=1e-6))

    def test_diagonal_dtw_equals_rigid_frequency_distance(self):
        torch.manual_seed(9)
        source = torch.randn(3, 5, 8)
        target = torch.randn(3, 5, 8)
        dtw = pairwise_soft_dtw_divergence(
            source, target, gamma=0.1, band_radius=0,
        )
        rigid = pairwise_rigid_frequency_distance(source, target)
        self.assertTrue(torch.allclose(dtw, rigid, atol=1e-6))

    def test_natural_diagonal_dtw_equals_flattened_cosine_distance(self):
        torch.manual_seed(10)
        source = torch.randn(3, 5, 8)
        target = torch.randn(3, 5, 8)
        dtw = pairwise_soft_dtw_divergence(
            source, target, gamma=0.02, band_radius=0, cost_mode='natural',
        )
        source_flat = torch.nn.functional.normalize(source.flatten(start_dim=1), dim=-1)
        target_flat = torch.nn.functional.normalize(target.flatten(start_dim=1), dim=-1)
        expected = 1.0 - source_flat @ target_flat.T
        rigid = pairwise_rigid_frequency_distance(source, target, cost_mode='natural')
        self.assertTrue(torch.allclose(dtw, expected, atol=1e-6))
        self.assertTrue(torch.allclose(rigid, expected, atol=1e-6))

    def test_soft_dtw_contrastive_loss_has_finite_gradients(self):
        for cost_mode in ('equal_k', 'natural'):
            torch.manual_seed(11)
            source = torch.randn(3, 4, 6, requires_grad=True)
            target_data = source.detach() + 0.05 * torch.randn(3, 4, 6)
            target = target_data.requires_grad_(True)
            distance = pairwise_soft_dtw_divergence(source, target, gamma=0.2, cost_mode=cost_mode)
            loss = contrastive_distance_loss(distance, temperature=0.1)
            loss.backward()

            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(source.grad).all())
            self.assertTrue(torch.isfinite(target.grad).all())
            self.assertGreater(source.grad.norm().item(), 0.0)
            self.assertGreater(target.grad.norm().item(), 0.0)

    def test_rigid_distance_prefers_parallel_pairs(self):
        torch.manual_seed(13)
        source = torch.randn(4, 3, 7)
        target = source + 0.01 * torch.randn(4, 3, 7)
        distance = pairwise_rigid_frequency_distance(source, target)
        diagonal = distance.diagonal().mean()
        off_diagonal = distance[~torch.eye(4, dtype=torch.bool)].mean()
        self.assertLess(diagonal.item(), off_diagonal.item())

    def test_paired_distances_match_pairwise_diagonal(self):
        torch.manual_seed(15)
        source = torch.randn(4, 5, 7)
        target = torch.randn(4, 5, 7)

        paired_rigid = paired_rigid_frequency_distance(source, target)
        pairwise_rigid = pairwise_rigid_frequency_distance(source, target)
        self.assertTrue(torch.allclose(paired_rigid, pairwise_rigid.diagonal()))

        for cost_mode in ('equal_k', 'natural'):
            paired_dtw = paired_soft_dtw_divergence(source, target, gamma=0.2, cost_mode=cost_mode)
            pairwise_dtw = pairwise_soft_dtw_divergence(source, target, gamma=0.2, cost_mode=cost_mode)
            self.assertTrue(torch.allclose(paired_dtw, pairwise_dtw.diagonal()))

    def test_antidiagonal_recursion_matches_scalar_recursion(self):
        torch.manual_seed(19)
        for n_rows, n_cols in ((3, 3), (4, 7), (9, 2), (1, 1), (16, 16)):
            for gamma in (0.01, 0.1, 1.0):
                for band_radius in (None, 1, 3):
                    cost = torch.rand(4, n_rows, n_cols)
                    vectorized = cost.clone().requires_grad_(True)
                    scalar = cost.clone().requires_grad_(True)
                    got = soft_dtw(vectorized, gamma, band_radius)
                    want = scalar_soft_dtw(scalar, gamma, band_radius)
                    if not torch.isfinite(want).all():
                        continue
                    self.assertTrue(torch.allclose(got, want, atol=1e-4))
                    got.sum().backward()
                    want.sum().backward()
                    self.assertTrue(
                        torch.allclose(vectorized.grad, scalar.grad, atol=1e-4)
                    )

    def test_unreachable_corner_stays_finite(self):
        # A band too narrow to reach the corner must not produce inf or nan,
        # because an all-unreachable soft-min breaks the backward pass.
        cost = torch.rand(2, 6, 2, requires_grad=True)
        value = soft_dtw(cost, gamma=0.1, band_radius=0)
        value.sum().backward()
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.isfinite(cost.grad).all())
        self.assertTrue(bool((value >= DTW_UNREACHABLE * 0.99).all()))

    def test_cached_self_term_matches_inline_computation(self):
        torch.manual_seed(21)
        source = torch.randn(5, 6, 8)
        target = torch.randn(5, 6, 8)
        inline = paired_soft_dtw_divergence(source, target, gamma=0.2)
        cached = paired_soft_dtw_divergence(
            source, target, gamma=0.2,
            self_x=self_soft_dtw(source, gamma=0.2),
            self_y=self_soft_dtw(target, gamma=0.2),
        )
        self.assertTrue(torch.allclose(inline, cached, atol=1e-6))

    def test_monotone_dilation_is_cheaper_than_nonmonotone_shuffle(self):
        # Warping the frequency axis monotonically is what soft-DTW is meant to
        # absorb; an order-breaking permutation must not get the same discount.
        torch.manual_seed(23)
        n_freq = 16
        base = torch.randn(1, n_freq, 32)
        dilated = base[:, torch.arange(n_freq) // 2]
        shuffled = base[:, torch.randperm(n_freq)]

        dilation_gain = (
            pairwise_rigid_frequency_distance(base, dilated)
            - pairwise_soft_dtw_divergence(base, dilated, gamma=0.1)
        ).item()
        shuffle_gain = (
            pairwise_rigid_frequency_distance(base, shuffled)
            - pairwise_soft_dtw_divergence(base, shuffled, gamma=0.1)
        ).item()
        self.assertGreater(dilation_gain, 0.0)
        self.assertGreater(dilation_gain, shuffle_gain)

    def test_frequency_loss_backpropagates_through_dct(self):
        torch.manual_seed(17)
        source = torch.randn(4, 12, 6)
        paired = torch.stack(
            [source, source + 0.05 * torch.randn_like(source)], dim=1,
        )
        hidden = paired.reshape(8, 12, 6).requires_grad_(True)
        masks = torch.ones(8, 12, dtype=torch.bool)
        loss = frequency_distance_loss(
            hidden, masks, n_blocks=4, n_langs=2,
            k_indices=[1, 2, 3, 4], mode='freq_dtw',
            temperature=0.1, lang_pair=(0, 1), dtw_blocks=4,
            gamma=0.2,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(hidden.grad.norm().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
