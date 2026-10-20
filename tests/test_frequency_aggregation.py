"""Tests for low-frequency coefficient aggregation."""
import unittest

import numpy as np

from direction2.probe_frequency_aggregation import aggregate_vectors


class FrequencyAggregationTest(unittest.TestCase):
    """Check aggregation semantics without loading a model."""

    def test_natural_mean_averages_selected_coefficients(self):
        """Natural averaging should preserve energy before final normalization."""
        coeffs = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])
        actual = aggregate_vectors(coeffs, 1, 3, 'mean', 'natural')
        expected = np.array([[2.0, 1.0]]) / np.sqrt(5.0)
        self.assertTrue(np.allclose(actual, expected))

    def test_equal_k_mean_removes_coefficient_scale(self):
        """Equal-k averaging should give each selected coefficient equal weight."""
        coeffs = np.array([[[0.0, 0.0], [20.0, 0.0], [0.0, 1.0]]])
        actual = aggregate_vectors(coeffs, 1, 3, 'mean', 'equal_k')
        expected = np.array([[1.0, 1.0]]) / np.sqrt(2.0)
        self.assertTrue(np.allclose(actual, expected))

    def test_single_coefficient_mean_matches_concatenation(self):
        """All aggregation choices should coincide when only one k is selected."""
        coeffs = np.array([
            [[0.0, 0.0], [2.0, 1.0]],
            [[0.0, 0.0], [-1.0, 3.0]],
        ])
        mean = aggregate_vectors(coeffs, 1, 2, 'mean', 'natural')
        concatenated = aggregate_vectors(coeffs, 1, 2, 'concatenate', 'natural')
        self.assertTrue(np.allclose(mean, concatenated))

    def test_unknown_aggregation_is_rejected(self):
        """Typos should fail instead of silently changing the representation."""
        coeffs = np.zeros((1, 2, 2))
        with self.assertRaises(ValueError):
            aggregate_vectors(coeffs, 1, 2, 'sum', 'natural')


if __name__ == '__main__':
    unittest.main()
