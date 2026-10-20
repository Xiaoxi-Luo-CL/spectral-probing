"""Tests for label-weighted accuracy across batches of unequal size."""
import unittest

import torch

from models.losses import LabelLoss
from utils.training import weighted_accuracy


class WeightedAccuracyTest(unittest.TestCase):
    """Each batch counts in proportion to the labels it scored, not equally."""

    def test_short_last_batch_does_not_dominate(self):
        """96 correct labels and 4 wrong ones give 0.96, not the 0.5 batch mean."""
        stats = {'accuracy': [1.0, 0.0], 'num_scored': [96, 4]}
        self.assertAlmostEqual(weighted_accuracy(stats), 0.96)

    def test_zero_count_nan_batch_is_ignored(self):
        """A batch that scored nothing (nan accuracy) must not turn the total into nan."""
        stats = {'accuracy': [float('nan'), 0.8], 'num_scored': [0, 10]}
        self.assertAlmostEqual(weighted_accuracy(stats), 0.8)

    def test_no_scored_labels(self):
        """An evaluation that scored nothing reports zero rather than dividing by zero."""
        self.assertEqual(weighted_accuracy({'accuracy': [0.0], 'num_scored': [0]}), 0.0)

    def test_label_loss_counts_only_known_labels(self):
        """Target labels outside the source label set are neither scored nor counted."""
        loss = LabelLoss(['A', 'B'])
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
        acc, count = loss.get_accuracy(logits, ['A', 'A', 'Z'], return_count=True)
        self.assertEqual(count, 2)   # 'Z' is not a source class, so it is not scored
        self.assertAlmostEqual(acc, 0.5)
        self.assertAlmostEqual(loss.get_accuracy(logits, ['A', 'A', 'Z']), 0.5)

    def test_batch_of_unknown_labels_gets_zero_weight(self):
        """A batch with no scorable label must not pull the accuracy towards zero."""
        loss = LabelLoss(['A', 'B'])
        logits = torch.tensor([[[2.0, 0.0]]])
        self.assertEqual(loss.get_accuracy(logits, ['Z'], return_count=True), (0.0, 0))


if __name__ == '__main__':
    unittest.main()
