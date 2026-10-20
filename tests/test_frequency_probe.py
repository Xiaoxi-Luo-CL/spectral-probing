"""Tests for frequency-probe score fusion and control utilities."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from direction2.probe_frequency_dtw import (
    candidate_indices,
    fused_candidate_distances,
    fusion_condition,
    resolve_fusion,
    row_zscore,
    validate_fusion_source,
)


class FrequencyProbeTest(unittest.TestCase):
    """Check fusion invariants without loading a model."""

    def test_row_zscore_is_per_query(self):
        """Each candidate row should be centered and scaled independently."""
        values = np.array([[1.0, 2.0, 3.0], [10.0, 14.0, 18.0]])
        standardized = row_zscore(values)
        self.assertTrue(np.allclose(standardized.mean(axis=1), 0.0))
        self.assertTrue(np.allclose(standardized.std(axis=1), 1.0))

    def test_zero_alpha_preserves_base_ranking(self):
        """Fusion with alpha zero must exactly recover candidate base ranking."""
        similarity = np.array([[0.9, 0.2, 0.5], [0.1, 0.8, 0.4], [0.3, 0.6, 0.7]])
        candidates = candidate_indices(similarity, 3)
        dtw = np.array([[0.8, 0.1, 0.4], [0.2, 0.7, 0.3], [0.6, 0.5, 0.4]])
        fused = fused_candidate_distances(similarity, candidates, dtw, alpha=0.0)
        ranked = np.take_along_axis(candidates, np.argsort(fused, axis=1), axis=1)
        self.assertTrue(np.array_equal(ranked[:, 0], np.argmax(similarity, axis=1)))

    def test_fusion_condition_is_filename_safe(self):
        """Condition labels should be stable and avoid decimal punctuation."""
        self.assertEqual(fusion_condition(0.05), 'fusion_a0p05')

    def test_fusion_config_loads_frozen_alpha(self):
        """A target run should recover the selected source-corpus alpha."""
        settings = {
            'model': 'mbert', 'main_layer': 8, 'target_layer': 8,
            'k_indices': [1, 2], 'block_filter_k': 64, 'candidates': 32,
            'gamma': 0.1, 'band_radius': -1, 'position_weight': 0.0,
            'coefficient_dtype': 'float16', 'dtw_cost': 'natural',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            path.write_text(json.dumps({
                'metadata': settings,
                'fusion': {'selected_alpha': 0.2},
            }))
            args = SimpleNamespace(fusion_config=str(path), fusion_alpha=None, fusion_alphas=None)
            values, metadata = resolve_fusion(args)
            validate_fusion_source(metadata, settings)
        self.assertEqual(values, [0.2])
        self.assertEqual(metadata['mode'], 'config')
        self.assertEqual(metadata['source'], str(path))

    def test_fusion_config_rejects_different_layer(self):
        """A frozen alpha must not cross incompatible probe configurations."""
        settings = {field: None for field in (
            'model', 'main_layer', 'target_layer', 'k_indices', 'block_filter_k',
            'candidates', 'gamma', 'band_radius', 'position_weight', 'coefficient_dtype', 'dtw_cost',
        )}
        settings['main_layer'] = 16
        fusion = {'mode': 'config', 'source_settings': settings}
        changed = dict(settings, main_layer=24)
        with self.assertRaises(SystemExit):
            validate_fusion_source(fusion, changed)


if __name__ == '__main__':
    unittest.main()
