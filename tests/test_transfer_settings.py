"""Tests for the transfer setting matrix: W in {froze, tune, scratch} x gamma in {none, froze, tune, scratch}, plus plain."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from direction1.transfer.run_batch import reseed
from direction1.transfer.zero_shot import (GRID, PLAIN, SETTINGS, TRAIN_SETTINGS, ZERO_SHOT_SETTINGS, build_setting,
                                           check_plain, needs_training, split_budget, split_setting)
from models.encoders import PrismEncoder
from utils.datasets import LabelledDataset

try:
    from direction1.transfer.analyze_pairs import paired
except ImportError:  # the analysis script needs matplotlib, which not every env has
    paired = None

CLASSES = ['nsubj', 'obj', 'root']
TARGET = ['nsubj', 'obj', 'obl', 'root']
RECIPE = {'frq_init': 'ones', 'frq_noise': 0.1, 'frq_scale': 1.0}


def make_run(root, name, filt, classes=CLASSES, pool='avg', lang='zh'):
    """Write a classify.py-style run directory (config.json + logged arguments) and return its path."""
    exp = os.path.join(root, name)
    os.makedirs(exp)
    cfg = {'lm_name': 'stub', 'language': lang, 'task_name': 'relations', 'filter': filt, **RECIPE}
    with open(os.path.join(exp, 'config.json'), 'w') as f:
        json.dump(cfg, f)
    with open(os.path.join(exp, 'classify.log'), 'w') as f:
        f.write(f"Namespace(pool_word_embed='{pool}', filter='{filt}')\n")
    return exp


def make_ckpt(with_filter, classes=CLASSES):
    """A best.pt-style dict with a linear probe and, optionally, raw filter logits."""
    ckpt = {'language_model_name': 'stub', 'classes': classes, 'classifier': nn.Linear(4, len(classes))}
    if with_filter:
        ckpt['frequency_filter'] = nn.Parameter(torch.linspace(-1.0, 2.0, 512))
    return ckpt


class SettingTest(unittest.TestCase):
    """Each setting gets the probe and filter its name describes.

    Classifiers move themselves to CUDA when one is visible, so tensors are compared on CPU.
    """

    def setUp(self):
        """A filtered source run, its filter-free twin, and a stub LM."""
        self.tmp = tempfile.TemporaryDirectory()
        self.src_exp = make_run(self.tmp.name, 'source', 'auto(512)')
        self.plain_exp = make_run(self.tmp.name, 'plain', 'nofilter()')
        self.src, self.plain = make_ckpt(True), make_ckpt(False)
        lm = SimpleNamespace(config=SimpleNamespace(hidden_size=4, num_hidden_layers=1))
        self.proto = SimpleNamespace(_lm=lm, _tok=None)

    def tearDown(self):
        """Remove the temporary run directories."""
        self.tmp.cleanup()

    def build(self, name):
        """Classifier and classes for one setting."""
        return build_setting(name, 'stub', self.src, self.src_exp, 'avg', plain=self.plain, proto_encoder=self.proto)

    def test_partition(self):
        """Zero-shot and training settings split the 13 settings exactly, with no repeats."""
        self.assertEqual(len(GRID), 12)
        self.assertEqual(sorted(SETTINGS), sorted(GRID + [PLAIN]))
        self.assertEqual(set(ZERO_SHOT_SETTINGS) & set(TRAIN_SETTINGS), set())
        self.assertTrue(all(not needs_training(s) for s in ZERO_SHOT_SETTINGS))
        self.assertTrue(all(needs_training(s) for s in TRAIN_SETTINGS))

    def test_matrix(self):
        """Probe and filter trainability, initial values and gating follow the setting name."""
        P = self.src['frequency_filter'].detach()
        for name in GRID:
            w, g = split_setting(name)
            torch.manual_seed(0)
            clf, classes = self.build(name)
            with self.subTest(name):
                self.assertEqual(classes, CLASSES)
                lbl = list(clf._lbl.parameters())
                self.assertTrue(all(p.requires_grad == (w != 'froze') for p in lbl))
                if w != 'scratch':
                    self.assertTrue(torch.equal(clf._lbl.weight.detach().cpu(), self.src['classifier'].weight))
                filt = clf._emb._frq_filter
                if g == 'none':
                    self.assertIsNone(filt)
                    continue
                self.assertTrue(clf._emb._frq_logits)  # learned filters always pass through the sigmoid
                self.assertEqual(clf._emb._frq_tuning, g in ('tune', 'scratch'))
                if g in ('froze', 'tune'):
                    self.assertTrue(torch.equal(filt.detach().cpu(), P))
                else:
                    torch.manual_seed(0)
                    expected = PrismEncoder.gen_auto_filter_initialization(512, 'ones', 0.1, 1.0)
                    self.assertTrue(torch.equal(filt.detach().cpu(), expected))

    def test_scratch_uses_source_recipe(self):
        """A scratch filter starts near-flat (gates ~0.73), not from the noisy function defaults."""
        clf, _ = self.build('Wscratch_Gscratch')
        gates = torch.sigmoid(clf._emb._frq_filter.detach().cpu())
        self.assertAlmostEqual(gates.mean().item(), 0.73, delta=0.02)
        self.assertLess(gates.std().item(), 0.05)

    def test_checkpoints_are_copied(self):
        """Training or freezing a setting never touches the loaded source checkpoint."""
        before = self.src['classifier'].weight.detach().clone()
        clf, _ = self.build('Wtune_Gtune')
        with torch.no_grad():
            clf._lbl.weight.add_(1.0)
            clf._emb._frq_filter.add_(1.0)
        self.assertTrue(torch.equal(self.src['classifier'].weight, before))
        self.assertTrue(torch.equal(self.src['frequency_filter'].detach(), torch.linspace(-1.0, 2.0, 512)))
        self.build('Wfroze_Gnone')
        self.assertTrue(all(p.requires_grad for p in self.src['classifier'].parameters()))

    def test_plain(self):
        """The plain setting uses the filter-free probe, frozen, with no filter."""
        clf, classes = self.build(PLAIN)
        self.assertIsNone(clf._emb._frq_filter)
        self.assertTrue(torch.equal(clf._lbl.weight.detach().cpu(), self.plain['classifier'].weight))
        self.assertFalse(any(p.requires_grad for p in clf._lbl.parameters()))
        self.assertEqual(classes, CLASSES)
        check_plain(self.plain, self.plain_exp, self.src, self.src_exp)

    def test_plain_is_validated(self):
        """A plain probe with a filter, other pooling, or other labels is rejected."""
        with self.assertRaises(AssertionError):
            check_plain(make_ckpt(True), self.plain_exp, self.src, self.src_exp)
        with self.assertRaises(AssertionError):
            check_plain(self.plain, self.src_exp, self.src, self.src_exp)  # trained with a filter
        other_pool = make_run(self.tmp.name, 'plain_last', 'nofilter()', pool='last')
        with self.assertRaises(AssertionError):
            check_plain(self.plain, other_pool, self.src, self.src_exp)
        with self.assertRaises(AssertionError):
            check_plain(make_ckpt(False, ['nsubj', 'root']), self.plain_exp, self.src, self.src_exp)

    def test_missing_plain_is_skipped(self):
        """Without a filter-free probe, plain raises the AssertionError the batch runner skips on."""
        with self.assertRaises(AssertionError):
            build_setting(PLAIN, 'stub', self.src, self.src_exp, 'avg', plain=None, proto_encoder=self.proto)

    def test_target_labels(self):
        """A fresh probe takes the given label set; a probe taken from the source keeps the source labels."""
        for name in GRID:
            w, _ = split_setting(name)
            clf, classes = build_setting(name, 'stub', self.src, self.src_exp, 'avg', proto_encoder=self.proto,
                                         classes=TARGET)
            with self.subTest(name):
                self.assertEqual(classes, TARGET if w == 'scratch' else CLASSES)
                self.assertEqual(clf._lbl.out_features, len(classes))

    def test_reseed_same_probe(self):
        """After reseeding, settings with and without a filter start from the same fresh probe."""
        weights = []
        for name in ('Wscratch_Gnone', 'Wscratch_Gfroze'):
            reseed(7)
            clf, _ = self.build(name)
            weights.append(clf._lbl.weight.detach().cpu())
        self.assertTrue(torch.equal(*weights))


class BudgetTest(unittest.TestCase):
    """The fixed-budget split keeps the pool first and leaves enough, but not too many, test sentences."""

    def split(self, total, test_tail=0):
        """Pool and test sizes for a dataset of total sentences, plus the first test sentence."""
        data = LabelledDataset([f's{i}' for i in range(total)], [['root'] for _ in range(total)])
        pool, test = split_budget(data, pool_size=200, min_test=100, max_test=2000, test_tail=test_tail)
        return pool.get_input_count(), test.get_input_count(), test._inputs[0]

    def test_sizes(self):
        """Small targets shrink the pool, mid ones keep everything else as test, large ones are capped."""
        self.assertEqual(self.split(260), (160, 100, 's160'))   # sq-sized
        self.assertEqual(self.split(357), (200, 157, 's200'))   # bho-sized
        self.assertEqual(self.split(5000), (200, 2000, 's200'))

    def test_official_test(self):
        """A treebank with a train split keeps its official test (the file tail) and pools from train."""
        self.assertEqual(self.split(1522, test_tail=613), (200, 613, 's909'))   # oc: train 830 + dev 79, test 613
        self.assertEqual(self.split(1000, test_tail=500), (200, 500, 's500'))   # nds: train 500, test 500


@unittest.skipIf(paired is None, 'analyze_pairs needs matplotlib')
class PairAnalysisTest(unittest.TestCase):
    """Related-minus-unrelated gains are paired by seed, so a seed missing on one side is dropped."""

    def test_shared_seeds_only(self):
        """Only seeds both sources have enter the difference."""
        rel, unrel = {42: 1.0, 43: 3.0, 44: 5.0}, {42: 0.0, 43: 1.0}
        self.assertEqual(paired(rel, unrel).tolist(), [1.0, 2.0])


if __name__ == '__main__':
    unittest.main()
