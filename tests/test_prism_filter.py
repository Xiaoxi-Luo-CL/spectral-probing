"""Tests for how PrismEncoder gates DCT coefficients with learned and fixed filters."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from models.encoders import PrismEncoder


def make_encoder(frq_filter, frq_tuning, frq_logits=None):
    """Build an encoder around a stub LM; filter() never touches the LM."""
    lm = SimpleNamespace(config=SimpleNamespace(hidden_size=4, num_hidden_layers=1))
    return PrismEncoder.from_lm(lm=lm, tok=None, lm_name='stub', frq_filter=frq_filter.clone(),
                                frq_tuning=frq_tuning, pool_word_embed=None, frq_logits=frq_logits)


class FilterGateTest(unittest.TestCase):
    """A learned filter is gated through the sigmoid whether or not it is still trained."""

    def setUp(self):
        """Two sequences, the second padded, and a 512-entry filter of logits."""
        torch.manual_seed(0)
        self.emb = torch.randn(2, 9, 4)
        self.att = torch.tensor([[1] * 9, [1] * 6 + [0] * 3], dtype=torch.bool)
        self.logits = torch.linspace(-2.0, 4.0, 512)

    def test_frozen_learned_filter_matches_training_gate(self):
        """Freezing a learned filter must not change what it does to the input."""
        trained = make_encoder(self.logits, frq_tuning=True).filter(self.emb, self.att)
        frozen = make_encoder(self.logits, frq_tuning=False, frq_logits=True).filter(self.emb, self.att)
        self.assertTrue(torch.allclose(frozen, trained))

    def test_raw_logits_are_not_the_trained_filter(self):
        """Guards the transfer bug: multiplying by the logits directly is a different filter."""
        trained = make_encoder(self.logits, frq_tuning=True).filter(self.emb, self.att)
        raw = make_encoder(self.logits, frq_tuning=False, frq_logits=False).filter(self.emb, self.att)
        self.assertFalse(torch.allclose(raw, trained))

    def test_band_mask_is_applied_as_is(self):
        """An all-pass band mask reproduces the input; a sigmoid would scale it by 0.73."""
        out = make_encoder(torch.ones(512), frq_tuning=False).filter(self.emb, self.att)
        self.assertTrue(torch.allclose(out, self.emb * self.att[..., None], atol=1e-5))

    def test_default_follows_tuning(self):
        """Without frq_logits, callers keep the old behaviour."""
        self.assertTrue(make_encoder(self.logits, frq_tuning=True)._frq_logits)
        self.assertFalse(make_encoder(self.logits, frq_tuning=False)._frq_logits)


class LoadTest(unittest.TestCase):
    """PrismEncoder.load treats a filter read from a checkpoint as a learned logit."""

    def load_kwargs(self, objects, **load_args):
        """Save objects as a checkpoint and return the arguments load() builds the encoder with."""
        seen = {}

        def record(enc, **kwargs):
            nn.Module.__init__(enc)
            enc._lm = None
            seen.update(kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'best.pt')
            torch.save(objects, path)
            # stub __init__ so no language model is loaded
            with mock.patch.object(PrismEncoder, '__init__', autospec=True, side_effect=record):
                PrismEncoder.load(path, **load_args)
        return seen

    def test_checkpoint_filter_is_logits_even_when_frozen(self):
        """A frozen filter from a checkpoint must still go through the sigmoid."""
        seen = self.load_kwargs({'language_model_name': 'stub', 'frequency_filter': torch.zeros(512)},
                                frq_tuning=False)
        self.assertTrue(seen['frq_logits'])

    def test_caller_mask_keeps_old_behaviour(self):
        """With no filter in the checkpoint, a caller's band mask follows frq_tuning as before."""
        seen = self.load_kwargs({'language_model_name': 'stub'}, frq_filter=torch.ones(512), frq_tuning=False)
        self.assertIsNone(seen['frq_logits'])


if __name__ == '__main__':
    unittest.main()
