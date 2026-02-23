"""Tests for latent representations in latents.py.

Focuses on TwoHotDist correctness: stochastic sampling, unimix, KL properties.
"""

import sys
import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dreamerv3.latents import (
    create_latent, OneHotLatent, TwoHotLatent, TwoHotDist,
)


STOCH, CLASSES = 4, 8
B = 2


def _make_logits(seed=0):
    """Random logits with shape (B, STOCH, CLASSES)."""
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(B, STOCH, CLASSES).astype(np.float32))


class TestTwoHotDist:

    def test_sample_is_stochastic(self):
        """Samples with different seeds should differ."""
        logits = _make_logits()
        dist = TwoHotDist(logits, STOCH, CLASSES, unimix=0.01)
        s1 = dist.sample(jax.random.PRNGKey(0))
        s2 = dist.sample(jax.random.PRNGKey(1))
        assert not np.allclose(s1, s2), \
            'TwoHotDist.sample() should produce different results for different seeds'

    def test_sample_shape(self):
        """Sample shape should be (B, stoch, classes)."""
        logits = _make_logits()
        dist = TwoHotDist(logits, STOCH, CLASSES)
        s = dist.sample(jax.random.PRNGKey(0))
        assert s.shape == (B, STOCH, CLASSES)

    def test_sample_sums_to_one(self):
        """Each two-hot vector should sum to ~1 (excluding the ST gradient term)."""
        logits = _make_logits()
        dist = TwoHotDist(logits, STOCH, CLASSES, unimix=0.01)
        s = dist.sample(jax.random.PRNGKey(42))
        # The straight-through term (probs - sg(probs)) sums to 0 in forward,
        # so sample should sum to 1 per (batch, stoch).
        sums = s.sum(-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_unimix_prevents_zero_prob(self):
        """With unimix > 0, no probability should be exactly zero."""
        # Create extreme logits: one class dominates
        logits = jnp.zeros((B, STOCH, CLASSES))
        logits = logits.at[..., 0].set(100.0)
        unimix = 0.01
        dist = TwoHotDist(logits, STOCH, CLASSES, unimix=unimix)
        min_prob = dist._probs.min()
        expected_min = unimix / CLASSES
        assert float(min_prob) >= expected_min * 0.99, \
            f'Min prob {min_prob} should be >= ~{expected_min} with unimix={unimix}'

    def test_kl_nonnegative(self):
        """KL divergence should be >= 0."""
        logits_p = _make_logits(seed=0)
        logits_q = _make_logits(seed=1)
        dist_p = TwoHotDist(logits_p, STOCH, CLASSES, unimix=0.01)
        dist_q = TwoHotDist(logits_q, STOCH, CLASSES, unimix=0.01)
        kl = dist_p.kl(dist_q)
        assert (kl >= -1e-6).all(), f'KL should be non-negative, got min={kl.min()}'

    def test_kl_self_is_zero(self):
        """KL(p || p) should be 0."""
        logits = _make_logits()
        dist = TwoHotDist(logits, STOCH, CLASSES, unimix=0.01)
        kl = dist.kl(dist)
        np.testing.assert_allclose(kl, 0.0, atol=1e-6)

    def test_entropy_nonnegative(self):
        """Entropy should be >= 0."""
        logits = _make_logits()
        dist = TwoHotDist(logits, STOCH, CLASSES, unimix=0.01)
        ent = dist.entropy()
        assert (ent >= -1e-6).all(), f'Entropy should be non-negative, got min={ent.min()}'

    def test_entropy_shape(self):
        """Entropy should have batch shape only (summed over stoch)."""
        logits = _make_logits()
        dist = TwoHotDist(logits, STOCH, CLASSES)
        ent = dist.entropy()
        assert ent.shape == (B,)

    def test_kl_shape(self):
        """KL should have batch shape only (summed over stoch)."""
        logits_p = _make_logits(seed=0)
        logits_q = _make_logits(seed=1)
        dist_p = TwoHotDist(logits_p, STOCH, CLASSES)
        dist_q = TwoHotDist(logits_q, STOCH, CLASSES)
        kl = dist_p.kl(dist_q)
        assert kl.shape == (B,)


class TestTwoHotLatentVsOneHot:
    """Verify TwoHotLatent has compatible interface with OneHotLatent."""

    def test_same_shape(self):
        oh = OneHotLatent(STOCH, CLASSES)
        th = TwoHotLatent(STOCH, CLASSES)
        assert oh.get_shape() == th.get_shape()

    def test_sample_shape_matches(self):
        logits = _make_logits()
        oh = OneHotLatent(STOCH, CLASSES, unimix=0.01)
        th = TwoHotLatent(STOCH, CLASSES, unimix=0.01)
        s_oh = oh.create_dist(logits).sample(jax.random.PRNGKey(0))
        s_th = th.create_dist(logits).sample(jax.random.PRNGKey(0))
        assert s_oh.shape == s_th.shape


class TestCreateLatent:
    """Test the factory function."""

    def test_creates_twohot(self):
        latent = create_latent('twohot', STOCH, CLASSES, unimix=0.01)
        assert isinstance(latent, TwoHotLatent)
        assert latent.unimix == 0.01

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match='Unknown latent type'):
            create_latent('nonexistent', STOCH, CLASSES)
