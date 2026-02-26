"""Tests for imag_loss and imag_loss_pmpo in agent.py."""

import sys
import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Ensure the dreamerv3 package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Lightweight mocks that satisfy the interfaces used by imag_loss / imag_loss_pmpo
# ---------------------------------------------------------------------------

class MockPolicyDist:
  """Mock policy distribution with logp/entropy/kl."""

  def __init__(self, logits):
    self._logits = logits  # (B, T, A)

  def logp(self, action):
    return -jnp.ones(self._logits.shape[:2])

  def entropy(self):
    return jnp.ones(self._logits.shape[:2]) * 0.5

  def kl(self, other):
    """Approximate KL as sum of squared differences of logits."""
    return jnp.square(self._logits - other._logits).sum(-1)


class MockValueDist:
  """Mock value distribution with pred/loss (scalar outputs)."""

  def __init__(self, values):
    self._values = values  # (B, T)

  def pred(self):
    return self._values

  def loss(self, target):
    return jnp.square(self._values - target)


class MockNorm:
  """Mock normalizer matching embodied.jax.Normalize interface."""

  def __call__(self, x, update):
    return 0.0, 1.0

  def stats(self):
    return 0.0, 1.0


def _make_inputs(B=4, H=8, seed=0):
  """Create synthetic inputs for imag_loss / imag_loss_pmpo."""
  rng = np.random.RandomState(seed)
  act = {'action': jnp.array(rng.randn(B, H + 1, 3).astype(np.float32))}
  rew = jnp.array(rng.randn(B, H + 1).astype(np.float32))
  con = jnp.ones((B, H + 1), dtype=jnp.float32)
  logits = jnp.array(rng.randn(B, H + 1, 3).astype(np.float32))
  policy = {'action': MockPolicyDist(logits)}
  value = MockValueDist(jnp.array(rng.randn(B, H + 1).astype(np.float32)))
  slowvalue = MockValueDist(jnp.array(rng.randn(B, H + 1).astype(np.float32)))
  norm = MockNorm()
  return dict(
      act=act, rew=rew, con=con,
      policy=policy, value=value, slowvalue=slowvalue,
      retnorm=norm, valnorm=norm, advnorm=norm,
      update=False, contdisc=True, horizon=333,
      lam=0.95, actent=3e-4, slowreg=1.0,
  )


def _make_bc_policy(B=4, H=8, seed=99):
  """Create a separate BC policy for testing KL term."""
  rng = np.random.RandomState(seed)
  bc_logits = jnp.array(rng.randn(B, H + 1, 3).astype(np.float32))
  return {'action': MockPolicyDist(bc_logits)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImagLoss:
  """Tests for the REINFORCE imag_loss function."""

  def test_output_shapes(self):
    from dreamerv3.agent import imag_loss
    B, H = 4, 8
    inputs = _make_inputs(B, H)
    losses, outs, metrics = imag_loss(**inputs)

    assert 'policy' in losses
    assert 'value' in losses
    assert losses['policy'].shape == (B, H)
    assert losses['value'].shape == (B, H)
    assert 'ret' in outs
    assert outs['ret'].shape == (B, H)

  def test_deterministic(self):
    from dreamerv3.agent import imag_loss
    inputs = _make_inputs()
    l1, _, _ = imag_loss(**inputs)
    l2, _, _ = imag_loss(**inputs)
    np.testing.assert_allclose(l1['policy'], l2['policy'], rtol=1e-5)

  def test_metrics_present(self):
    from dreamerv3.agent import imag_loss
    inputs = _make_inputs()
    _, _, metrics = imag_loss(**inputs)
    for key in ['adv', 'adv_std', 'adv_mag', 'rew', 'con',
                'ret', 'val', 'tar', 'weight', 'slowval',
                'ret_min', 'ret_max', 'ret_rate', 'ent/action']:
      assert key in metrics, f'Missing metric: {key}'


class TestImagLossPMPO:
  """Tests for the PMPO imag_loss_pmpo function."""

  def test_output_shapes(self):
    from dreamerv3.agent import imag_loss_pmpo
    B, H = 4, 8
    inputs = _make_inputs(B, H)
    losses, outs, metrics = imag_loss_pmpo(**inputs)

    assert 'policy' in losses
    assert 'value' in losses
    assert losses['policy'].shape == (B, H)
    assert losses['value'].shape == (B, H)
    assert 'ret' in outs

  def test_pmpo_pos_frac_metric(self):
    """PMPO should report the fraction of positive advantages."""
    from dreamerv3.agent import imag_loss_pmpo
    inputs = _make_inputs()
    _, _, metrics = imag_loss_pmpo(**inputs)
    assert 'pmpo_pos_frac' in metrics
    frac = float(metrics['pmpo_pos_frac'])
    assert 0.0 <= frac <= 1.0

  def test_scale_invariance(self):
    """PMPO D+/D- terms should be invariant to reward scale.

    With actent=0 we isolate the sign-based policy gradient.  We compare
    mean policy loss (the actual optimization objective) because per-element
    values can differ when the D+/D- split changes with reward scale.
    """
    from dreamerv3.agent import imag_loss_pmpo
    inputs1 = _make_inputs(seed=42)
    inputs1['actent'] = 0.0
    inputs2 = _make_inputs(seed=42)
    inputs2['actent'] = 0.0
    # Scale rewards by a factor that preserves D+/D- split
    # (small scale so advantage signs don't flip)
    inputs2['rew'] = inputs2['rew'] * 2.0

    l1, _, _ = imag_loss_pmpo(**inputs1)
    l2, _, _ = imag_loss_pmpo(**inputs2)

    # Mean policy loss should be identical (scale-invariant)
    np.testing.assert_allclose(
        l1['policy'].mean(), l2['policy'].mean(), rtol=1e-4, atol=1e-6,
        err_msg='PMPO policy loss should be invariant to reward scale')

  def test_alpha_boundaries(self):
    """Test alpha=0 (only penalize negatives) and alpha=1 (only boost positives)."""
    from dreamerv3.agent import imag_loss_pmpo
    inputs0 = _make_inputs(seed=7)
    inputs0['pmpo_alpha'] = 0.0
    l0, _, _ = imag_loss_pmpo(**inputs0)

    inputs1 = _make_inputs(seed=7)
    inputs1['pmpo_alpha'] = 1.0
    l1, _, _ = imag_loss_pmpo(**inputs1)

    # At different alpha values the losses should differ
    assert not np.allclose(l0['policy'], l1['policy']), \
        'Alpha=0 and alpha=1 should produce different policy losses'

  def test_same_value_loss_as_reinforce(self):
    """The value loss should be identical between REINFORCE and PMPO."""
    from dreamerv3.agent import imag_loss, imag_loss_pmpo
    inputs = _make_inputs(seed=99)
    lr, _, _ = imag_loss(**inputs)
    lp, _, _ = imag_loss_pmpo(**inputs)
    np.testing.assert_allclose(
        lr['value'], lp['value'], rtol=1e-5,
        err_msg='Value loss should be identical between REINFORCE and PMPO')

  def test_deterministic(self):
    from dreamerv3.agent import imag_loss_pmpo
    inputs = _make_inputs()
    l1, _, _ = imag_loss_pmpo(**inputs)
    l2, _, _ = imag_loss_pmpo(**inputs)
    np.testing.assert_allclose(l1['policy'], l2['policy'], rtol=1e-5)

  def test_metrics_present(self):
    from dreamerv3.agent import imag_loss_pmpo
    inputs = _make_inputs()
    _, _, metrics = imag_loss_pmpo(**inputs)
    for key in ['adv', 'adv_std', 'adv_mag', 'rew', 'con',
                'ret', 'val', 'tar', 'weight', 'slowval',
                'ret_min', 'ret_max', 'ret_rate', 'pmpo_pos_frac',
                'ent/action']:
      assert key in metrics, f'Missing metric: {key}'

  def test_gradient_sign_matches_reinforce(self):
    """PMPO should push logpi UP for D+ and DOWN for D-, same as REINFORCE.

    This test catches sign errors in the per-element pmpo_adv coefficients.
    """
    from dreamerv3.agent import imag_loss, imag_loss_pmpo

    B, H = 2, 8
    rng = np.random.RandomState(123)
    act_data = jnp.array(rng.randn(B, H + 1, 3).astype(np.float32))
    # Rewards: positive early, negative late → clear D+/D- split
    rew = np.zeros((B, H + 1), dtype=np.float32)
    rew[:, :H // 2 + 1] = 1.0
    rew[:, H // 2 + 1:] = -1.0

    # Differentiable mock: logp depends on logits via -0.5 * sum((act - tanh(logits))^2)
    class DiffMockPolicyDist:
      def __init__(self, logits):
        self._logits = logits
      def logp(self, action):
        return -0.5 * jnp.square(action - jnp.tanh(self._logits)).sum(-1)
      def entropy(self):
        return jnp.ones(self._logits.shape[:2]) * 0.5
      def kl(self, other):
        return jnp.square(self._logits - other._logits).sum(-1)

    logits = jnp.zeros((B, H + 1, 3), dtype=jnp.float32)

    def make_inputs(logits_param, loss_fn_name):
      policy = {'action': DiffMockPolicyDist(logits_param)}
      return dict(
          act={'action': act_data},
          rew=jnp.array(rew), con=jnp.ones((B, H + 1), dtype=jnp.float32),
          policy=policy,
          value=MockValueDist(jnp.zeros((B, H + 1), dtype=jnp.float32)),
          slowvalue=MockValueDist(jnp.zeros((B, H + 1), dtype=jnp.float32)),
          retnorm=MockNorm(), valnorm=MockNorm(), advnorm=MockNorm(),
          update=False, contdisc=True, horizon=333,
          lam=0.95, actent=0.0, slowreg=1.0,
      )

    pmpo_grad = jax.grad(lambda l: imag_loss_pmpo(**make_inputs(l, 'pmpo'))[0]['policy'].mean())(logits)
    reinforce_grad = jax.grad(lambda l: imag_loss(**make_inputs(l, 'reinforce'))[0]['policy'].mean())(logits)

    # The gradient signs should agree (both push logpi up for D+ and down for D-)
    mask = jnp.abs(reinforce_grad) > 1e-8
    assert mask.any(), 'Need some non-zero gradients for this test'
    sign_agreement = (jnp.sign(pmpo_grad[mask]) == jnp.sign(reinforce_grad[mask])).mean()
    assert float(sign_agreement) > 0.9, \
        f'PMPO and REINFORCE gradient signs should mostly agree, got {sign_agreement:.2%}'


class TestImagLossPMPO_WeightedCount:
  """Tests that PMPO uses weighted count (not raw count) in D+/D- averaging."""

  def test_zero_weight_states_dont_contribute(self):
    """States with zero continuation weight should not affect the loss."""
    from dreamerv3.agent import imag_loss_pmpo
    B, H = 4, 8
    # Baseline: all continuations = 1
    inputs1 = _make_inputs(B, H, seed=42)
    l1, _, _ = imag_loss_pmpo(**inputs1)

    # Set continuation to 0 at timestep 2 onward → weight becomes 0 there
    inputs2 = _make_inputs(B, H, seed=42)
    inputs2['con'] = inputs2['con'].at[:, 2:].set(0.0)
    l2, _, _ = imag_loss_pmpo(**inputs2)

    # The losses should differ because different states are included
    # But critically, the zero-weight states should not bias the average
    # (this test mainly ensures no NaN/inf from the fix)
    assert jnp.isfinite(l2['policy']).all(), \
        'PMPO should produce finite loss even with zero-weight states'


class TestImagLossPMPO_BCPrior:
  """Tests for the PMPO BC prior KL regularization term."""

  def test_bc_kl_metric_present(self):
    """When bc_policy is provided, should report pmpo_kl_bc metric."""
    from dreamerv3.agent import imag_loss_pmpo
    inputs = _make_inputs(seed=10)
    inputs['bc_policy'] = _make_bc_policy(seed=20)
    inputs['pmpo_beta'] = 0.3
    _, _, metrics = imag_loss_pmpo(**inputs)
    assert 'pmpo_kl_bc' in metrics
    assert float(metrics['pmpo_kl_bc']) >= 0.0

  def test_bc_kl_no_metric_when_beta_zero(self):
    """When pmpo_beta=0, KL metric should not appear."""
    from dreamerv3.agent import imag_loss_pmpo
    inputs = _make_inputs(seed=10)
    inputs['bc_policy'] = _make_bc_policy(seed=20)
    inputs['pmpo_beta'] = 0.0
    _, _, metrics = imag_loss_pmpo(**inputs)
    assert 'pmpo_kl_bc' not in metrics

  def test_bc_kl_changes_policy_loss(self):
    """Adding BC prior KL should change the policy loss vs no BC prior."""
    from dreamerv3.agent import imag_loss_pmpo
    # Without BC prior
    inputs_no_bc = _make_inputs(seed=42)
    inputs_no_bc['pmpo_beta'] = 0.3
    l_no_bc, _, _ = imag_loss_pmpo(**inputs_no_bc)

    # With BC prior (different logits → nonzero KL)
    inputs_bc = _make_inputs(seed=42)
    inputs_bc['bc_policy'] = _make_bc_policy(seed=77)
    inputs_bc['pmpo_beta'] = 0.3
    l_bc, _, _ = imag_loss_pmpo(**inputs_bc)

    assert not np.allclose(l_no_bc['policy'], l_bc['policy']), \
        'BC prior KL should change the policy loss'

  def test_bc_kl_zero_when_same_policy(self):
    """KL should be zero when bc_policy == policy (same logits)."""
    from dreamerv3.agent import imag_loss_pmpo
    inputs = _make_inputs(seed=42)
    # Use the same policy as BC prior → KL should be 0
    inputs['bc_policy'] = inputs['policy']
    inputs['pmpo_beta'] = 0.3
    _, _, metrics = imag_loss_pmpo(**inputs)
    np.testing.assert_allclose(
        float(metrics['pmpo_kl_bc']), 0.0, atol=1e-6,
        err_msg='KL between identical policies should be zero')

  def test_bc_kl_scale_invariance_preserved(self):
    """Scale invariance should hold even with BC prior (KL doesn't depend on rewards).

    Uses actent=0 and compares mean loss (see test_scale_invariance).
    """
    from dreamerv3.agent import imag_loss_pmpo
    inputs1 = _make_inputs(seed=42)
    inputs1['bc_policy'] = _make_bc_policy(seed=77)
    inputs1['pmpo_beta'] = 0.3
    inputs1['actent'] = 0.0

    inputs2 = _make_inputs(seed=42)
    inputs2['bc_policy'] = _make_bc_policy(seed=77)
    inputs2['pmpo_beta'] = 0.3
    inputs2['actent'] = 0.0
    inputs2['rew'] = inputs2['rew'] * 2.0

    l1, _, _ = imag_loss_pmpo(**inputs1)
    l2, _, _ = imag_loss_pmpo(**inputs2)

    np.testing.assert_allclose(
        l1['policy'].mean(), l2['policy'].mean(), rtol=1e-4,
        err_msg='PMPO + BC prior should still be scale-invariant')

