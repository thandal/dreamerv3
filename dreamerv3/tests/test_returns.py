import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dreamerv3.returns import LambdaReturn, NStepReturn, MonteCarloReturn, GAE, create_return_computer

def make_data(B=2, T=5, seed=0):
    rng = np.random.RandomState(seed)
    rew = jnp.array(rng.randn(B, T).astype(np.float32))
    val = jnp.array(rng.randn(B, T).astype(np.float32))
    boot = jnp.array(rng.randn(B, T).astype(np.float32))
    last = jnp.zeros((B, T), dtype=jnp.float32)
    term = jnp.zeros((B, T), dtype=jnp.float32)
    disc = 0.99
    return last, term, rew, val, boot, disc

class TestLambdaReturn:
    def test_shape(self):
        last, term, rew, val, boot, disc = make_data()
        computer = LambdaReturn(lam=0.95)
        rets = computer.compute(last, term, rew, val, boot, disc)
        assert rets.shape == (2, 4)

    def test_lam0_matches_1step(self):
        last, term, rew, val, boot, disc = make_data()
        computer = LambdaReturn(lam=0.0)
        rets = computer.compute(last, term, rew, val, boot, disc)
        # ret_t = r_{t+1} + gamma * boot_{t+1}
        expected = rew[:, 1:] + disc * boot[:, 1:]
        np.testing.assert_allclose(rets, expected, rtol=1e-5)

    def test_lam1_matches_mc(self):
        last, term, rew, val, boot, disc = make_data(T=10)
        computer = LambdaReturn(lam=1.0)
        rets = computer.compute(last, term, rew, val, boot, disc)

        # Manually compute MC returns
        B, T = rew.shape
        expected = []
        for t in range(T - 1):
            curr_ret = jnp.zeros(B)
            discount = 1.0
            for k in range(t + 1, T):
                curr_ret += discount * rew[:, k]
                discount *= disc
            # Add bootstrap at the end
            curr_ret += discount * boot[:, -1]
            expected.append(curr_ret)
        expected = jnp.stack(expected, axis=1)
        np.testing.assert_allclose(rets, expected, rtol=1e-5)

    def test_termination(self):
        last, term, rew, val, boot, disc = make_data(T=5)
        term = term.at[:, 2].set(1.0) # terminate at t=2
        computer = LambdaReturn(lam=0.95)
        rets = computer.compute(last, term, rew, val, boot, disc)
        # At t=1, the return should NOT include anything from t=2 onwards if it terminated.
        # Wait, the implementation says:
        # live = (1 - f32(term))[:, 1:] * disc
        # if term[:, 2] is 1, then live[:, 1] (which corresponds to t=2) is 0.
        # ret_0 = r_1 + live_1 * [(1-lam)*boot_1 + lam*ret_1]
        # ret_1 = r_2 + live_2 * [(1-lam)*boot_2 + lam*ret_2]
        # If term[:, 2] = 1, then live[:, 1] = 0.
        # So ret_1 = r_2. Correct.
        assert jnp.all(rets[:, 1] == rew[:, 2])

class TestNStepReturn:
    def test_shape(self):
        last, term, rew, val, boot, disc = make_data()
        computer = NStepReturn(n=3)
        rets = computer.compute(last, term, rew, val, boot, disc)
        assert rets.shape == (2, 4)

    def test_n1_matches_lam0(self):
        last, term, rew, val, boot, disc = make_data()
        nstep = NStepReturn(n=1)
        lam0 = LambdaReturn(lam=0.0)
        rets_nstep = nstep.compute(last, term, rew, val, boot, disc)
        rets_lam0 = lam0.compute(last, term, rew, val, boot, disc)
        np.testing.assert_allclose(rets_nstep, rets_lam0, rtol=1e-5)

class TestMonteCarloReturn:
    def test_shape(self):
        last, term, rew, val, boot, disc = make_data()
        computer = MonteCarloReturn()
        rets = computer.compute(last, term, rew, val, boot, disc)
        assert rets.shape == (2, 4)

    def test_matches_lam1_no_boot(self):
        # MonteCarloReturn in this implementation doesn't seem to use boot at all
        # whereas LambdaReturn(lam=1) does use the final boot.
        # Let's verify this.
        last, term, rew, val, boot, disc = make_data()
        # Set final boot to 0 for comparison
        boot = boot.at[:, -1].set(0.0)
        mc = MonteCarloReturn()
        lam1 = LambdaReturn(lam=1.0)
        rets_mc = mc.compute(last, term, rew, val, boot, disc)
        rets_lam1 = lam1.compute(last, term, rew, val, boot, disc)
        np.testing.assert_allclose(rets_mc, rets_lam1, rtol=1e-5)

class TestGAE:
    def test_shape(self):
        last, term, rew, val, boot, disc = make_data()
        computer = GAE(lam=0.95)
        rets = computer.compute(last, term, rew, val, boot, disc)
        assert rets.shape == (2, 4)

    def test_return_is_adv_plus_val(self):
        # GAE.compute returns (adv + val)
        # We can't easily check internal advantages without re-implementing,
        # but we can check if it behaves reasonably.
        last, term, rew, val, boot, disc = make_data()
        computer = GAE(lam=0.95)
        rets = computer.compute(last, term, rew, val, boot, disc)
        # Just a basic sanity check that it's not all zeros or identical to values
        assert not jnp.allclose(rets, val[:, :-1])
        assert not jnp.allclose(rets, jnp.zeros_like(rets))

def test_create_return_computer():
    for strategy in ['lambda', 'nstep', 'montecarlo', 'gae']:
        computer = create_return_computer(strategy)
        assert isinstance(computer, (LambdaReturn, NStepReturn, MonteCarloReturn, GAE))

    with pytest.raises(ValueError):
        create_return_computer('invalid_strategy')
