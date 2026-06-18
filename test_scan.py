import jax
import jax.numpy as jnp
from dreamerv3.returns import LambdaReturn, NStepReturn, MonteCarloReturn, GAE

f32 = jnp.float32

class LambdaReturnScan(LambdaReturn):
    def compute(self, last, term, rew, val, boot, disc, lam=None, **kwargs):
        lam = lam if lam is not None else self.lam

        #chex.assert_equal_shape((last, term, rew, val, boot))

        live = (1 - f32(term))[:, 1:] * disc
        cont = (1 - f32(last))[:, 1:] * lam
        interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]

        def step(carry, inputs):
            interm_t, live_t, cont_t = inputs
            ret_t = interm_t + live_t * cont_t * carry
            return ret_t, ret_t

        inputs = (interm.T, live.T, cont.T)
        _, rets = jax.lax.scan(step, boot[:, -1], inputs, reverse=True)

        return rets.T


class MonteCarloReturnScan(MonteCarloReturn):
    def compute(self, last, term, rew, val, boot, disc, **kwargs):
        live = (1 - f32(term))[:, 1:] * (1 - f32(last))[:, 1:]

        def step(carry, inputs):
            rew_t_plus_1, live_t_plus_1 = inputs
            ret_t = rew_t_plus_1 + disc * live_t_plus_1 * carry
            return ret_t, ret_t

        init_carry = jnp.zeros(rew.shape[0], dtype=f32)
        inputs = (rew[:, 1:].T, live.T)
        _, rets = jax.lax.scan(step, init_carry, inputs, reverse=True)

        return rets.T


class GAEScan(GAE):
    def compute(self, last, term, rew, val, boot, disc, lam=None, **kwargs):
        lam = lam if lam is not None else self.lam

        next_val = jnp.concatenate([val[:, 1:-1], boot[:, -1:]], axis=1)
        live = (1 - f32(term))[:, 1:] * disc
        td_errors = rew[:, 1:] + live * next_val - val[:, :-1]

        not_last = (1 - f32(last))[:, 1:]

        def step(gae_t_plus_1, inputs):
            td_t, live_t, not_last_t = inputs
            gae_t = td_t + live_t * lam * gae_t_plus_1 * not_last_t
            return gae_t, gae_t

        init_gae = jnp.zeros(val.shape[0], dtype=f32)
        inputs = (td_errors.T, live.T, not_last.T)
        _, advs = jax.lax.scan(step, init_gae, inputs, reverse=True)

        returns = advs.T + val[:, :-1]

        return returns

def make_data(B=2, T=5, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    rew = jnp.array(rng.randn(B, T).astype(np.float32))
    val = jnp.array(rng.randn(B, T).astype(np.float32))
    boot = jnp.array(rng.randn(B, T).astype(np.float32))
    last = jnp.zeros((B, T), dtype=jnp.float32)
    term = jnp.zeros((B, T), dtype=jnp.float32)
    disc = 0.99
    return last, term, rew, val, boot, disc

if __name__ == '__main__':
    last, term, rew, val, boot, disc = make_data()

    # Lambda
    orig_lam = LambdaReturn(lam=0.95).compute(last, term, rew, val, boot, disc)
    scan_lam = LambdaReturnScan(lam=0.95).compute(last, term, rew, val, boot, disc)
    assert jnp.allclose(orig_lam, scan_lam, atol=1e-5), f"Lambda differs:\norig={orig_lam}\nscan={scan_lam}"

    # MonteCarlo
    orig_mc = MonteCarloReturn().compute(last, term, rew, val, boot, disc)
    scan_mc = MonteCarloReturnScan().compute(last, term, rew, val, boot, disc)
    assert jnp.allclose(orig_mc, scan_mc, atol=1e-5), f"MC differs:\norig={orig_mc}\nscan={scan_mc}"

    # GAE
    orig_gae = GAE(lam=0.95).compute(last, term, rew, val, boot, disc)
    scan_gae = GAEScan(lam=0.95).compute(last, term, rew, val, boot, disc)
    assert jnp.allclose(orig_gae, scan_gae, atol=1e-5), f"GAE differs:\norig={orig_gae}\nscan={scan_gae}"

    print("All tests passed!")
