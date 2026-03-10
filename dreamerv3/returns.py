"""
Pluggable return computation strategies for DreamerV3.

This module provides different methods for computing returns and advantages
in the actor-critic framework.

Available strategies:
- LambdaReturn: TD(λ) returns with exponential weighting (default DreamerV3)
- NStepReturn: Fixed N-step bootstrapped returns
- MonteCarloReturn: Full episode returns without bootstrapping
- GAE: Generalized Advantage Estimation
"""

import chex
import jax
import jax.numpy as jnp

f32 = jnp.float32


class ReturnComputer:
    """Base class for return computation strategies."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def compute(self, last, term, rew, val, boot, disc, **kwargs):
        """Compute returns from trajectory data.

        Args:
            last: Boolean array indicating episode boundaries (B, T)
            term: Boolean array indicating terminal states (B, T)
            rew: Rewards (B, T)
            val: Value predictions (B, T)
            boot: Bootstrap values for the final step (B, T)
            disc: Discount factor (scalar or array)
            **kwargs: Additional strategy-specific parameters

        Returns:
            Returns array of shape (B, T-1)
        """
        raise NotImplementedError


class LambdaReturn(ReturnComputer):
    """TD(λ) return computation (default DreamerV3 strategy).

    Computes exponentially weighted mixture of n-step returns:
        ret_t = r_t + γ * [(1-λ) * V_{t+1} + λ * ret_{t+1}]

    This balances bias (low λ) vs variance (high λ).

    Reference:
        Sutton & Barto (2018) - "Reinforcement Learning: An Introduction"
        DreamerV3 paper uses λ=0.95

    Args:
        lam: Lambda parameter for exponential weighting (default: 0.95)
             lam=0: 1-step TD (high bias, low variance)
             lam=1: Monte Carlo (low bias, high variance)
    """

    def __init__(self, lam=0.95, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

    def compute(self, last, term, rew, val, boot, disc, lam=None, **kwargs):
        """Compute TD(λ) returns.

        Args:
            last: Episode boundaries (B, T)
            term: Terminal states (B, T)
            rew: Rewards (B, T)
            val: Value predictions (B, T)
            boot: Bootstrap values (B, T)
            disc: Discount factor
            lam: Override lambda (default: use self.lam)

        Returns:
            Returns (B, T-1)
        """
        lam = lam if lam is not None else self.lam

        chex.assert_equal_shape((last, term, rew, val, boot))

        # Compute effective discount accounting for termination
        live = (1 - f32(term))[:, 1:] * disc

        # Continuation weight (0 at episode boundaries)
        cont = (1 - f32(last))[:, 1:] * lam

        # Intermediate value: immediate reward + (1-λ) * bootstrapped value
        interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]

        # Backward pass to compute returns
        # Using jax.lax.scan to avoid O(T) node graph size during JIT compilation
        def step(carry, inputs):
            interm_t, live_t, cont_t = inputs
            ret_t = interm_t + live_t * cont_t * carry
            return ret_t, ret_t

        carry = boot[:, -1]
        # Transpose inputs to (T, B) for scan along time dimension
        interm_T = jnp.swapaxes(interm, 0, 1)
        live_T = jnp.swapaxes(live, 0, 1)
        cont_T = jnp.swapaxes(cont, 0, 1)

        _, rets_T = jax.lax.scan(step, carry, (interm_T, live_T, cont_T), reverse=True)

        # Transpose back to (B, T)
        return jnp.swapaxes(rets_T, 0, 1)


class NStepReturn(ReturnComputer):
    """Fixed N-step bootstrapped returns.

    Computes returns using exactly N steps of rewards and then bootstrapping:
        ret_t = r_t + γ*r_{t+1} + ... + γ^{N-1}*r_{t+N-1} + γ^N*V_{t+N}

    This is a special case of TD(λ) with less computation.

    Args:
        n: Number of steps to look ahead (default: 5)
            n=1: Standard TD learning
            n=∞: Approaches Monte Carlo
    """

    def __init__(self, n=5, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def compute(self, last, term, rew, val, boot, disc, n=None, **kwargs):
        """Compute N-step returns.

        Args:
            last: Episode boundaries (B, T)
            term: Terminal states (B, T)
            rew: Rewards (B, T)
            val: Value predictions (B, T)
            boot: Bootstrap values (B, T)
            disc: Discount factor
            n: Override N (default: use self.n)

        Returns:
            Returns (B, T-1)
        """
        n = n if n is not None else self.n

        chex.assert_equal_shape((last, term, rew, val, boot))

        B, T = rew.shape

        def calc_for_t(t):
            def step(carry, k):
                ret, discount = carry
                idx = t + k + 1
                valid = idx < T

                safe_idx = jnp.minimum(idx, T - 1)

                # Add discounted reward
                r = jnp.where(valid, rew[:, safe_idx], 0.0)
                ret = ret + discount * r

                # Stop at episode boundaries or terminals
                l = jnp.where(valid, (1 - f32(term[:, safe_idx])) * (1 - f32(last[:, safe_idx])), 0.0)
                next_discount = jnp.where(valid, discount * disc * l, discount)

                return (ret, next_discount), None

            init_carry = (jnp.zeros(B, dtype=f32), jnp.ones(B, dtype=f32))
            (ret, discount), _ = jax.lax.scan(step, init_carry, jnp.arange(n))

            # Add bootstrapped value at n-th step (or end of sequence)
            final_idx = jnp.minimum(t + n, T - 1)
            ret = ret + discount * boot[:, final_idx]
            return ret

        ts = jnp.arange(T - 1)
        rets = jax.vmap(calc_for_t)(ts)
        return jnp.transpose(rets, (1, 0))


class MonteCarloReturn(ReturnComputer):
    """Monte Carlo returns (full episode rewards).

    Computes returns using all rewards until episode end, without bootstrapping:
        ret_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ... + γ^{T-t}*r_T

    This has zero bias but high variance.

    Useful for:
    - Short episodes
    - Environments where value estimation is unreliable
    - Baseline comparisons
    """

    def compute(self, last, term, rew, val, boot, disc, **kwargs):
        """Compute Monte Carlo returns.

        Args:
            last: Episode boundaries (B, T)
            term: Terminal states (B, T)
            rew: Rewards (B, T)
            val: Value predictions (ignored, for compatibility)
            boot: Bootstrap values (ignored, for compatibility)
            disc: Discount factor

        Returns:
            Returns (B, T-1)
        """
        chex.assert_equal_shape((last, term, rew, val, boot))

        B, T = rew.shape
        rets = []

        # Start from the end and work backwards
        for t in range(T - 1):
            ret = jnp.zeros(B, dtype=f32)
            discount = 1.0

            for k in range(t + 1, T):
                ret = ret + discount * rew[:, k]

                # Stop at episode boundaries or terminals
                live = (1 - f32(term[:, k])) * (1 - f32(last[:, k]))
                discount = discount * disc * live

            rets.append(ret)

        return jnp.stack(rets, 1)


class GAE(ReturnComputer):
    """Generalized Advantage Estimation.

    Computes advantages using exponentially-weighted TD errors:
        A_t = δ_t + γλδ_{t+1} + (γλ)^2*δ_{t+2} + ...
        where δ_t = r_t + γV_{t+1} - V_t

    Returns are then: ret_t = A_t + V_t

    This is the standard method used in PPO and other policy gradient algorithms.

    Reference:
        Schulman et al. (2016) - "High-Dimensional Continuous Control Using GAE"

    Args:
        lam: GAE lambda parameter (default: 0.95)
             Controls bias-variance tradeoff like TD(λ)
    """

    def __init__(self, lam=0.95, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

    def compute(self, last, term, rew, val, boot, disc, lam=None, **kwargs):
        """Compute GAE returns.

        Args:
            last: Episode boundaries (B, T)
            term: Terminal states (B, T)
            rew: Rewards (B, T)
            val: Value predictions (B, T)
            boot: Bootstrap values (B, T)
            disc: Discount factor
            lam: Override lambda (default: use self.lam)

        Returns:
            Returns (B, T-1)
        """
        lam = lam if lam is not None else self.lam

        chex.assert_equal_shape((last, term, rew, val, boot))

        # Compute TD errors: δ_t = r_t + γV_{t+1} - V_t
        next_val = jnp.concatenate([val[:, 1:-1], boot[:, -1:]], axis=1)
        live = (1 - f32(term))[:, 1:] * disc
        td_errors = rew[:, 1:] + live * next_val - val[:, :-1]

        # Compute GAE advantages via backward pass
        # Using jax.lax.scan to avoid O(T) node graph size during JIT compilation
        def step(carry, inputs):
            td_t, live_t, last_t_plus_1 = inputs
            gae_t = td_t + live_t * lam * carry * (1 - last_t_plus_1)
            return gae_t, gae_t

        carry = jnp.zeros(val.shape[0], dtype=f32)

        # Transpose inputs to (T, B) for scan along time dimension
        td_T = jnp.swapaxes(td_errors, 0, 1)
        live_T = jnp.swapaxes(live, 0, 1)
        last_t_plus_1_T = jnp.swapaxes(f32(last)[:, 1:], 0, 1)

        _, advs_T = jax.lax.scan(step, carry, (td_T, live_T, last_t_plus_1_T), reverse=True)

        # Transpose back to (B, T)
        advs = jnp.swapaxes(advs_T, 0, 1)

        # Returns = advantages + values
        returns = advs + val[:, :-1]

        return returns


# Registry for return computation strategies
RETURN_COMPUTERS = {
    'lambda': LambdaReturn,
    'nstep': NStepReturn,
    'montecarlo': MonteCarloReturn,
    'gae': GAE,
}


def create_return_computer(strategy: str, **kwargs):
    """Factory function to create a return computer.

    Args:
        strategy: Return computation strategy ('lambda', 'nstep', 'montecarlo', 'gae')
        **kwargs: Strategy-specific parameters

    Returns:
        ReturnComputer instance

    Example:
        >>> computer = create_return_computer('lambda', lam=0.95)
        >>> returns = computer.compute(last, term, rew, val, boot, disc)
    """
    if strategy not in RETURN_COMPUTERS:
        raise ValueError(
            f"Unknown return strategy: {strategy}. "
            f"Available strategies: {list(RETURN_COMPUTERS.keys())}"
        )

    return RETURN_COMPUTERS[strategy](**kwargs)
