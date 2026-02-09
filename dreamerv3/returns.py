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

        # Start from the last bootstrap value
        rets = [boot[:, -1]]

        # Compute effective discount accounting for termination
        live = (1 - f32(term))[:, 1:] * disc

        # Continuation weight (0 at episode boundaries)
        cont = (1 - f32(last))[:, 1:] * lam

        # Intermediate value: immediate reward + (1-λ) * bootstrapped value
        interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]

        # Backward pass to compute returns
        for t in reversed(range(live.shape[1])):
            rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])

        return jnp.stack(list(reversed(rets))[:-1], 1)


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
        rets = []

        # For each timestep, sum n future rewards
        for t in range(T - 1):
            ret = jnp.zeros(B, dtype=f32)
            discount = 1.0

            for k in range(min(n, T - t - 1)):
                # Add discounted reward
                ret = ret + discount * rew[:, t + k + 1]

                # Stop at episode boundaries or terminals
                live = (1 - f32(term[:, t + k + 1])) * (1 - f32(last[:, t + k + 1]))
                discount = discount * disc * live

            # Add bootstrapped value at n-th step (or end of sequence)
            final_idx = min(t + n, T - 1)
            ret = ret + discount * boot[:, final_idx]

            rets.append(ret)

        return jnp.stack(rets, 1)


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
        next_val = jnp.concatenate([val[:, 1:], boot[:, -1:]], axis=1)
        live = (1 - f32(term))[:, 1:] * disc
        td_errors = rew[:, 1:] + live * next_val - val[:, :-1]

        # Compute GAE advantages via backward pass
        advs = []
        gae = jnp.zeros(val.shape[0], dtype=f32)

        for t in reversed(range(td_errors.shape[1])):
            gae = td_errors[:, t] + live[:, t] * lam * gae * (1 - f32(last)[:, t + 1])
            advs.append(gae)

        advs = jnp.stack(list(reversed(advs)), 1)

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
