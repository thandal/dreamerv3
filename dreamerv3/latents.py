"""
Pluggable latent representations for RSSM.

This module provides different types of latent variable representations
that can be used in the RSSM (Recurrent State-Space Model).

Available latent types:
- OneHotLatent: Categorical distribution with one-hot encoding (default)
- TwoHotLatent: Weighted two-hot encoding for smoother representations
- GaussianLatent: Continuous Gaussian distribution
- VQVAELatent: Vector-quantized discrete codes via learned codebook
"""

import jax
import jax.numpy as jnp
import embodied.jax
import embodied.jax.outs as outs
import ninjax as nj

f32 = jnp.float32
sg = jax.lax.stop_gradient


class LatentBase:
    """Base class for latent representations.

    All latent types must implement these methods to be compatible with RSSM.
    """

    def __init__(self, stoch: int, classes: int, unimix: float = 0.0, **kwargs):
        """
        Args:
            stoch: Number of stochastic variables
            classes: Number of classes per variable (for discrete) or ignored (for continuous)
            unimix: Uniform mixture ratio for discrete distributions
        """
        self.stoch = stoch
        self.classes = classes
        self.unimix = unimix
        self.kwargs = kwargs

    def create_dist(self, logits):
        """Create a distribution object from logits.

        Args:
            logits: Logits tensor with shape (..., stoch, classes) for discrete
                   or (..., stoch * 2) for Gaussian (mean + log_stddev)

        Returns:
            Distribution object with sample(), logp(), entropy(), kl() methods
        """
        raise NotImplementedError

    def sample(self, logits, seed):
        """Sample from the latent distribution.

        Args:
            logits: Logits tensor
            seed: Random seed

        Returns:
            Sample tensor with shape matching the latent representation
        """
        dist = self.create_dist(logits)
        return dist.sample(seed)

    def get_shape(self):
        """Return the shape of the latent representation (excluding batch dims)."""
        raise NotImplementedError


class OneHotLatent(LatentBase):
    """One-hot categorical latent representation.

    This is the default DreamerV3 latent type. Uses categorical distributions
    with one-hot encoding and straight-through gradients.

    Shape: (stoch, classes) - Each stoch variable is a one-hot vector
    """

    def create_dist(self, logits):
        """Create aggregated one-hot distribution.

        Args:
            logits: Shape (..., stoch, classes)

        Returns:
            Aggregated distribution that sums over stoch dimension
        """
        # Create one-hot distribution for each stochastic variable
        dist = outs.OneHot(logits, self.unimix)
        # Aggregate over the stoch dimension (sum KL, entropy, etc.)
        dist = outs.Agg(dist, 1, jnp.sum)
        return dist

    def get_shape(self):
        return (self.stoch, self.classes)


class TwoHotLatent(LatentBase):
    """Two-hot weighted latent representation.

    Uses weighted interpolation between two adjacent bins for smoother
    representations. Better for representing continuous-like values.

    Shape: (stoch, classes) - Each stoch variable is a two-hot vector
    """

    def create_dist(self, logits):
        """Create two-hot distribution via softmax + weighted encoding.

        Args:
            logits: Shape (..., stoch, classes)

        Returns:
            Custom distribution object
        """
        return TwoHotDist(logits, self.stoch, self.classes, self.unimix)

    def get_shape(self):
        return (self.stoch, self.classes)


class TwoHotDist:
    """Distribution for two-hot encoding with straight-through gradients."""

    def __init__(self, logits, stoch, classes, unimix=0.0):
        self.logits = f32(logits)
        self.stoch = stoch
        self.classes = classes
        self.unimix = unimix
        # Apply unimix regularization (same as Categorical)
        probs = jax.nn.softmax(self.logits, -1)
        if unimix:
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
        self._probs = probs

    def sample(self, seed):
        """Sample using stochastic two-hot encoding.

        Samples a categorical index, then constructs a two-hot vector
        by spreading weight between the sampled bin and a neighbor
        (determined by the expected position's fractional part).
        Uses straight-through gradients from the underlying probs.
        """
        probs = self._probs

        # Stochastic sampling: draw a categorical index
        idx = jax.random.categorical(seed, jnp.log(probs), -1)  # (..., stoch)

        # Compute expected position from probs to determine neighbor direction
        bins = jnp.arange(self.classes, dtype=f32)
        expected_pos = (probs * bins).sum(-1)  # (..., stoch)

        # Determine neighbor: if expected position > sampled index, neighbor
        # is above; otherwise below. Clamp to valid range.
        idx_f = idx.astype(f32)
        go_up = expected_pos >= idx_f
        neighbor = jnp.where(go_up, idx + 1, idx - 1)
        neighbor = jnp.clip(neighbor, 0, self.classes - 1)

        # Fractional weight between idx and neighbor based on expected position
        frac = jnp.abs(expected_pos - idx_f)
        frac = jnp.clip(frac, 0.0, 1.0)

        # Construct two-hot: (1-frac) on sampled bin + frac on neighbor
        stoch = (
            jax.nn.one_hot(idx, self.classes) * (1.0 - frac)[..., None] +
            jax.nn.one_hot(neighbor, self.classes) * frac[..., None]
        )

        # Straight-through gradient from probs
        return f32(stoch + (probs - sg(probs)))

    def logp(self, event):
        """Log probability (treat as categorical)."""
        logprob = jnp.log(self._probs)
        return (logprob * event).sum(-1).sum(-1)  # Sum over classes and stoch

    def entropy(self):
        """Entropy of the underlying categorical distribution."""
        logprob = jnp.log(self._probs)
        entropy = -(self._probs * logprob).sum(-1)  # Per stoch variable
        return entropy.sum(-1)  # Sum over stoch dimension

    def kl(self, other):
        """KL divergence between two-hot distributions."""
        assert isinstance(other, TwoHotDist), other
        logprob = jnp.log(self._probs)
        logother = jnp.log(other._probs)
        kl = (self._probs * (logprob - logother)).sum(-1)  # Per stoch variable
        return kl.sum(-1)  # Sum over stoch dimension


class GaussianLatent(LatentBase):
    """Gaussian (diagonal multivariate normal) latent representation.

    Uses continuous Gaussian distributions instead of discrete categoricals.
    This is more similar to the original RSSM (Hafner et al., 2019).

    Shape: (stoch,) - Each stoch variable is a continuous scalar

    The 'classes' parameter is ignored for Gaussian latents.
    """

    def __init__(self, stoch: int, classes: int = None,
                 min_stddev: float = 0.1, max_stddev: float = 1.0, **kwargs):
        """
        Args:
            stoch: Dimensionality of the Gaussian latent
            classes: Ignored (kept for compatibility)
            min_stddev: Minimum standard deviation
            max_stddev: Maximum standard deviation
        """
        super().__init__(stoch, classes or 1, **kwargs)
        self.min_stddev = min_stddev
        self.max_stddev = max_stddev

    def create_dist(self, logits):
        """Create diagonal Gaussian distribution.

        Args:
            logits: Shape (..., stoch * 2) where first half is mean,
                   second half is log stddev

        Returns:
            Aggregated Gaussian distribution
        """
        # Split into mean and log_stddev
        mean, log_stddev = jnp.split(logits.reshape(logits.shape[:-2] + (-1,)), 2, -1)

        # Constrain stddev to reasonable range
        stddev = jnp.exp(log_stddev)
        stddev = jnp.clip(stddev, self.min_stddev, self.max_stddev)

        # Create Normal distribution
        dist = outs.Normal(mean, stddev)

        # Aggregate over the stoch dimension
        dist = outs.Agg(dist, 1, jnp.sum)

        return dist

    def get_shape(self):
        # For Gaussian, we store as flat vector (not classes dimension)
        return (self.stoch,)


class VQVAELatent(LatentBase):
    """VQ-VAE discrete latent representation.

    Uses categorical distributions with deterministic (argmax) sampling
    instead of stochastic sampling. This is the key difference from OneHot:
    the forward pass always selects the highest-probability codebook entry,
    while gradients flow through via straight-through estimator.

    Shape: (stoch, classes) - Same as one-hot (one-hot over codebook indices)

    KL divergence, entropy, and logp use standard categorical formulations,
    making this a true drop-in replacement for OneHot.
    """

    def __init__(self, stoch: int, classes: int, **kwargs):
        """
        Args:
            stoch: Number of stochastic variables (each picks a codebook entry)
            classes: Codebook size (number of embedding vectors)
        """
        super().__init__(stoch, classes, **kwargs)

    def create_dist(self, logits):
        """Create VQ-VAE distribution from logits.

        Args:
            logits: Shape (..., stoch, classes) — treated as codebook scores

        Returns:
            VQVAEDist distribution object
        """
        return VQVAEDist(logits, self.stoch, self.classes, self.unimix)

    def get_shape(self):
        return (self.stoch, self.classes)


class VQVAEDist:
    """Distribution for VQ-VAE with straight-through gradients.

    A categorical distribution that uses deterministic argmax sampling
    (instead of stochastic sampling like OneHot). Gradients flow through
    the straight-through estimator. All distributional quantities (KL,
    entropy, logp) use standard categorical formulations.
    """

    def __init__(self, logits, stoch, classes, unimix=0.0):
        self.stoch = stoch
        self.classes = classes
        self.unimix = unimix
        # Apply unimix regularization (matching Categorical behavior)
        logits = f32(logits)
        probs = jax.nn.softmax(logits, -1)
        if unimix:
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
        self._probs = probs
        # Recompute logits from mixed probs for consistency
        self.logits = jnp.log(probs)

    def sample(self, seed):
        """Sample by selecting the argmax codebook entry (one-hot).

        Uses straight-through gradient: forward pass is hard one-hot,
        backward pass uses soft probabilities.
        """
        # Hard selection: argmax over classes dimension
        indices = jnp.argmax(self.logits, axis=-1)  # (..., stoch)
        hard = jax.nn.one_hot(indices, self.classes)  # (..., stoch, classes)
        # Straight-through: hard forward, soft backward
        return f32(hard + (self._probs - sg(self._probs)))

    def logp(self, event):
        """Log probability (categorical over codebook entries)."""
        return (self.logits * event).sum(-1).sum(-1)  # Sum over classes and stoch

    def entropy(self):
        """Entropy of the categorical distribution over codebook entries."""
        entropy = -(self._probs * self.logits).sum(-1)  # Per stoch variable
        return entropy.sum(-1)  # Sum over stoch dimension

    def kl(self, other):
        """KL divergence between VQ-VAE distributions (categorical KL).

        Uses standard categorical KL: sum_x p(x) * (log p(x) - log q(x)).
        No internal stop-gradients — the RSSM handles gradient routing
        by applying sg() to the appropriate argument before calling kl().
        """
        assert isinstance(other, VQVAEDist), type(other)
        kl = (self._probs * (self.logits - other.logits)).sum(-1)  # Per stoch
        return kl.sum(-1)  # Sum over stoch dimension


# Registry for latent types
LATENT_TYPES = {
    'onehot': OneHotLatent,
    'twohot': TwoHotLatent,
    'gaussian': GaussianLatent,
    'vqvae': VQVAELatent,
}


def create_latent(latent_type: str, stoch: int, classes: int, **kwargs):
    """Factory function to create a latent representation.

    Args:
        latent_type: Type of latent ('onehot', 'twohot', 'gaussian')
        stoch: Number of stochastic variables
        classes: Classes per variable (or ignored for Gaussian)
        **kwargs: Additional parameters for specific latent types

    Returns:
        LatentBase instance

    Example:
        >>> latent = create_latent('onehot', stoch=32, classes=64, unimix=0.01)
        >>> dist = latent.create_dist(logits)
        >>> sample = dist.sample(seed)
    """
    if latent_type not in LATENT_TYPES:
        raise ValueError(
            f"Unknown latent type: {latent_type}. "
            f"Available types: {list(LATENT_TYPES.keys())}"
        )

    return LATENT_TYPES[latent_type](stoch=stoch, classes=classes, **kwargs)
