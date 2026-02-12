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
        return TwoHotDist(logits, self.stoch, self.classes)

    def get_shape(self):
        return (self.stoch, self.classes)


class TwoHotDist:
    """Distribution for two-hot encoding with straight-through gradients."""

    def __init__(self, logits, stoch, classes):
        self.logits = f32(logits)
        self.stoch = stoch
        self.classes = classes

    def sample(self, seed):
        """Sample using two-hot encoding."""
        # Get probabilities
        probs = jax.nn.softmax(self.logits, -1)

        # Compute weighted average position
        bins = jnp.arange(self.classes, dtype=f32)
        pos = (probs * bins).sum(-1)  # Shape: (..., stoch)

        # Convert to two-hot
        below = jnp.floor(pos).astype(jnp.int32)
        above = below + 1
        below = jnp.clip(below, 0, self.classes - 1)
        above = jnp.clip(above, 0, self.classes - 1)

        weight_above = pos - below.astype(f32)
        weight_below = 1.0 - weight_above

        stoch = (
            jax.nn.one_hot(below, self.classes) * weight_below[..., None] +
            jax.nn.one_hot(above, self.classes) * weight_above[..., None]
        )

        # Straight-through gradient from probs
        return f32(stoch + (probs - sg(probs)))

    def logp(self, event):
        """Log probability (treat as categorical)."""
        # Approximate by treating two-hot as soft one-hot
        logprob = jax.nn.log_softmax(self.logits, -1)
        return (logprob * event).sum(-1).sum(-1)  # Sum over classes and stoch

    def entropy(self):
        """Entropy of the underlying categorical distribution."""
        logprob = jax.nn.log_softmax(self.logits, -1)
        prob = jax.nn.softmax(self.logits, -1)
        entropy = -(prob * logprob).sum(-1)  # Per stoch variable
        return entropy.sum(-1)  # Sum over stoch dimension

    def kl(self, other):
        """KL divergence between two-hot distributions."""
        assert isinstance(other, TwoHotDist), other
        logprob = jax.nn.log_softmax(self.logits, -1)
        logother = jax.nn.log_softmax(other.logits, -1)
        prob = jax.nn.softmax(self.logits, -1)
        kl = (prob * (logprob - logother)).sum(-1)  # Per stoch variable
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

    Uses a learned codebook of embedding vectors. The RSSM produces logits
    that are treated as continuous queries; the nearest codebook entry is
    selected via argmin distance, with straight-through gradients.

    Shape: (stoch, classes) - Same as one-hot (one-hot over codebook indices)

    The commitment loss (encoder encouraged to commit to codebook vectors)
    is exposed through the distribution's kl() method so it integrates
    naturally as the KL term in the RSSM loss.
    """

    def __init__(self, stoch: int, classes: int,
                 commitment_cost: float = 0.25,
                 temperature: float = 1.0, **kwargs):
        """
        Args:
            stoch: Number of stochastic variables (each picks a codebook entry)
            classes: Codebook size (number of embedding vectors)
            commitment_cost: Weight for commitment loss (β in VQ-VAE paper)
            temperature: Temperature for softmax over distances
        """
        super().__init__(stoch, classes, **kwargs)
        self.commitment_cost = commitment_cost
        self.temperature = temperature

    def create_dist(self, logits):
        """Create VQ-VAE distribution from logits.

        Args:
            logits: Shape (..., stoch, classes) — treated as continuous queries

        Returns:
            VQVAEDist distribution object
        """
        return VQVAEDist(
            logits, self.stoch, self.classes,
            self.commitment_cost, self.temperature)

    def get_shape(self):
        return (self.stoch, self.classes)


class VQVAEDist:
    """Distribution for VQ-VAE with straight-through gradients.

    Treats input logits as continuous queries into a discrete codebook.
    Each stoch variable independently selects the nearest codebook entry.
    """

    def __init__(self, logits, stoch, classes,
                 commitment_cost=0.25, temperature=1.0):
        self.logits = f32(logits)  # (..., stoch, classes)
        self.stoch = stoch
        self.classes = classes
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        # Compute soft probabilities from logits (used for entropy/logp)
        self._probs = jax.nn.softmax(self.logits / self.temperature, -1)

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
        """Log probability (treat as categorical over codebook)."""
        logprob = jax.nn.log_softmax(self.logits / self.temperature, -1)
        return (logprob * event).sum(-1).sum(-1)  # Sum over classes and stoch

    def entropy(self):
        """Entropy of the categorical distribution over codebook entries."""
        logprob = jax.nn.log_softmax(self.logits / self.temperature, -1)
        entropy = -(self._probs * logprob).sum(-1)  # Per stoch variable
        return entropy.sum(-1)  # Sum over stoch dimension

    def kl(self, other):
        """Commitment loss as KL surrogate.

        For VQ-VAE, the "KL" between posterior and prior is replaced by
        the commitment loss: β * ||z_e - sg(e)||² + ||sg(z_e) - e||²

        When called as dist(sg(post)).kl(dist(prior)) [dynamics loss],
        this measures how far the prior's logits are from the posterior's
        selected codebook entries. When called as
        dist(post).kl(dist(sg(prior))) [representation loss], it measures
        how far the posterior's logits are from the prior's entries.

        This maps naturally to the RSSM's dyn/rep loss structure.
        """
        assert isinstance(other, VQVAEDist), type(other)
        # Codebook loss: pull codebook entries toward encoder outputs
        codebook_loss = jnp.square(self.logits - sg(other.logits)).sum(-1)
        # Commitment loss: pull encoder outputs toward codebook entries
        commit_loss = jnp.square(sg(self.logits) - other.logits).sum(-1)
        # Combined per-stoch loss, then sum over stoch
        return (codebook_loss + self.commitment_cost * commit_loss).sum(-1)


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
