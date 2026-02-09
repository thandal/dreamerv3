"""
Encoder architectures for DreamerV3.

This module provides different encoder architectures that can be used
to process observations into latent representations.

Available encoders:
- SimpleEncoder: Standard CNN + MLP encoder (default DreamerV3)
"""

import math

import embodied.jax
import embodied.jax.nets as nn
import jax.numpy as jnp
import ninjax as nj


class SimpleEncoder(nj.Module):
    """Simple CNN + MLP encoder (default DreamerV3 architecture).

    Processes image observations with a CNN and vector observations with an MLP,
    then concatenates the results.

    Architecture:
    - Images: Multi-scale CNN with pooling/strided convolutions
    - Vectors: Multi-layer perceptron
    - Output: Concatenated encoded representation

    Parameters:
        units: MLP hidden units (default: 1024)
        depth: Base CNN channel depth (default: 64)
        mults: Channel multipliers for CNN layers (default: [2, 3, 4, 4])
        layers: Number of MLP layers (default: 3)
        kernel: Convolution kernel size (default: 5)
        act: Activation function (default: 'gelu')
        norm: Normalization type (default: 'rms')
        symlog: Apply symlog transformation to observations (default: True)
        outer: Use outer convolution (default: False)
        strided: Use strided convolutions instead of pooling (default: False)
    """

    units: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kw):
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.kw = kw

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def __call__(self, carry, obs, reset, training, single=False):
        bdims = 1 if single else 2
        outs = []
        bshape = reset.shape

        # Process vector observations with MLP
        if self.veckeys:
            vspace = {k: self.obs_space[k] for k in self.veckeys}
            vecs = {k: obs[k] for k in self.veckeys}
            squish = nn.symlog if self.symlog else lambda x: x
            x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            for i in range(self.layers):
                x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
            outs.append(x)

        # Process image observations with CNN
        if self.imgkeys:
            K = self.kernel
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            assert all(x.dtype == jnp.uint8 for x in imgs)
            x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
            x = x.reshape((-1, *x.shape[bdims:]))
            for i, depth in enumerate(self.depths):
                if self.outer and i == 0:
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
                elif self.strided:
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
                else:
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
                    B, H, W, C = x.shape
                    x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
                x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
            assert 3 <= x.shape[-3] <= 16, x.shape
            assert 3 <= x.shape[-2] <= 16, x.shape
            x = x.reshape((x.shape[0], -1))
            outs.append(x)

        # Concatenate vector and image encodings
        x = jnp.concatenate(outs, -1)
        tokens = x.reshape((*bshape, *x.shape[1:]))
        entries = {}
        return carry, entries, tokens


# Registry for encoder architectures
ENCODERS = {
    'simple': SimpleEncoder,
}


def create_encoder(encoder_type: str, obs_space, **kwargs):
    """Factory function to create an encoder.

    Args:
        encoder_type: Type of encoder ('simple')
        obs_space: Observation space dict
        **kwargs: Encoder-specific parameters

    Returns:
        Encoder instance

    Example:
        >>> encoder = create_encoder('simple', obs_space, units=1024, depth=64)
    """
    if encoder_type not in ENCODERS:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available types: {list(ENCODERS.keys())}"
        )

    return ENCODERS[encoder_type](obs_space, **kwargs)
