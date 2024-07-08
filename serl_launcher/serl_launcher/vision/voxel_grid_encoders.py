from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.lax as lax


class MLPEncoder(nn.Module):
    mlp: nn.module = None
    bottleneck_dim: Optional[int] = None

    @nn.compact
    def __call__(
            self,
            observations: jnp.ndarray,
            encode: bool = True,
            train: bool = True,
    ):
        # add batch dim if missing
        no_batch_dim = len(observations.shape) < 4
        if no_batch_dim:
            observations = observations[None]

        # flatten but keep batch tim
        x = jnp.reshape(observations, (observations.shape[0], -1))

        if encode:
            x = self.mlp(x, train=train)

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x[0] if no_batch_dim else x


class VoxNet(nn.Module):
    """
    Voxnet implementation: https://github.com/AutoDeep/VoxNet/blob/master/src/nets/voxNet.py
    """

    use_conv_bias: bool = False
    bottleneck_dim: Optional[int] = None

    @nn.compact
    def __call__(
            self,
            observations: jnp.ndarray,
            encode: bool = True,
            train: bool = True,
    ):
        # observations has shape (B, X, Y, Z) (boolean for now)
        no_batch_dim = len(observations.shape) < 4
        if no_batch_dim:
            observations = observations[None]

        observations = observations.astype(jnp.float32)[..., None] / 8.      # add conv channel and scale to [0, 1]

        conv = partial(nn.Conv, kernel_init=nn.initializers.xavier_normal(), use_bias=self.use_conv_bias, padding="valid")
        l_relu = partial(nn.leaky_relu, negative_slope=0.1)

        x = observations
        x = conv(
            features=32,
            kernel_size=(5, 5, 5),
            strides=(2, 2, 2),
            name="conv_5x5",
        )(x)
        x = l_relu(x)  # shape (B, (X-3)/2, (Y-3)/2, (Z-3)/2)

        x = conv(
            features=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            name="conv_3x3"
        )(x)
        x = l_relu(x)  # shape (B, (X-4)/2, (Y-4)/2, (Z-4)/2)

        x = lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID'
        )
        # print(x.shape, end='  ')

        # reshape and dense (preserve batch dim)
        x = jnp.reshape(x, (1 if no_batch_dim else x.shape[0], -1))
        # print(x.shape)

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x[0] if no_batch_dim else x
