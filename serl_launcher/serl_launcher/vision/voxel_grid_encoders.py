from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.lax as lax

import jax


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

        observations = observations.astype(jnp.float32)[..., None] / 1.  # add conv channel

        conv = partial(nn.Conv, kernel_init=nn.initializers.xavier_normal(), use_bias=self.use_conv_bias,
                       padding="valid")
        l_relu = partial(nn.leaky_relu, negative_slope=0.1)

        x = observations
        x = conv(
            features=32,
            kernel_size=(5, 5, 5),
            strides=(2, 2, 2),
            name="conv_5x5",
        )(x)
        x = l_relu(x)  # shape (B, (X-3)/2, (Y-3)/2, (Z-3)/2, 32)

        x = conv(
            features=32,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            name="conv_3x3_1"
        )(x)
        x = l_relu(x)  # shape (B, (X-4)/4, (Y-4)/4, (Z-4)/4, 32)

        x = conv(
            features=32,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            name="conv_3x3_2"
        )(x)
        x = l_relu(x)  # shape (B, (X-5)/8, (Y-5)/8, (Z-5)/8, 32)

        x = jnp.mean(x, axis=(-2))      # average over z dim

        # 1x1 convolution (dimensionality reduction, no features), not used for now
        # x = conv(
        #     features=1,
        #     kernel_size=(1, 1, 1),
        # )(x)

        # x = lax.reduce_window(
        #     x,
        #     init_value=-jnp.inf,
        #     computation=lax.max,
        #     window_dimensions=(1, 2, 2, 2, 1),
        #     window_strides=(1, 2, 2, 2, 1),
        #     padding='VALID'
        # )

        # reshape and dense (preserve batch dim)
        x = jnp.reshape(x, (1 if no_batch_dim else x.shape[0], -1))
        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x[0] if no_batch_dim else x
