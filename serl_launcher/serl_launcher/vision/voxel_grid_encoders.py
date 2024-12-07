from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.lax as lax

import jax


class SpatialSoftArgmax3D(nn.Module):
    """
    3D Implementation of Spatial Soft Argmax
    why arg-max and not max: see https://github.com/tensorflow/tensorflow/issues/6271#issuecomment-266893850
    """

    x_len: int
    y_len: int
    z_len: int
    channel: int
    temperature: float = 1.0

    def setup(self):
        pos_x, pos_y, pos_z = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, self.x_len),
            jnp.linspace(-1.0, 1.0, self.y_len),
            jnp.linspace(-1.0, 1.0, self.z_len),
            indexing="ij",
        )
        self.pos_x = pos_x.reshape(-1)  # shape (x*y*z)
        self.pos_y = pos_y.reshape(-1)
        self.pos_z = pos_z.reshape(-1)

    @nn.compact
    def __call__(self, features):
        # add batch dim if missing
        no_batch_dim = len(features.shape) < 5
        if no_batch_dim:
            features = features[None]

        assert len(features.shape) == 5
        batch_size, num_featuremaps = features.shape[0], features.shape[-1]
        features = features.transpose(0, 4, 1, 2, 3).reshape(
            batch_size, num_featuremaps, self.x_len * self.y_len * self.z_len
        )

        softmax_attention = nn.softmax(features / self.temperature, axis=-1)
        expected_x = jnp.sum(self.pos_x * softmax_attention, axis=-1)
        expected_y = jnp.sum(self.pos_y * softmax_attention, axis=-1)
        expected_z = jnp.sum(self.pos_z * softmax_attention, axis=-1)
        expected_xyz = jnp.concatenate([expected_x, expected_y, expected_z], axis=-1)

        expected_xy = jnp.reshape(expected_xyz, (batch_size, 3, num_featuremaps))

        if no_batch_dim:
            expected_xy = expected_xy[0]
        return expected_xy


class VoxNet(nn.Module):
    """
    VoxNet-like implementation: https://github.com/AutoDeep/VoxNet/blob/master/src/nets/voxNet.py
    """

    use_conv_bias: bool = False
    bottleneck_dim: Optional[int] = None
    final_activation: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.tanh
    pretrained: bool = False
    scale_factor: float = 1.0

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        encode: bool = True,
        train: bool = True,
    ):
        # observations has shape (B, X, Y, Z)
        no_batch_dim = len(observations.shape) < 4
        if no_batch_dim:
            observations = observations[None]

        observations = (
            observations.astype(jnp.float32)[..., None] / self.scale_factor
        )  # add conv channel

        conv3d = partial(
            nn.Conv,
            kernel_init=nn.initializers.xavier_normal(),
            use_bias=self.use_conv_bias,
            padding="valid",
            bias_init=nn.zeros_init(),
        )
        l_relu = partial(nn.leaky_relu, negative_slope=0.1)
        max_pool = partial(nn.max_pool, window_shape=(2, 2, 2), strides=(2, 2, 2))

        if self.pretrained:
            feature_dimensions = (64, 64, 32)
        else:
            feature_dimensions = (32, 16, 8)

        x = observations
        x = conv3d(
            features=feature_dimensions[0],
            kernel_size=(5, 5, 5),
            strides=(2, 2, 2),
            name="conv_5x5x5",
        )(x)
        x = nn.LayerNorm()(x)
        x = l_relu(x)

        x = conv3d(
            features=feature_dimensions[1],
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            name="conv_3x3x3",
        )(x)
        x = max_pool(x)

        if self.pretrained:
            x = jax.lax.stop_gradient(
                x
            )  # unfortunately also cuts gradients of the LayerNorm above

        x = nn.LayerNorm()(x)
        x = l_relu(x)

        x = conv3d(
            features=feature_dimensions[
                2
            ],  # if pretrained, uses [..] out of 128 pretrained params as initial weights
            kernel_size=(2, 2, 2),
            strides=(2, 2, 2),
            name="conv_2x2x2",
        )(x)
        x = nn.LayerNorm()(x)
        x = l_relu(x)

        # x = SpatialSoftArgmax3D(10, 10, 8, 64)(x)  # not used for now

        # reshape and dense (preserve batch dim)
        x = jnp.reshape(x, (1 if no_batch_dim else x.shape[0], -1))
        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = self.final_activation(x)

        return x[0] if no_batch_dim else x
