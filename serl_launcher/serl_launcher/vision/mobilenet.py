from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from serl_launcher.vision.spatial import SpatialLearnedEmbeddings


class MobileNetEncoder(nn.Module):
    """
    this serves as a wrapper for a imagenet pretrained mobilenet encoder.
    Params:
        encoder: the encoder network
        params: the parameters of the encoder
    """

    encoder: nn.Module
    params: FrozenDict
    pool_method: str = "spatial_learned_embeddings"
    bottleneck_dim: Optional[int] = None
    spatial_block_size: Optional[int] = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, train=False) -> jnp.ndarray:
        """
        encode an image using the mobilenet encoder

        Params:
            x: input image
            train: whether the network is in training mode
            divide_by: whether to divide the image by 255
            reshape: whether to reshape the image before passing into encoder
        Return:
            the encoded image
        """
        # normalize inputs using imagenet mean and std
        mean = jnp.array((0.485, 0.456, 0.406))[None, ...]
        std = jnp.array((0.229, 0.224, 0.225))[None, ...]
        x = x.astype(jnp.float32) / 255.0
        x = (x - mean) / std

        reshape = False
        if x.ndim == 3:
            x = x[None, ...]
            reshape = True

        x = self.encoder.apply(self.params, x, mutable=False, training=False)
        x = jax.lax.stop_gradient(x)

        if self.pool_method == "max":
            x = jnp.max(x, axis=(-3, -2))
        elif self.pool_method == "avg":
            x = jnp.mean(x, axis=(-3, -2))
        elif self.pool_method == "spatial_learned_embeddings":
            if self.spatial_block_size is None:
                raise ValueError(
                    "spatial_block_size must be set when using spatial_learned_embeddings"
                )
            x = SpatialLearnedEmbeddings(*(x.shape[-3:]), self.spatial_block_size)(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)

        if reshape:
            x = x.reshape(-1)

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x
