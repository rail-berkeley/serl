from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn

from serl_launcher.vision.spatial import SpatialLearnedEmbeddings


class SmallEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32)
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    padding: Union[Sequence[int], str] = (1, 1, 1)
    pool_method: str = "spatial_learned_embeddings"
    bottleneck_dim: Optional[int] = None
    spatial_block_size: Optional[int] = 8
    num_kp: Optional[int] = 32

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train=False, encode=True) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0

        for i in range(len(self.features)):
            if isinstance(self.padding, str):
                padding = self.padding
            else:
                padding = self.padding[i]

            x = nn.Conv(
                self.features[i],
                kernel_size=(self.kernel_sizes[i], self.kernel_sizes[i]),
                strides=(self.strides[i], self.strides[i]),
                padding=padding,
            )(x)
            x = nn.relu(x)

        if self.pool_method == "max":
            x = jnp.max(x, axis=(-3, -2))
        elif self.pool_method == "avg":
            x = jnp.mean(x, axis=(-3, -2))
        elif self.pool_method == "spatial_learned_embeddings":
            if self.spatial_block_size is None:
                raise ValueError(
                    "spatial_block_size must be set when using spatial_learned_embeddings"
                )
            x = nn.Conv(                # 512 to num_kp features (less complexity)
                features=self.num_kp,
                kernel_size=1,
                use_bias=False,
                dtype=jnp.float32,
                kernel_init=nn.initializers.kaiming_normal(),
                name="SLE_1xconv",
            )(x)
            x = SpatialLearnedEmbeddings(*(x.shape[-3:]), self.spatial_block_size)(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x


small_configs = {"small": SmallEncoder}
