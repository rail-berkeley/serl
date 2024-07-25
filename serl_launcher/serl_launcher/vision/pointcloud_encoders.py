from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.lax as lax

import jax


class PointNetBasic(nn.Module):
    """ Basic PointNet, input is BxNx3, output Bx40 """
    @nn.compact
    def __call__(
            self,
            observations: jnp.ndarray,
            train: bool = True,
    ):
        x = observations

        num_points = observations.shape[-2]

        conv = partial(
            nn.Conv,
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros_init(),
            padding="VALID"
        )
        batchnorm = partial(
            nn.BatchNorm,
            momentum=0.9,
            epsilon=1e-5,
            use_running_average=not train,
        )
        dense = partial(
            nn.Dense,
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal(),
        )

        x = x[..., None]  # expand dims
        # shape (B, N, 3, 1)

        x = conv(
            features=64,
            kernel_size=(1, 3),
            strides=(1, 1),
            name="conv1"
        )(x)
        # shape (B, N, 1, 64)

        x = batchnorm(
            name="bn1"
        )(x)

        # conv2, conv3, conv4, conv5
        for i in range(2, 6):
            features = {2: 64, 3: 64, 4: 128, 5: 1024}
            x = conv(
                features=features[i],
                kernel_size=(1, 1),
                strides=(1, 1),
                name=f"conv{i}"
            )(x)

            x = batchnorm(
                name=f"bn{i}"
            )(x)

        # shape (B, N, 1, 1024)
        x = nn.max_pool(x, (num_points, 1))

        #  shape (B, 1, 1, 1024)
        x = dense(features=512, name="fc1")(x)
        x = nn.relu(x)

        x = dense(features=256, name="fc2")(x)
        x = nn.relu(x)

        x = nn.Dropout(rate=0.3, name="dp1", deterministic=not train)(x)
        x = nn.relu(x)

        x = dense(features=40, name="fn3")(x)
        # TODO original has no activation, for me it would me sense
        return x


