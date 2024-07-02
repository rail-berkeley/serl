from typing import Any, Callable, Optional, Sequence, Tuple, Optional

import flax.linen as nn
import jax.numpy as jnp
import jax


class RangeSensorEncoder(nn.Module):
    """
    Takes depth-images as input with shape (B, H, W, 1) and returns range values for each keypoint tuple provided. The
    values are pooled with 'func_pool' over a size of 'keypoint_size'. The output is scaled to [0., 1.]
    """
    keypoints: Tuple[Tuple[int, int], ...]  # (x, y) coordinates in depth image
    keypoint_size: Tuple[int, int] = (3, 3)
    pool_func: Callable = jnp.median

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False, encode: bool = True):
        x = observations / 255.  # convert to float and within [0, 1]

        assert x.shape[-1] == 1
        assert self.keypoint_size[0] % 2 == 1 and self.keypoint_size[1] % 2 == 1

        # add batch dim if missing
        if len(x.shape) < 4:
            x = x[None]

        points = []
        for i, keypoint in enumerate(self.keypoints):
            k_x, k_y = keypoint
            s_x, s_y = self.keypoint_size[0] // 2, self.keypoint_size[1] // 2
            points.append(self.pool_func(x[:, k_x - s_x:k_x + s_x + 1, k_y - s_y:k_y + s_y + 1, 0], axis=(1, 2)))
        points = jnp.stack(points, axis=-1)

        if points.shape[0] == 1:
            points = points[0]

        return points
