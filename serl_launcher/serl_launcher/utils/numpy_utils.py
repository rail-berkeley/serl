import numpy as np
import jax.numpy as jnp

"""
numpy uses uint8 to save a boolean array, so it is really inefficient.
with this simple conversion, we increase the efficiency by 8 fold.
"""


def bool_2_int8(array: np.ndarray) -> np.ndarray:
    # from bool (x, y, z) to uint8 (x, y, z/8)
    assert array.shape[-1] % 8 == 0
    return np.dot(array.reshape((*array.shape[:-1], array.shape[-1] // 8, 8)), 2 ** np.arange(8)).astype(np.uint8)


def int8_2_bool(array: np.ndarray) -> np.ndarray:
    bool_arr = np.zeros((*array.shape[:], 8)).astype(np.bool_)
    for i in range(8):
        bool_arr[..., i] = np.bitwise_and(np.right_shift(array, i), 0x1)
    return bool_arr.reshape(*array.shape[:-1], array.shape[-1] * 8)


def int8_2_bool_jnp(array: jnp.ndarray) -> jnp.ndarray:
    bool_arr = []
    for i in range(8):
        bool_arr.append(jnp.bitwise_and(jnp.right_shift(array, i), 0x1))
    bool_arr = jnp.stack(bool_arr, axis=-1)
    return bool_arr.reshape(*array.shape[:-1], array.shape[-1] * 8)


def bool_2_int8_jnp(array: jnp.ndarray) -> jnp.ndarray:
    assert array.shape[-1] % 8 == 0
    return jnp.dot(array.reshape((*array.shape[:-1], array.shape[-1] // 8, 8)), 2 ** np.arange(8)).astype(np.uint8)
