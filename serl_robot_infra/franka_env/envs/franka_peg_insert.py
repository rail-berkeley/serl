import numpy as np
import gymnasium as gym

from franka_env.envs.franka_env import FrankaEnv


class FrankaPegInsert(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            self._TARGET_POSE[:3]
            - np.array([self.random_xy_range, self.random_xy_range, 0]),
            self._TARGET_POSE[:3]
            + np.array([self.random_xy_range, self.random_xy_range, 0.1]),
            dtype=np.float32,
        )
        rpy_delta_range = np.array([0.01, 0.01, self.random_rz_range])
        self.rpy_bounding_box = gym.spaces.Box(
            self._TARGET_POSE[3:] - rpy_delta_range,
            self._TARGET_POSE[3:] + rpy_delta_range,
            dtype=np.float32,
        )
