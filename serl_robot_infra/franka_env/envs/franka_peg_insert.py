import numpy as np
import gymnasium as gym
from gymnasium import spaces

from franka_env.envs.franka_robotiq_env import FrankaRobotiq
from franka_env.utils.rotations import euler_2_quat


class FrankaRobotiqPegInsert(FrankaRobotiq):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._TARGET_POSE = np.array(
            [
                0.5906439143742067,
                0.07771711953459341,
                0.0937835826958042,
                # 0.8391759571503397,0.5437992747665743,0.002890933345090582,0.007596328623303431
                3.1099675,
                0.0146619,
                -0.0078615,
            ]
        )
        self.resetpos[:3] = self._TARGET_POSE[:3]
        self.resetpos[2] += 0.1
        self.resetpos[3:] = euler_2_quat(self._TARGET_POSE[3:])

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
