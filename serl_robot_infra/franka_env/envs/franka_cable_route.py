import numpy as np
import gymnasium as gym
import time
import requests
import copy

from franka_env.envs.franka_env import FrankaEnv, FrankaEnvConfig
from franka_env.utils.rotations import euler_2_quat


# TODO: Move this to a example config file, and have an example script
class FrankaCableRouteConfig(FrankaEnvConfig):
    TARGET_POSE = np.array(
        [
            0.460639895728905,
            -0.02439473272513422,
            0.026321125814908725,
            3.1331234,
            0.0182487,
            1.5824805,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    ACTION_SCALE = (0.05, 0.3, 1)


class FrankaCableRoute(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=FrankaCableRouteConfig)
        # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            self._TARGET_POSE[:3]
            - np.array([self.random_xy_range, self.random_xy_range, 0.001]),
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

    def crop_image(self, image):
        return image[:, 80:560, :]

    def go_to_rest(self, jpos=False):
        self.update_currpos()
        self._send_pos_command(self.clip_safety_box(self.currpos))
        time.sleep(0.5)

        requests.post(self.url + "precision_mode")
        time.sleep(0.5)  # wait for mode switching

        self.update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.05
        self.interpolate_move(reset_pose, timeout=1)

        reset_pose = self.resetpos.copy()
        self.interpolate_move(reset_pose, timeout=1)

        # perform random reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._TARGET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1.5)

        if jpos:
            requests.post(self.url + "precision_mode")
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)
            self.interpolate_move(self.resetpos, timeout=5)

        requests.post(self.url + "cable_wrap_compliance_mode")
        return True
