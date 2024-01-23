import numpy as np
import gymnasium as gym
import time
import requests
import copy

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.pcb_env.config import PCBEnvConfig


class FrankaPCBInsert(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=PCBEnvConfig)

    def crop_image(self, name, image):
        """Crop realsense images to be a square."""
        if name == "wrist_1":
            return image[90:390, 170:470, :]
        elif name == "wrist_2":
            return image[90:390, 170:470, :]
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def go_to_rest(self, joint_reset=False):
        self.update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # Move up to clear the slot
        self.update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.03
        self.interpolate_move(reset_pose, timeout=1)

        # Change to precision mode for reset
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._TARGET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1.5)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1.5)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
