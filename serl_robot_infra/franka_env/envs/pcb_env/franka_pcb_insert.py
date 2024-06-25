import numpy as np
import gym
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
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._send_gripper_command(-1)
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # Move up to clear the slot
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.03
        self.interpolate_move(reset_pose, timeout=1)

        # Change to precision mode for reset
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # execute the go_to_rest method from the parent class
        super().go_to_rest(joint_reset)
