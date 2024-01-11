import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import requests
import copy
import cv2
import queue

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.envs.franka_robotiq_env import FrankaRobotiq
from franka_env.utils.rotations import euler_2_quat


class FrankaRobotiqPCBInsert(FrankaRobotiq):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._TARGET_POSE = np.array(
            [
                0.5668657154487453,
                0.002050321710641817,
                0.05462772570641611,
                # 0.9998817571282431,-0.010581843089284471,0.008902600824764906,0.0067260729646395475
                3.1279511,
                0.0176617,
                0.0212859,
            ]
        )
        self.resetpos[:3] = self._TARGET_POSE[:3]
        self.resetpos[2] += 0.04
        self.resetpos[3:] = euler_2_quat(self._TARGET_POSE[3:])

        # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            self._TARGET_POSE[:3]
            - np.array([self.random_xy_range, self.random_xy_range, 0.005]),
            self._TARGET_POSE[:3]
            + np.array([self.random_xy_range, self.random_xy_range, 0.05]),
            dtype=np.float32,
        )
        rpy_delta_range = np.array([0.05, 0.05, self.random_rz_range])
        self.rpy_bounding_box = gym.spaces.Box(
            self._TARGET_POSE[3:] - rpy_delta_range,
            self._TARGET_POSE[3:] + rpy_delta_range,
            dtype=np.float32,
        )

        self._REWARD_THRESHOLD = [0.005, 0.005, 0.001, 0.1, 0.1, 0.1]
        self.action_scale = (0.02, 0.2, 1)

    def crop_image(self, image):
        return image[90:390, 170:470, :]

    def go_to_rest(self, jpos=False):
        # requests.post(self.url + "pcb_compliance_mode")
        self.update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.03
        self.interpolate_move(reset_pose, timeout=1.5)

        # time.sleep(2)
        requests.post(self.url + "precision_mode")
        time.sleep(1)  # wait for mode switching

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
            self.interpolate_move(reset_pose, timeout=1)

        if jpos:
            requests.post(self.url + "precision_mode")
            print("JOINT RESET")
            requests.post(self.url + "jointreset")

        requests.post(self.url + "pcb_compliance_mode")
        return True
