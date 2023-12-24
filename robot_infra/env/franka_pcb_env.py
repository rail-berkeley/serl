import gym
from gym import spaces
import numpy as np
# from franka.scripts.spacemouse_teleop import SpaceMouseExpert
import time
from franka_robotiq_env import FrankaRobotiq
import copy
import requests

class PCBEnv(FrankaRobotiq):
    def __init__(self):

        super().__init__()
        self._TARGET_POSE = [0.6479450830785974,0.17181947852969695,0.056419218166284224, 3.1415, 0.0, 0.0 ]
        self._REWARD_THRESHOLD = [0.005, 0.005, 0.0006, 0.03, 0.03,  0.05]
        self.observation_space = spaces.Dict(
            {
                "state_observation": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(6,)), # xyz + euler
                        "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "image_observation": spaces.Dict(
                    {
                    "wrist_1": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                    "wrist_1_full": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    "wrist_2": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                    "wrist_2_full": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            np.array((-0.01, -0.01, -0.01, -0.05, -0.05, -0.05)),
            np.array((0.01, 0.01, 0.01, 0.05, 0.05, 0.05))
        )
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.62, 0.15, 0.03)),
            np.array((0.67, 0.19, 0.09)),
            dtype=np.float64
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((np.pi-0.15, -0.05, -0.1)),
            np.array((np.pi+0.1, 0.15, 0.1)),
            dtype=np.float64
        )
        self.resetpos[:3] = np.array([0.645, 0.17, 0.07])
        self.resetpos[3:] = self.euler_2_quat(np.pi, 0.03, 0)
        self.episodes = 1
        self.randomreset = False

    def _get_state(self):
        state = super()._get_state()
        state.pop('gripper_pose')
        return state

    def go_to_rest(self, jpos=False):
        count = 0
        if self.currpos[2] < 0.06:
            restp_new = copy.deepcopy(self.currpos)
            restp_new[2] += 0.02
            dp = restp_new - self.currpos
            while count < 200 and (
                np.linalg.norm(dp[:3]) > 0.01 or np.linalg.norm(dp[3:]) > 0.03
            ):
                if np.linalg.norm(dp[3:]) > 0.02:
                    dp[3:] = 0.05 * dp[3:] / np.linalg.norm(dp[3:])
                if np.linalg.norm(dp[:3]) > 0.02:
                    dp[:3] = 0.02 * dp[:3] / np.linalg.norm(dp[:3])
                self._send_pos_command(self.currpos + dp)
                time.sleep(0.1)
                self.update_currpos()
                dp = restp_new - self.currpos
                count += 1


        requests.post(self.url + "precision_mode")
        if jpos:
            restp_new = copy.deepcopy(self.currpos)
            restp_new[2] = 0.2
            dp = restp_new - self.currpos
            count_1 = 0
            self._send_pos_command(self.currpos)
            requests.post(self.url + "precision_mode")
            while (
                (np.linalg.norm(dp[:3]) > 0.03 or np.linalg.norm(dp[3:]) > 0.04)
            ) and count_1 < 50:
                if np.linalg.norm(dp[3:]) > 0.05:
                    dp[3:] = 0.05 * dp[3:] / np.linalg.norm(dp[3:])
                if np.linalg.norm(dp[:3]) > 0.03:
                    dp[:3] = 0.03 * dp[:3] / np.linalg.norm(dp[:3])
                self._send_pos_command(self.currpos + dp)
                time.sleep(0.1)
                self.update_currpos()
                dp = restp_new - self.currpos
                count_1 += 1

            print("JOINT RESET")
            requests.post(self.url + "jointreset")
        else:
            # print("RESET")
            restp = copy.deepcopy(self.resetpos[:])
            if self.randomreset:
                restp[:2] += np.random.uniform(-0.005, 0.005, (2,))
                restp[2] += np.random.uniform(-0.005, 0.005, (1,))
                # restyaw += np.random.uniform(-np.pi / 6, np.pi / 6)
                # restp[3:] = self.euler_2_quat(np.pi, 0, restyaw)

            restp_new = copy.deepcopy(restp)
            restp_new[2] = 0.07         #PCB
            self.update_currpos()
            dp = restp_new - self.currpos
            while count < 200 and (
                np.linalg.norm(dp[:3]) > 0.005 or np.linalg.norm(dp[3:]) > 0.03
            ):
                if np.linalg.norm(dp[3:]) > 0.02:
                    dp[3:] = 0.05 * dp[3:] / np.linalg.norm(dp[3:])
                if np.linalg.norm(dp[:3]) > 0.02:
                    dp[:3] = 0.02 * dp[:3] / np.linalg.norm(dp[:3])
                self._send_pos_command(self.currpos + dp)
                time.sleep(0.1)
                self.update_currpos()
                dp = restp_new - self.currpos
                count += 1

            requests.post(self.url + "pcb_compliance_mode")
        return count < 200