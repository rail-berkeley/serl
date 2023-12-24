import gym
from gym import spaces
import numpy as np
from franka_robotiq_env import FrankaRobotiq
import time
from scipy.spatial.transform import Rotation
import requests
import copy
import cv2
from camera.video_capture import VideoCapture
from camera.rs_capture import RSCapture
import queue

class RouteCableEnv(FrankaRobotiq):
    def __init__(self):
        super().__init__()
        # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.51, -0.1, 0.04)), np.array((0.59, 0, 0.12)), dtype=np.float64
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((np.pi-0.001, 0-0.001, np.pi/4)),
            np.array((np.pi+0.001, 0+0.001, 3*np.pi/4)),
            dtype=np.float64,
        )
        ## Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.array((-0.02, -0.02, -0.02, -0.05, -0.05, -0.1, -1)),
            np.array((0.02, 0.02, 0.02, 0.05, 0.05, 0.1, 1)),
        )
        # enable gripper in observation space
        self.observation_space['state_observation']['gripper_pose'] = spaces.Box(-np.inf, np.inf, shape=(1,))
        # [0.48012088982197254,-0.07218941280725254,0.11078303293108258,0.6995269546628874,0.7134059993136379,0.028532587996196627,0.029996854262000595]
        self.resetpos[:3] = np.array([0.55,-0.05,0.09])
        self.resetpos[3:] = self.euler_2_quat(np.pi, 0.03, np.pi/2)

    def go_to_rest(self, jpos=False):
        count = 0
        requests.post(self.url + "precision_mode")
        if jpos:
            restp_new = copy.deepcopy(self.currpos)
            restp_new[2] = 0.3
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
            self.update_currpos()
            restp = copy.deepcopy(self.resetpos[:])
            if self.randomreset:
                restp[:2] += np.random.uniform(-0.005, 0.005, (2,))
                restp[2] += np.random.uniform(-0.005, 0.005, (1,))
                # restyaw += np.random.uniform(-np.pi / 6, np.pi / 6)
                # restp[3:] = self.euler_2_quat(np.pi, 0, restyaw)

            restp_new = copy.deepcopy(restp)
            restp_new[2] = 0.15        #cable
            dp = restp_new - self.currpos

            height = np.zeros_like(self.resetpos)
            height[2] = 0.02
            while count < 10:
                self._send_pos_command(self.currpos + height)
                time.sleep(0.1)
                self.update_currpos()
                count += 1

            count = 0
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

            dp = restp - self.currpos
            count = 0
            while count < 20 and (
                np.linalg.norm(dp[:3]) > 0.01 or np.linalg.norm(dp[3:]) > 0.01
            ):
                if np.linalg.norm(dp[3:]) > 0.05:
                    dp[3:] = 0.05 * dp[3:] / np.linalg.norm(dp[3:])
                if np.linalg.norm(dp[:3]) > 0.02:
                    dp[:3] = 0.02 * dp[:3] / np.linalg.norm(dp[:3])
                self._send_pos_command(self.currpos + dp)
                time.sleep(0.1)
                self.update_currpos()
                dp = restp - self.currpos
                count += 1
            requests.post(self.url + "peg_compliance_mode")
        return count < 50

    def get_im(self):
        images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                # images[key] = cv2.resize(rgb, self.observation_space['image_observation'][key].shape[:2][::-1])
                if key == 'wrist_1':
                    # cropped_rgb = rgb[ 100:400, 50:350, :]
                    cropped_rgb = rgb[:, 80:560, :]
                if key == 'wrist_2':
                    # cropped_rgb = rgb[ 50:350, 200:500, :] #150:450
                    cropped_rgb = rgb[:, 80:560, :]
                # if key == 'side_1':
                #     cropped_rgb = rgb[150:330, 230:410, :]

                images[key] = cv2.resize(cropped_rgb, self.observation_space['image_observation'][key].shape[:2][::-1])
                # images[key] = cv2.resize(rgb, self.observation_space['image_observation'][key].shape[:2][::-1])
                images[key + "_full"] = rgb
                # images[f"{key}_depth"] = depth
            except queue.Empty:
                input(f'{key} camera frozen. Check connect, then press enter to relaunch...')
                cap.close()
                # if key == 'side_1':
                #     cap = RSCapture(name='side_1', serial_number='128422270679', depth=True)
                # elif key == 'side_2':
                #     cap = RSCapture(name='side_2', serial_number='127122270146', depth=True)
                if key == 'wrist_1':
                    cap = RSCapture(name='wrist_1', serial_number='130322274175', depth=False)
                elif key == 'wrist_2':
                    # cap = RSCapture(name='wrist_2', serial_number='127122270572', depth=False)
                    cap = RSCapture(name='wrist_2', serial_number='127122270572', depth=False)
                elif key == 'side_1':
                    cap = RSCapture(name='side_1', serial_number='128422272758', depth=False)
                else:
                    raise KeyError
                self.cap[key] = VideoCapture(cap)
                return self.get_im()

        self.img_queue.put(images)
        return images

    def step(self, action):
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.actionnoise > 0:
            a = action[:3] + np.random.uniform(
                -self.actionnoise, self.actionnoise, (3,)
            )
        else:
            a = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + a

        ### GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        self._send_pos_command(self.clip_safety_box(self.nextpos))
        # only change the gripper if the action is above a threshold, either open or close
        if len(action) == 7:
            if action[-1] > 0.8:
                self.set_gripper(1)
            elif action[-1] < -0.8:
                self.set_gripper(0)

        self.curr_path_length += 1
        dl = time.time() - start_time

        time.sleep(max(0, (1.0 / self.hz) - dl))

        self.update_currpos()
        ob = self._get_obs()
        obs_xyz = ob['state_observation']['tcp_pose'][:3]
        obs_rpy = ob['state_observation']['tcp_pose'][3:]
        reward = 0
        done = self.curr_path_length >= 30 #100
        # if not self.xyz_bounding_box.contains(obs_xyz) or not self.rpy_bounding_box.contains(obs_rpy):
        #     # print('Truncated: Bouding Box')
        #     print("xyz: ", self.xyz_bounding_box.contains(obs_xyz), obs_xyz)
        #     print("rortate: ", self.rpy_bounding_box.contains(obs_rpy), obs_rpy)
        #     return ob, 0, True, True, {}
        return ob, int(reward), done, done, {}

    def reset(self, jpos=False, gripper=None, require_input=False):
        self.cycle_count += 1
        if self.cycle_count % 1500 == 0:
            self.cycle_count = 0
            jpos=True
        # requests.post(self.url + "reset_gripper")
        # time.sleep(3)
        # self.set_gripper(self.start_gripper, block=False)
        self.currgrip = self.start_gripper

        success = self.go_to_rest(jpos=jpos)
        self.update_currpos()
        self.curr_path_length = 0
        self.recover()
        if jpos == True:
            self.go_to_rest(jpos=False)
            self.update_currpos()
            self.recover()

        if require_input:
            input("Reset Environment, Press Enter Once Complete: ")
        # print("RESET COMPLETE")
        self.update_currpos()
        # self.last_quat = self.currpos[3:]
        o = self._get_obs()
        return o, {}


class ResetCableEnv(FrankaRobotiq):
    def __init__(self):
        super().__init__()
        # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.62, 0.0, 0.05)), np.array((0.71, 0.08, 0.3)), dtype=np.float64
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((np.pi-0.1, -0.1, 1.35)),
            np.array((np.pi+0.1, 0.1, 1.7)),
            dtype=np.float64,
        )
        ## Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.array((-0.02, -0.02, -0.02, -0.05, -0.05, -0.05, 0 - 1e-8)),
            np.array((0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 1 + 1e-8)),
        )
        # self.resetpos[:3] = np.array([0.645, 0.17, 0.07])
        # self.resetpos[3:] = self.euler_2_quat(np.pi, 0.03, 0)