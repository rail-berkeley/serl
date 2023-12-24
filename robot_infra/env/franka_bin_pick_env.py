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

class BinPickEnv(FrankaRobotiq):
    def __init__(self):
        super().__init__()
        # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.44, -0.12, 0.04)), np.array((0.53, 0.12, 0.1)), dtype=np.float64
        )
        self.rpy_bounding_box = gym.spaces.Box(
            # np.array((np.pi-0.001, 0-0.001, np.pi/4)),
            # np.array((np.pi+0.001, 0+0.001, 3*np.pi/4)),
            np.array((np.pi-0.001, 0-0.001, 0-0.01)),
            np.array((np.pi+0.001, 0+0.001, 0+0.01)),
            dtype=np.float64,
        )
        self.inner_box = gym.spaces.Box(
            np.array([0.44, -0.04, 0.04]),
            np.array([0.53,  0.04, 0.08]),
            dtype=np.float64
        )
        self.drop_box = gym.spaces.Box(
            np.array([0.44, -0.04]),
            np.array([0.53,  0.04]),
            dtype=np.float64
        )
        ## Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.array((-0.03, -0.03, -0.03, -0.05, -0.05, -0.2, -1)),
            np.array((0.03, 0.03, 0.03, 0.05, 0.05, 0.2, 1)),
        )
        # enable gripper in observation space
        self.observation_space['state_observation']['gripper_pose'] = spaces.Box(-np.inf, np.inf, shape=(1,))
        self.centerpos = copy.deepcopy(self.resetpos)
        self.centerpos[:3] = np.mean((self.xyz_bounding_box.high, self.xyz_bounding_box.low), axis=0) #np.array([0.55,-0.05,0.09])
        self.centerpos[2] += 0.01
        self.resetpos = copy.deepcopy(self.centerpos)
        self.resetpos[3:] = self.euler_2_quat(np.pi, 0., 0)

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
            restp_new[2] = 0.13        #cable
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

    def clip_safety_box(self, pose):
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )

        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
        old_sign = np.sign(euler[0])
        euler[0] = (
            np.clip(
                euler[0] * old_sign,
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
            * old_sign
        )
        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        # Clip xyz to inner box
        if self.inner_box.contains(pose[:3]):
            print(f'Command: {pose[:3]}')
            pose[:3] = self.intersect_line_bbox(self.currpos[:3], pose[:3], self.inner_box.low, self.inner_box.high)
            print(f'Clipped: {pose[:3]}')

        return pose

    def intersect_line_bbox(self, p1, p2, bbox_min, bbox_max):
        # Define the parameterized line segment
        # P(t) = p1 + t(p2 - p1)
        tmin = 0
        tmax = 1

        for i in range(3):
            if p1[i] < bbox_min[i] and p2[i] < bbox_min[i]:
                return None
            if p1[i] > bbox_max[i] and p2[i] > bbox_max[i]:
                return None
            
            # For each axis (x, y, z), compute t values at the intersection points
            if abs(p2[i] - p1[i]) > 1e-10:  # To prevent division by zero
                t1 = (bbox_min[i] - p1[i]) / (p2[i] - p1[i])
                t2 = (bbox_max[i] - p1[i]) / (p2[i] - p1[i])
                
                # Ensure t1 is smaller than t2
                if t1 > t2:
                    t1, t2 = t2, t1
                
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                
                if tmin > tmax:
                    return None

        # Compute the intersection point using the t value
        intersection = p1 + tmin * (p2 - p1)

        return intersection

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

        gripper = action[-1]
        if gripper > 0:
            if not self.drop_box.contains(self.currpos[:2]):
                gripper = (self.currgrip + 1) % 2
                self.set_gripper(gripper)

        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dl = time.time() - start_time

        time.sleep(max(0, (1.0 / self.hz) - dl))

        self.update_currpos()
        ob = self._get_obs()
        obs_xyz = ob['state_observation']['tcp_pose'][:3]
        obs_rpy = ob['state_observation']['tcp_pose'][3:]
        reward = 0
        done = self.curr_path_length >= 40 #100
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
        requests.post(self.url + "open")
        self.currgrip = 0
        time.sleep(1)

        self.update_currpos()
        # self.last_quat = self.currpos[3:]
        o = self._get_obs()
        return o, {}