"""Gym Interface for Franka"""
import numpy as np
import gym
from pyquaternion import Quaternion
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
from gym import core, spaces
from camera.video_capture import VideoCapture
from camera.rs_capture import RSCapture
import queue
from PIL import Image
from queue import Queue
import threading
import os

class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.stop_signal = False
        self.daemon = True  # make this a daemon thread

        self.video = []

        video_dir = '/home/undergrad/franka_fwbw_pick_screw_vice_ckpts'
        os.makedirs(video_dir, exist_ok=True)
        uuid = time.strftime("%Y%m%d-%H%M%S")
        self.wrist1 = cv2.VideoWriter(os.path.join(video_dir, f'wrist_1_{uuid}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 480))
        self.wrist2 = cv2.VideoWriter(os.path.join(video_dir, f'wrist_2_{uuid}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 480))
        self.frame_counter = 0

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break
            # pair1 = np.concatenate([img_array['wrist_1_full'], img_array['wrist_2_full']], axis=0)
            pair1 = np.concatenate([img_array['wrist_1'], img_array['wrist_2']], axis=0)
            # pair1 = np.concatenate([img_array['wrist_1'], img_array['wrist_2'], img_array['side_1']], axis=0)
            # pair2 = np.concatenate([img_array['side_2_full'], img_array['side_1_full']], axis=0)
            # concatenated = np.concatenate([pair1, pair2], axis=1)
            cv2.imshow('wrist', pair1/255.)
            cv2.waitKey(1)

            self.wrist1.write(img_array['wrist_1_full'])
            self.wrist2.write(img_array['wrist_2_full'])
            self.frame_counter += 1
            if self.frame_counter == 400:
                self.wrist1.release()
                self.wrist2.release()


class FrankaRobotiq(gym.Env):
    def __init__(
        self,
        randomReset=False,
        hz=10,
        start_gripper=0,
    ):

        self._TARGET_POSE = [0.6636488814118523,0.05388642290645651,0.09439445897864279, 3.1339503, 0.009167, 1.5550434]
        self._REWARD_THRESHOLD = [0.01, 0.01, 0.01, 0.2, 0.2,  0.2]
        self.resetpos = np.zeros(7)

        self.resetpos[:3] = self._TARGET_POSE[:3]
        self.resetpos[2] += 0.07
        self.resetpos[3:] = self.euler_2_quat(self._TARGET_POSE[3], self._TARGET_POSE[4], self._TARGET_POSE[5])

        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))
        self.currjacobian = np.zeros((6, 7))
        self.start_gripper = start_gripper
        self.currgrip = self.start_gripper #start_gripper
        self.lastsent = time.time()
        self.randomreset = randomReset
        self.actionnoise = 0
        self.hz = hz

        ## NUC
        self.ip = "127.0.0.1"
        self.url = "http://" + self.ip + ":5000/"

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

        self.observation_space = spaces.Dict(
            {
                "state_observation": spaces.Dict(
                    {
                        # "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                        "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(6,)), # xyz + euler
                        "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": spaces.Box(-1, 1, shape=(1,)),
                        # "q": spaces.Box(-np.inf, np.inf, shape=(7,)),
                        # "dq": spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,)),
                        # "jacobian": spaces.Box(-np.inf, np.inf, shape=((6, 7))),
                    }
                ),
                "image_observation": spaces.Dict(
                    {
                    "wrist_1": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                    "wrist_1_full": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    "wrist_2": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                    "wrist_2_full": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    # "side_1": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                    # "side_1_full": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    }
                ),
            }
        )
        self.cycle_count = 0
        self.cap_wrist_1 = VideoCapture(RSCapture(name='wrist_1', serial_number='130322274175', depth=False))
        self.cap_wrist_2 = VideoCapture(RSCapture(name='wrist_2', serial_number='127122270572', depth=False))
        # self.cap_side_1 = VideoCapture(RSCapture(name='side_1', serial_number='128422272758', depth=False))

        # self.cap_side_1 = VideoCapture(
        #     RSCapture(name="side_1", serial_number="128422270679", depth=True)
        # )
        # self.cap_side_2 = VideoCapture(
        #     RSCapture(name="side_2", serial_number="127122270146", depth=True)
        # )
        self.cap = {
            # "side_1": self.cap_side_1,
            # "side_2": self.cap_side_2,
            "wrist_1": self.cap_wrist_1,
            "wrist_2": self.cap_wrist_2,
        }

        self.img_queue = queue.Queue()
        self.displayer = ImageDisplayer(self.img_queue)
        self.displayer.start()
        print("Initialized Franka")

    def recover(self):
        requests.post(self.url + "clearerr")

    def _send_pos_command(self, pos):
        self.recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def update_currpos(self):
        ps = requests.post(self.url + "getstate").json()
        self.currpos[:] = np.array(ps["pose"])
        self.currvel[:] = np.array(ps["vel"])

        self.currforce[:] = np.array(ps["force"])
        self.currtorque[:] = np.array(ps["torque"])
        self.currjacobian[:] = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q[:] = np.array(ps["q"])
        self.dq[:] = np.array(ps["dq"])

    def set_gripper(self, position, block=True):
        if position == 1:
            st = 'close'
        elif position == 0:
            st = 'open'
        else:
            raise ValueError(f'Gripper position {position} not supported')

        ### IMPORTANT, IF FRANKA GRIPPER GETS OPEN/CLOSE COMMANDS TOO QUICKLY IT WILL FREEZE
        ### THIS MAKES SURE CONSECUTIVE GRIPPER CHANGES ONLY HAPPEN 1 SEC APART
        now = time.time()
        delta = now - self.lastsent
        if delta >= 1:
            requests.post(self.url + st)
            self.lastsent = time.time()
            self.currgrip = position
        # time.sleep(max(0, 1.5 - delta))


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

        return pose

    def move_to_pos(self, pos):
        start_time = time.time()
        self._send_pos_command(self.clip_safety_box(pos))
        dl = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dl))
        self.update_currpos()
        obs = self._get_obs()
        return obs

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

        # self.nextpos = self.clip_safety_box(self.nextpos)
        # self._send_pos_command(self.nextpos)
        self._send_pos_command(self.clip_safety_box(self.nextpos))
        # self.set_gripper(action[-1])

        self.curr_path_length += 1
        dl = time.time() - start_time

        time.sleep(max(0, (1.0 / self.hz) - dl))

        self.update_currpos()
        ob = self._get_obs()
        obs_xyz = ob['state_observation']['tcp_pose'][:3]
        # obs_rpy = self.quat_2_euler(ob['state_observation']['tcp_pose'][3:7])
        obs_rpy = ob['state_observation']['tcp_pose'][3:]
        reward = self.binary_reward_tcp(ob['state_observation']['tcp_pose'])
        done = self.curr_path_length >= 100
        # if not self.xyz_bounding_box.contains(obs_xyz) or not self.rpy_bounding_box.contains(obs_rpy):
        #     # print('Truncated: Bouding Box')
        #     # print("xyz: ", self.xyz_bounding_box.contains(obs_xyz), obs_xyz)
        #     # print("rortate: ", self.rpy_bounding_box.contains(obs_rpy), obs_rpy)
        #     return ob, 0, True, True, {}
        # return ob, int(reward), done or reward, done, {}
        return ob, int(reward), done, done, {}


    def binary_reward_tcp(self, current_pose,):
        # euler_angles = np.abs(R.from_quat(current_pose[3:]).as_euler("xyz"))
        euler_angles = np.abs(current_pose[3:])
        current_pose = np.hstack([current_pose[:3],euler_angles])
        delta = np.abs(current_pose - self._TARGET_POSE)
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {_REWARD_THRESHOLD}')
            return False

    def get_im(self):
        images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                # images[key] = cv2.resize(rgb, self.observation_space['image_observation'][key].shape[:2][::-1])
                # if key == 'wrist_1':
                #     cropped_rgb = rgb[ 0:300, 150:450, :]
                # if key == 'wrist_2':
                #     cropped_rgb = rgb[ 50:350, 150:450, :]
                if key == 'wrist_1':
                    cropped_rgb = rgb[:, 80:560, :]
                if key == 'wrist_2':
                    cropped_rgb = rgb[:, 80:560, :]
                images[key] = cv2.resize(cropped_rgb, self.observation_space['image_observation'][key].shape[:2][::-1])
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
                    cap = RSCapture(name='wrist_2', serial_number='127122270572', depth=False)
                else:
                    raise KeyError
                self.cap[key] = VideoCapture(cap)
                return self.get_im()

        self.img_queue.put(images)
        return images

    def _get_state(self):
        state_observation = {
            "tcp_pose": np.concatenate((self.currpos[:3], self.quat_2_euler(self.currpos[3:]))),
            "tcp_vel": self.currvel,
            "gripper_pose": self.currgrip,
            # "q": self.q,
            # "dq": self.dq,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
            # "jacobian": self.currjacobian,
        }
        return state_observation

    def _get_obs(self):
        images = self.get_im()
        state_observation = self._get_state()

        return copy.deepcopy(dict(
                image_observation=images,
                state_observation=state_observation
            ))

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
            restp_new[2] = 0.2        #PEG
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

    def reset(self, jpos=False, gripper=None, require_input=False):
        self.cycle_count += 1
        if self.cycle_count % 150 == 0:
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

    def quat_2_euler(self, quat):
        # calculates and returns: yaw, pitch, roll from given quaternion
        if not isinstance(quat, Quaternion):
            quat = Quaternion(quat)
        yaw, pitch, roll = quat.yaw_pitch_roll
        return yaw + np.pi, pitch, roll

    def euler_2_quat(self, yaw=np.pi / 2, pitch=0.0, roll=np.pi):
        yaw = np.pi - yaw
        yaw_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0, 0, 1.0],
            ]
        )
        pitch_matrix = np.array(
            [
                [np.cos(pitch), 0.0, np.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        roll_matrix = np.array(
            [
                [1.0, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
        return Quaternion(matrix=rot_mat).elements

    def close_camera(self):
        # self.cap_top.close()
        # self.cap_side.close()
        self.cap_wrist_2.close()
        self.cap_wrist_1.close()