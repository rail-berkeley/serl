"""Gym Interface for Franka"""
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
from gymnasium import spaces
import queue
import threading
from datetime import datetime

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler


class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [img_array["wrist_1_full"], img_array["wrist_2_full"]], axis=0
            )

            cv2.imshow("RealSense Cameras", frame)
            cv2.waitKey(1)


class DefaultEnvConfig:
    """Default configuration for FrankaRobotiqEnv."""

    SERVER_URL = "http://127.0.0.1:5000/"
    WRIST_CAM1_SERIAL = "130322274175"
    WRIST_CAM2_SERIAL = "127122270572"
    TARGET_POSE = [
        0.5907729022946797,
        0.05342705145048531,
        0.09071618754222505,
        3.1339503,
        0.009167,
        1.5550434,
    ]
    REWARD_THRESHOLD = [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
    ACTION_SCALE = (0.02, 0.1, 1)


class FrankaRobotiqEnv(gym.Env):
    def __init__(
        self,
        randomReset=False,
        random_xy_range=0.05,
        random_rz_range=np.pi / 36,
        hz=10,
        start_gripper=0,
        fake_env=False,
        save_video=False,
        config=DefaultEnvConfig,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config

        self.resetpos = np.zeros(7)

        self.resetpos[:3] = self._TARGET_POSE[:3]
        self.resetpos[2] += 0.2
        self.resetpos[3:] = euler_2_quat(self._TARGET_POSE[3:])

        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))
        self.currjacobian = np.zeros((6, 7))

        self.curr_gripper_pos = 0
        self.lastsent = time.time()
        self.randomreset = randomReset
        self.random_xy_range = random_xy_range
        self.random_rz_range = random_rz_range
        self.hz = hz
        self.joint_reset_cycle = 200  # reset the robot joint every 200 cycles

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.56, 0.0, 0.05)), np.array((0.62, 0.08, 0.2)), dtype=np.float64
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((np.pi - 0.01, -0.01, 1.35)),
            np.array((np.pi + 0.01, 0.01, 1.7)),
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": spaces.Dict(
                    {
                        "wrist_1": spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                        "wrist_2": spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.init_cameras()
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

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

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

    def step(self, action):
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self.update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= 100 or reward
        return ob, int(reward), done, False, {}

    def compute_reward(self, obs):
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        euler_angles = quat_2_euler(current_pose[3:])
        euler_angles = np.abs(euler_angles)
        current_pose = np.hstack([current_pose[:3], euler_angles])
        delta = np.abs(current_pose - self._TARGET_POSE)
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {_REWARD_THRESHOLD}')
            return False

    def crop_image(self, image):
        """Crop realsense images to be a square."""
        return image[:, 80:560, :]

    def get_im(self):
        images = {}
        display_images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                if key == "wrist_1":
                    cropped_rgb = self.crop_image(rgb)
                if key == "wrist_2":
                    cropped_rgb = self.crop_image(rgb)
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras()
                return self.get_im()

        self.recording_frames.append(
            np.concatenate([display_images[f"{k}_full"] for k in self.cap], axis=0)
        )
        self.img_queue.put(display_images)
        return images

    def _get_state(self):
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return state_observation

    def _get_obs(self):
        images = self.get_im()
        state_observation = self._get_state()

        return copy.deepcopy(dict(images=images, state=state_observation))

    def interpolate_move(self, goal, timeout, dt=0.1):
        steps = int(timeout / dt)
        self.update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(dt)
        self.update_currpos()

    def go_to_rest(self, jpos=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        """
        raise NotImplementedError

    def reset(self, jpos=False, gripper=None, **kwargs):
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            jpos = True

        success = self.go_to_rest(jpos=jpos)
        self.update_currpos()
        self.curr_path_length = 0
        self.recover()

        if jpos:
            self.go_to_rest(jpos=False)
            self.update_currpos()
            self.recover()

        self.update_currpos()
        o = self._get_obs()

        return o, {}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'./videos/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    self.recording_frames[0].shape[:2][::-1],
                )
                for frame in self.recording_frames:
                    video_writer.write(frame)
                video_writer.release()
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self):
        """Init both wrist cameras."""
        cap_wrist_1 = VideoCapture(
            RSCapture(
                name="wrist_1", serial_number=self.config.WRIST_CAM1_SERIAL, depth=False
            )
        )
        cap_wrist_2 = VideoCapture(
            RSCapture(
                name="wrist_2", serial_number=self.config.WRIST_CAM2_SERIAL, depth=False
            )
        )
        self.cap = {
            "wrist_1": cap_wrist_1,
            "wrist_2": cap_wrist_2,
        }

    def close_cameras(self):
        """Close both wrist cameras."""
        self.cap["wrist_1"].close()
        self.cap["wrist_2"].close()
