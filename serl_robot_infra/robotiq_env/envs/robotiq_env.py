"""Gym Interface for Robotiq"""

import time
import threading
import copy
import numpy as np
import gymnasium as gym
import cv2
import queue
from typing import Dict
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

from robotiq_env.camera.video_capture import VideoCapture
from robotiq_env.camera.rs_capture import RSCapture

from robotiq_env.utils.real_time_plotter import DataClient
from robotiq_env.utils.rotations import rotvec_2_quat, quat_2_rotvec, pose2quat, pose2rotvec
from robot_controllers.robotiq_controller import RobotiqImpedanceController


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
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )
            cv2.namedWindow("RealSense Cameras", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RealSense Cameras", 300, 700)
            cv2.imshow("RealSense Cameras", frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for RobotiqEnv. Fill in the values below."""

    RESET_Q = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,), dtype=np.float32)

    ROBOT_IP: str = "localhost"
    CONTROLLER_HZ: int = 0
    GRIPPER_TIMEOUT: int = 0  # in milliseconds
    ERROR_DELTA: float = 0.
    FORCEMODE_DAMPING: float = 0.
    FORCEMODE_TASK_FRAME = np.zeros(6, )
    FORCEMODE_SELECTION_VECTOR = np.ones(6, )
    FORCEMODE_LIMITS = np.zeros(6, )

    REALSENSE_CAMERAS: Dict = {
        "shoulder": "",
        "wrist": "",
    }


##############################################################################


class RobotiqEnv(gym.Env):
    def __init__(
            self,
            hz: int = 10,
            fake_env=False,
            config=DefaultEnvConfig,
            max_episode_length: int = 100,
            save_video: bool = False,
            realtime_plot: bool = False,
            camera_mode: str = "rgb",  # one of (rgb, depth, both, None)
    ):
        self.max_episode_length = max_episode_length
        self.action_scale = config.ACTION_SCALE

        self.config = config

        self.resetQ = config.RESET_Q
        self.curr_reset_pose = np.zeros((7,), dtype=np.float32)

        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)
        self.curr_Q = np.zeros((6,), dtype=np.float32)  # TODO is (7,) for some reason in franka?? same in dq
        self.curr_Qd = np.zeros((6,), dtype=np.float32)
        self.curr_force = np.zeros((3,), dtype=np.float32)
        self.curr_torque = np.zeros((3,), dtype=np.float32)

        self.gripper_state = np.zeros((2,), dtype=np.float32)
        self.last_sent = time.time()
        self.random_reset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz

        camera_mode = None if camera_mode.lower() == "none" else camera_mode
        if camera_mode is not None and save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []
        self.camera_mode = camera_mode

        self.realtime_plot = realtime_plot

        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        image_space_definition = {}
        if camera_mode in ["rgb", "both"]:
            # image_space_definition["shoulder"] = gym.spaces.Box(      # TODO only temporary with one image
            #             0, 255, shape=(128, 128, 3), dtype=np.uint8
            # )
            image_space_definition["wrist"] = gym.spaces.Box(
                0, 255, shape=(128, 128, 3), dtype=np.uint8
            )

        if camera_mode in ["depth", "both"]:
            # image_space_definition["shoulder_depth"] = gym.spaces.Box(
            #     0, 255, shape=(128, 128, 1), dtype=np.uint8
            # )
            image_space_definition["wrist_depth"] = gym.spaces.Box(
                0, 255, shape=(128, 128, 1), dtype=np.uint8
            )

        if camera_mode is not None and camera_mode not in ["rgb", "both", "depth"]:
            raise NotImplementedError(f"camera mode {camera_mode} not implemented")

        state_space = gym.spaces.Dict(
            {
                "tcp_pose": gym.spaces.Box(
                    -np.inf, np.inf, shape=(7,)
                ),  # xyz + quat
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_state": gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )

        obs_space_definition = {"state": state_space}
        if self.camera_mode in ["rgb", "both", "depth"]:
            obs_space_definition["images"] = gym.spaces.Dict(
                image_space_definition
            )

        self.observation_space = gym.spaces.Dict(obs_space_definition)

        self.cycle_count = 0
        self.controller = None
        self.cap = None

        if fake_env:
            print("[RobotiqEnv] is fake!")
            return

        self.controller = RobotiqImpedanceController(
            robot_ip=config.ROBOT_IP,
            frequency=config.CONTROLLER_HZ,
            kp=10000,
            kd=2200,
            config=config,
            verbose=False,
            plot=False,
            # old_obs=camera_mode is None       # do not use anymore
        )
        self.controller.start()  # start Thread

        if self.camera_mode is not None:
            self.init_cameras(config.REALSENSE_CAMERAS)
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()
            print("[CAM] Cameras are ready!")

        if self.realtime_plot:
            try:
                self.plotting_client = DataClient()
            except ConnectionRefusedError:
                print("Plotting Client could not be opened, continuing without plotting")
                self.realtime_plot = False

        while not self.controller.is_ready():  # wait for controller
            time.sleep(0.1)
        print("[RIC] Controller has started and is ready!")

    def clip_safety_box(self,
                        next_pos: np.ndarray) -> np.ndarray:  # TODO make better, no euler -> quat -> euler -> quat
        """Clip the pose to be within the safety box."""
        next_pos[:3] = np.clip(
            next_pos[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = R.from_quat(next_pos[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        next_pos[3:] = R.from_euler("xyz", euler).as_quat()

        return next_pos

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # position
        next_pos = self.curr_pos.copy()
        next_pos[:3] = next_pos[:3] + action[:3] * self.action_scale[0]

        # orientation
        next_pos[3:] = (
                R.from_quat(next_pos[3:]) * R.from_euler("xyz", action[3:6] * self.action_scale[1])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        safe_pos = self.clip_safety_box(next_pos)
        self._send_pos_command(safe_pos)
        self._send_gripper_command(gripper_action)

        self.curr_path_length += 1

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        obs = self._get_obs()

        reward = self.compute_reward(obs, action)
        truncated = self._is_truncated()

        reward = reward if not truncated else reward - 100.  # truncation penalty

        done = self.curr_path_length >= self.max_episode_length or self.reached_goal_state(obs) or truncated
        return obs, reward, done, truncated, {}

    def compute_reward(self, obs, action) -> float:
        return 0.  # overwrite for each task

    def reached_goal_state(self, obs) -> bool:
        return False  # overwrite for each task

    def go_to_rest(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform Carteasian reset
        reset_Q = np.zeros((6))
        if self.resetQ.shape == (1, 6):
            reset_Q[:] = self.resetQ.copy()
        elif self.resetQ.shape[1] == 6 and self.resetQ.shape[0] > 1:
            choice = np.random.randint(self.resetQ.shape[0])
            reset_Q[:] = self.resetQ[choice, :].copy()  # make random guess
        else:
            raise ValueError(f"invalid resetQ dimension: {self.resetQ.shape}")

        self._send_reset_command(reset_Q)

        while not self.controller.is_reset():
            time.sleep(0.1)  # wait for the reset operation

        self._update_currpos()
        reset_pose = self.controller.get_target_pos()

        if self.random_reset:  # randomize reset position in xy plane
            # reset_pose = self.resetpos.copy()
            reset_shift = np.random.uniform(np.negative(self.random_xy_range), self.random_xy_range, (2,))
            reset_pose[:2] += reset_shift
            self.curr_reset_pose[:] = reset_pose

            # euler_random = reset_pose[3:]
            # euler_random[-1] += np.random.uniform(
            #     np.negative(self.random_rz_range), self.random_rz_range
            # )
            self.controller.set_target_pos(reset_pose)  # random movement after resetting
            time.sleep(0.1)
            while self.controller.is_moving():
                time.sleep(0.1)
            # print(reset_shift, reset_pose)
            return reset_shift
        else:
            self.curr_reset_pose[:] = reset_pose
            return np.zeros((2,))

    def reset(self, joint_reset=False, **kwargs):
        self.cycle_count += 1
        if self.save_video:
            self.save_video_recording()

        shift = self.go_to_rest(joint_reset=joint_reset)
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        return obs, {"reset_shift": shift}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'/home/nico/real-world-rl/spacemouse_tests/videos/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
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

    def init_cameras(self, name_serial_dict=None):
        """Init both cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, cam_serial in name_serial_dict.items():
            print(f"cam serial: {cam_serial}")
            rgb = self.camera_mode in ["rgb", "both"]
            depth = self.camera_mode in ["depth", "both"]
            cap = VideoCapture(
                RSCapture(name=cam_name, serial_number=cam_serial, rgb=rgb, depth=depth)
            )
            self.cap[cam_name] = cap

    def crop_image(self, name, image) -> np.ndarray:
        """Crop realsense images to be a square."""
        if name == "wrist":
            return image[:, 80:560, :]
        elif name == "shoulder":
            return image[:, 80:560, :]
        else:
            raise ValueError(f"Camera {name} not recognized in cropping")

    def get_image(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        for key, cap in self.cap.items():
            try:
                image = cap.read()
                if self.camera_mode in ["rgb", "both"]:
                    rgb = image[..., :3].astype(np.uint8)
                    cropped_rgb = self.crop_image(key, rgb)
                    resized = cv2.resize(
                        cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                    )
                    images[key] = resized[..., ::-1]
                    display_images[key] = resized
                    display_images[key + "_full"] = cropped_rgb

                if self.camera_mode in ["depth", "both"]:
                    depth_key = key + "_depth"
                    depth = image[..., -1:]
                    cropped_depth = self.crop_image(key, depth)

                    resized = cv2.resize(
                        cropped_depth, self.observation_space["images"][depth_key].shape[:2]
                    )[..., None]

                    images[depth_key] = resized
                    display_images[depth_key] = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
                    display_images[depth_key + "_full"] = cv2.applyColorMap(cropped_depth, cv2.COLORMAP_JET)

            except queue.Empty:
                input(f"{key} camera frozen. Check connect, then press enter to relaunch...")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_image()


        self.recording_frames.append(
            np.concatenate([image for key, image in display_images.items() if "full" in key], axis=0)
        )
        self.img_queue.put(display_images)
        return images

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _send_pos_command(self, target_pos: np.ndarray):
        """Internal function to send force command to the robot."""
        self.controller.set_target_pos(target_pos=target_pos)

    def _send_gripper_command(self, gripper_pos: np.ndarray):
        self.controller.set_gripper_pos(gripper_pos)

    def _send_reset_command(self, reset_Q: np.ndarray):
        self.controller.set_reset_Q(reset_Q)

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        state = self.controller.get_state()

        self.curr_pos[:] = state['pos']
        self.curr_vel[:] = state['vel']
        self.curr_force[:] = state['force']
        self.curr_torque[:] = state['torque']
        self.curr_Q[:] = state['Q']
        self.curr_Qd[:] = state['Qd']
        self.gripper_state[:] = state['gripper']

    def _is_truncated(self):
        return self.controller.is_truncated()

    def _get_obs(self) -> dict:
        state_observation = {
            "tcp_pose": self.curr_pos,
            "tcp_vel": self.curr_vel,
            "gripper_state": self.gripper_state,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
        }
        if self.realtime_plot:
            self.plotting_client.send(np.concatenate([self.curr_force, self.curr_torque]))

        if self.camera_mode is not None:
            images = self.get_image()
            # images = np.ones_like(images) * 255
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))

    def close(self):
        if self.controller:
            self.controller.stop()
        super().close()
