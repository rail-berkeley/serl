"""Gym Interface for Franka"""
import sys
sys.path.append("/home/aero/anaconda3/envs/serl/lib/python3.10/site-packages")

import numpy as np
import gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict
sys.path.append("/home/aero/omey_ws/serl-rros/src/")
from serl_robot_infra.franka_env.camera.video_capture import VideoCapture
from serl_robot_infra.franka_env.camera.rs_capture import RSCapture
from serl_robot_infra.franka_env.utils.rotations import euler_2_quat, quat_2_euler


# from kuka_server.kuka_server.robot_interface import RobotInterfaceNode
from kuka_server.robot_interface import RobotInterfaceNode


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

            cv2.imshow("RealSense Cameras", frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for KukaEnv. Fill in the values below."""

    ROBOT_IP: str = "192.168.10.122"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    BINARY_GRIPPER_THREASHOLD: float = 0.5
    APPLY_GRIPPER_PENALTY: bool = True
    GRIPPER_PENALTY: float = 0.1
    USE_GRIPPER: bool = False


##############################################################################


class KukaEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        max_episode_length=100,
    ):
        
        self.robot_interface_node = RobotInterfaceNode()
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.ROBOT_IP
        self.config = config
        self.max_episode_length = max_episode_length

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )

        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))
        self.currjacobian = np.zeros((6, 7))

        self.curr_gripper_pos = 0
        self.gripper_binary_state = 0  # 0 for open, 1 for closed
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = 200  # reset the robot joint every 200 cycles

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # boundary box
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

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {   
                        "wrist_1": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                        "wrist_2": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        self.img_queue = queue.Queue()
        self.displayer = ImageDisplayer(self.img_queue)
        self.displayer.start()
        self.use_gripper = config.USE_GRIPPER
        print("Initialized Kuka")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

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
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
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

        ##Remove gripper action NOTE: Omey
        if(self.use_gripper): 
            gripper_action = action[6] * self.action_scale[2]
            gripper_action_effective = self._send_gripper_command(gripper_action)
        
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        if(self.use_gripper):
            reward = self.compute_reward(ob, gripper_action_effective)
        else:
            reward = self.compute_reward(ob)

        done = self.curr_path_length >= self.max_episode_length or reward == 1
        return ob, reward, done, False, {}

    def compute_reward(self, obs, gripper_action_effective = None) -> bool:
        """We are using a sparse reward function."""
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        euler_angles = quat_2_euler(current_pose[3:])
        euler_angles = np.abs(euler_angles)
        current_pose = np.hstack([current_pose[:3], euler_angles])
        delta = np.abs(current_pose - self._TARGET_POSE)
        if np.all(delta < self._REWARD_THRESHOLD):
            reward = 1
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {_REWARD_THRESHOLD}')
            reward = 0

        if self.config.APPLY_GRIPPER_PENALTY and gripper_action_effective:
            reward -= self.config.GRIPPER_PENALTY

        return reward

    def crop_image(self, name, image) -> np.ndarray:
        """Crop realsense images to be a square."""
        if name == "wrist_1":
            return image[:, 80:560, :]
        elif name == "wrist_2":
            return image[:, 80:560, :]
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.crop_image(key, rgb)
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
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        self.recording_frames.append(
            np.concatenate([display_images[f"{k}_full"] for k in self.cap], axis=0)
        )
        self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self._update_currpos()

    def go_to_rest(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset
        # requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        reset_pose = self.resetpos.copy()
        self.interpolate_move(reset_pose, timeout=1.5)

        # Change to compliance mode
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self.go_to_rest(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()

        return obs, {}

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

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, cam_serial in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, serial_number=cam_serial, depth=False)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        print("Implement a function to recover from error state")
        return

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        self._recover()
        arr = np.array(pos).astype(np.float32)
        self.robot_interface_node.move_to_pose(arr)
        print("Done moving the robot")
        
    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (
                pos <= -self.config.BINARY_GRIPPER_THREASHOLD
                and self.gripper_binary_state == 0
            ):  # close gripper
                requests.post(self.url + "close_gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 1
                return True
            elif (
                pos >= self.config.BINARY_GRIPPER_THREASHOLD
                and self.gripper_binary_state == 1
            ):  # open gripper
                requests.post(self.url + "open_gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 0
                return True
            else:  # do nothing to the gripper
                return False
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = self.robot_interface_node.get_current_state()
        self.currpos[:] = np.array(ps["pose"])
        self.currvel[:] = np.array(ps["vel"])

        self.currforce[:] = np.array(ps["force"])
        self.currtorque[:] = np.array(ps["torque"])
        self.currjacobian[:] = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q[:] = np.array(ps["q"])
        self.dq[:] = np.array(ps["dq"])
        
        if(self.use_gripper):
            self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))


if __name__ == '__main__':

    env = gym.make("KukaEnv")
    
    
    