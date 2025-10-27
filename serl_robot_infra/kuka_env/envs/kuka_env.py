"""Gym Interface for Franka"""

import sys
import time
import getpass

username = getpass.getuser()
env = "serl"
print("Ensure to change anaconda3 or miniconda3 and change your environment name")
print("Conda Environment name is: ", env)
sys.path.append(
    "/home/" + username + "/miniconda3/envs/" + env + "/lib/python3.10/site-packages"
)

import numpy as np
import gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict
import logging
import os
import shutil

sys.path.append("/home/rp/SERL/src/")
from serl_robot_infra.franka_env.camera.video_capture import VideoCapture
from serl_robot_infra.franka_env.camera.rs_capture import RSCapture
from serl_robot_infra.kuka_env.utils.rotations import euler_2_quat, quat_2_euler


# from kuka_server.kuka_server.robot_interface import RobotInterfaceNode


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
        "wrist_1": "840412060409",
        "wrist_2": "932122060300",
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
        robot_interface_node=None,
    ):
        print("Initializing KukaEnv")
        self.robot_interface_node = robot_interface_node
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.ROBOT_IP
        self.config = config
        self.max_episode_length = max_episode_length
        self.max_episode_length = 200
        self.task_reward = 1.0

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )

        self.reset_joint_pos = config.RESET_JOINT_POSITION.copy()

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

        # nisara : Saving some important information here
        if not fake_env:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.exp_folder = os.path.join('experiments', f"exp_{timestamp}")
            os.makedirs(self.exp_folder, exist_ok=True)

            # Save the parameters (config.py) used for the experiment
            config_path = '/home/rp/SERL/src/serl_robot_infra/kuka_env/envs/peg_env/config.py'
            parameters_path = os.path.join(self.exp_folder, "parameters.txt")
            shutil.copy2(config_path, parameters_path)

            # Initialize a reward over the whole episode
            self.episode_max_reward = 0
            self.save_reward_stats = True
            self.reward_stats_file = os.path.join(self.exp_folder, "reward_stats.txt")
            
            # Setting logging
            self.logger = logging.getLogger()
            self.logger.handlers.pop(0)
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            file_handler = logging.FileHandler(os.path.join(self.exp_folder, 'reward_log.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False

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
        # TODO :: change to 6
        self.action_space = gym.spaces.Box(
            np.ones((6,), dtype=np.float32) * -1,
            np.ones((6,), dtype=np.float32),
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
            print("returning from fake env")
            return
        print("Initializing Cameras")
        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        self.img_queue = queue.Queue()
        self.displayer = ImageDisplayer(self.img_queue)
        self.displayer.start()
        self.use_gripper = config.USE_GRIPPER
        print("Initialized Kuka")

        # Initialization for the sdf based reward
        self.mesh_file_path = config.MESH_FILE_PATH
        self.num_sampled_points = 1000
        self.voxel_size = 0.001
        import trimesh
        import pysdf
        self.og_peg_mesh = trimesh.load(self.mesh_file_path)
        # The given mesh file is not in meters, convert it 
        self.og_peg_mesh.apply_scale(0.001)

        self.target_transform = np.eye(4)
        self.target_transform[:3, 3] = self._TARGET_POSE[:3]
        self.target_transform[:3, :3] = Rotation.from_euler("ZYX", self._TARGET_POSE[3:]).as_matrix()
        self.transformed_peg_mesh = self.og_peg_mesh.copy()
        self.transformed_peg_mesh.apply_transform(self.target_transform)
        
        vertices = np.array(self.transformed_peg_mesh.vertices, dtype=np.float32)
        faces = np.array(self.transformed_peg_mesh.faces, dtype=np.uint32)
 
        self.sdf = pysdf.SDF(vertices, faces, robust = True)
        self.local_surface_points, _ = trimesh.sample.sample_surface_even(self.transformed_peg_mesh, self.num_sampled_points)
        self.local_surface_points = np.array(self.local_surface_points)
        self.transform_target_inverse = np.linalg.inv(self.target_transform)


        


    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("ZYX")

        # Clip first two euler angles separately due to discontinuity from pi to -pi
        # sign = np.sign(euler[2])
        # euler[2] = sign * (
        #     np.clip(
        #         np.abs(euler[2]),
        #         self.rpy_bounding_box.low[2],
        #         self.rpy_bounding_box.high[2],
        #     )
        # )

        # nisara : Clipping the B and C euler angles separately because they are at inflection point
        
        # A
        euler[0] = (
            np.clip(
                euler[0],
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )
        # B - The max range is -10 to +10 degrees
        euler[1] = (
            np.clip(
                euler[1],
                self.rpy_bounding_box.low[1],
                self.rpy_bounding_box.high[1],
            )
        )
        # C - The max range is -170 to +170 degrees
        sign_C = np.sign(euler[2])
        euler[2] = sign_C * (
            np.clip(
                np.abs(euler[2]),
                self.rpy_bounding_box.high[2],
                np.abs(np.deg2rad(180)),
            )
        )

        pose[3:] = Rotation.from_euler("ZYX", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        # print("In step function")

        start_time = time.time()
        self.logger.info(f"Action before clipping: {action}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        self.logger.info(f"Action after clipping ==xyz_delta==: {xyz_delta}")

        self.nextpos = self.currpos.copy()
        # # nisara : Comment
        # print("Current position in step: ", self.nextpos)
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = Rotation.from_matrix(
            (
                np.matmul(
                    Rotation.from_quat(self.currpos[3:]).as_matrix(),
                    Rotation.from_euler(
                        "ZYX", action[3:6] * self.action_scale[1]
                    ).as_matrix(),
                )
            )
        ).as_quat()

        ##Remove gripper action NOTE: Omey
        if self.use_gripper:
            gripper_action = action[6] * self.action_scale[2]
            gripper_action_effective = self._send_gripper_command(gripper_action)

        self._send_pos_command(self.clip_safety_box(self.nextpos))
        
        # ## GO TO TARGET POSE FOR DEBUGGING
        # tp = self._TARGET_POSE.copy()
        # self.nextpos[:3] = tp[:3]
        # self.nextpos[3:] = euler_2_quat(tp[3:])
        # self._send_pos_command(self.clip_safety_box(self.nextpos))
        # ## END DEBUGGING

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        if self.use_gripper:
            reward = self.compute_reward(ob, gripper_action_effective)
        else:
            reward = self.compute_reward(ob)

        done = self.curr_path_length >= self.max_episode_length or reward == self.task_reward
        return ob, reward, done, False, {}

    def compute_reward(self, obs, gripper_action_effective=None) -> bool:
        """We are using a sparse reward function."""
        current_pose = obs["state"]["tcp_pose"]

        # convert from quat to euler first
        euler_angles_degrees = Rotation.from_quat(current_pose[3:]).as_euler("ZYX", degrees=True)
        euler_angles = quat_2_euler(current_pose[3:])
        
        # nisara: IMPORTANT: Convert the target pose to absolute values too 
        target_pose = self._TARGET_POSE.copy()

        current_pose = np.hstack([current_pose[:3], euler_angles])
        curr_pose_print = np.hstack([current_pose[:3], euler_angles_degrees])
        
        give_reward = np.zeros(6, dtype=bool)
        difference = np.zeros(6)

        difference[:3] = np.abs(current_pose[:3] - target_pose[:3])
        give_reward[:3] = difference[:3] <= self._REWARD_THRESHOLD[:3]

        # Calculation of B and C's delta is going to be different 
        # For B, the range is -5 to +5 degrees
        # For C, the range is -175 to -180 and +175 to +180 degrees

        ## A

        difference[3] = np.abs(current_pose[3] - target_pose[3])
        give_reward[3] = difference[3] <= self._REWARD_THRESHOLD[3]

        ## B
        difference[4] = np.abs(current_pose[4] - target_pose[4])
        if difference[4] <= self._REWARD_THRESHOLD[4]:
                give_reward[4] = 1

        ## C
        difference[5] = np.pi - np.abs(current_pose[5])
        if difference[5] <= self._REWARD_THRESHOLD[5]:
                give_reward[5] = 1

        if np.all(give_reward):
            print("RECEIVED COMPLETE REWARD 1.0!!!!")
            self.logger.info("RECEIVED COMPLETE REWARD 1.0!!!!")
            self.logger.info(f"Received reward 1.0 at \n\tcurrent pose: {curr_pose_print} and \n\tdiff: {difference}")
            reward = self.task_reward

        else:
            ## DO i add reward for 0.099 = z-value, reward = 0.98
            if current_pose[2] <= 0.099:
                reward = 0.98
                self.logger.info("Received surface reward 0.98")
            else:
                # SDF BASED REWARD
                tcp_transform = np.eye(4)
                tcp_transform[:3, 3] = current_pose[:3]
                tcp_transform[:3, :3] = Rotation.from_euler("ZYX", current_pose[3:]).as_matrix()
                transform_target_current = np.linalg.inv(tcp_transform) @ self.target_transform
                tcp_rot_matrix = transform_target_current[:3, :3]
                tcp_t = transform_target_current[:3, 3]
                global_surface_points = (tcp_rot_matrix @ self.local_surface_points.T).T + tcp_t

                # Calculate the signed distance field
                sdf_values = np.array([self.sdf(point) for point in global_surface_points])

                # Calculate the reward based on the sdf values
                reward = np.sqrt(np.mean(np.square(sdf_values)))
                reward = 1.0 - (reward / 0.1)  # Normalize the reward
                reward = np.clip(reward, -1.0, 1.0)  # Ensure the reward is between -1 and 1
            self.logger.info(f"Goal not reached, the difference is \n\tcurrent pose: {curr_pose_print} and \n\tdiff (in rad): {difference}  ")

        """ Commenting out the existing reward function to make place for sdf
        else:
            # if current_pose[2] < 0.10000:
            #         # Let's give more reward if z is even lower
            #         self.logger.info("Received surface reward 0.6")
            #         reward = 0.7
            # else:
            #     if current_pose[2] < 0.10300:
            #         self.logger.info("Received surface reward 0.4")
            #         reward = 0.4
            #     else:
            #         reward = 0.0

            ## Give reward based on euclidean distance
            euclidean_distance = np.linalg.norm(np.abs(current_pose) - np.abs(target_pose))
            ## The range of th edistance is 0.06 and 0.018 in demo buffer so we follow the same thing here
            reward = np.abs(1.0 - (euclidean_distance / 0.06))
            self.logger.info(f"Goal not reached, the difference is \n\tcurrent pose: {curr_pose_print} and \n\tdiff (in rad): {difference}  ")
        """
        self.logger.info(f"Reward: {reward}")
        self.logger.info("\n")
        
        # if self.config.APPLY_GRIPPER_PENALTY and gripper_action_effective:
        #     reward -= self.config.GRIPPER_PENALTY

        self.episode_max_reward = max(self.episode_max_reward, reward)
        return reward

    def crop_image(self, name, image) -> np.ndarray:
        """Crop realsense images to be a square."""
        if name == "wrist_1":
            return image[:, 80:650, :]
        elif name == "wrist_2":
            return image[:, 80:650, :]
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
                cropped_rgb = np.array(cropped_rgb)
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
            self._send_pos_command(reset_pose)
        else:
            reset_pose = self.resetpos.copy()

            # self._send_pos_command(reset_pose)

            # Reset to a specified joint position and not cartesian position to avoid singularity
            reset_joint_pose = self.reset_joint_pos.copy()
            self._send_reset_joint_command(reset_joint_pose)

        # Change to compliance mode
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def _send_reset_joint_command(self, joint_pos: np.ndarray):
        """
        Internal function to send joint position command to the robot.
        Make sure that the joint angles are in radians and not degrees.
        """
        pos = np.array(joint_pos).astype(np.float32)
        self.robot_interface_node.move_to_joint_pos(pos)
        
        

    def reset(self, joint_reset=False, **kwargs):
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

        # nisara: Have a maximum episode reward also and log that somewhere to maintain the increasing trend if any
        
        if self.save_reward_stats:
            with open(self.reward_stats_file, "a") as f:
                f.write(f"{self.episode_max_reward}\n")
            self.logger.info(f"Max reward for the episode: {self.episode_max_reward}")
        self.episode_max_reward = 0

        self.logger.info(f"\nResetting the environment\n")

        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self.go_to_rest(joint_reset=joint_reset)
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

        np.set_printoptions(precision=5, suppress=True)

        # # nisara : Comment
        # curr_pos_euler = Rotation.from_quat(self.currpos[3:]).as_euler("ZYX", degrees=True)
        # print_curr_pos = np.concatenate([self.currpos[:3] * 1000, curr_pos_euler])
        # print("Current pose in _send_pos_command: ", print_curr_pos)

        # nisara : Comment
        arr_euler = Rotation.from_quat(pos[3:]).as_euler("ZYX", degrees=True)                                       
        print_arr_pos = np.concatenate([pos[:3], arr_euler])
        self.logger.info(f"Sending position command to \n\tmove to pos: {print_arr_pos}")

        arr = np.array(pos).astype(np.float32)

        self.robot_interface_node.move_to_pose(arr, self.resetpos)
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
        self.currpos[:] = np.array(ps["pose"], dtype=np.float32)
        self.currvel[:] = np.array(ps["vel"], dtype=np.float32)

        self.currforce[:] = np.array(ps["force"], dtype=np.float32)
        self.currtorque[:] = np.array(ps["torque"], dtype=np.float32)
        self.currjacobian[:] = np.reshape(
            np.array(ps["jacobian"], dtype=np.float32), (6, 7)
        )

        self.q[:] = np.array(ps["q"], dtype=np.float32)
        self.dq[:] = np.array(ps["dq"], dtype=np.float32)

        if self.use_gripper:
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


if __name__ == "__main__":
    env = gym.make("KukaEnv")
