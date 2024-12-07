"""Gym Interface for UR5"""

import time
import threading
import copy
import numpy as np
import gym
import cv2
import queue
import warnings
import requests
import json
from typing import Dict, Tuple
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from ur_env.camera.video_capture import VideoCapture
from ur_env.camera.rs_capture import RSCapture

from ur_env.camera.utils import PointCloudFusion, CalibrationTread

from robot_controllers.ur5_controller import UrImpedanceController


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


class PointCloudDisplayer:
    def __init__(self):
        self.window = o3d.visualization.Visualizer()
        self.window.create_window(height=400, width=400, visible=True)

        self.pc = o3d.geometry.PointCloud()
        self.window.get_render_option().load_from_json(
            "/home/nico/.config/JetBrains/PyCharm2024.1/scratches/render_options.json"
        )

        self.param = o3d.io.read_pinhole_camera_parameters(
            "/home/nico/.config/JetBrains/PyCharm2024.1/scratches/camera_parameters.json"
        )
        self.ctr = self.window.get_view_control()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.01, origin=[0, 0, 0]
        )

    def display(self, points):
        self.pc.clear()
        # MASSIVE! speed up if float64 is used, see: https://github.com/isl-org/Open3D/issues/1045
        self.pc.points = o3d.utility.Vector3dVector(points.astype(np.float64) / 1000.0)
        self.window.clear_geometries()
        self.window.add_geometry(self.pc)
        # self.window.add_geometry(self.coord_frame)
        self.ctr.convert_from_pinhole_camera_parameters(self.param, True)

        self.window.poll_events()
        # self.window.update_renderer()

    def close(self):
        self.window.destroy_window()


##############################################################################


class DefaultEnvConfig:
    """Default configuration for UR5Env. Fill in the values below."""

    RESET_Q = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ABS_POSE_RANGE_LIMITS = np.zeros((2,))
    ACTION_SCALE = np.zeros((3,), dtype=np.float32)

    ROBOT_IP: str = "localhost"
    CONTROLLER_HZ: int = 0
    GRIPPER_TIMEOUT: int = 0  # in milliseconds
    ERROR_DELTA: float = 0.0
    FORCEMODE_DAMPING: float = 0.0
    FORCEMODE_TASK_FRAME = np.zeros(
        6,
    )
    FORCEMODE_SELECTION_VECTOR = np.ones(
        6,
    )
    FORCEMODE_LIMITS = np.zeros(
        6,
    )

    REALSENSE_CAMERAS: Dict = {
        "shoulder": "",
        "wrist": "",
    }


##############################################################################


class UR5Env(gym.Env):
    def __init__(
        self,
        hz: int = 10,
        fake_env=False,
        config=DefaultEnvConfig,
        max_episode_length: int = 100,
        save_video: bool = False,
        camera_mode: str = "rgb",  # one of (rgb, grey, depth, both(rgb depth), pointcloud, none)
    ):
        self.max_episode_length = max_episode_length
        self.curr_path_length = 0
        self.action_scale = config.ACTION_SCALE

        self.config = config

        self.resetQ = config.RESET_Q
        self.curr_reset_pose = np.zeros((7,), dtype=np.float32)

        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)
        self.curr_Q = np.zeros((6,), dtype=np.float32)
        self.curr_Qd = np.zeros((6,), dtype=np.float32)
        self.curr_force = np.zeros((3,), dtype=np.float32)
        self.curr_torque = np.zeros((3,), dtype=np.float32)

        self.gripper_state = np.zeros((2,), dtype=np.float32)
        self.random_reset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rot_range = config.RANDOM_ROT_RANGE
        self.hz = hz
        np.random.seed(0)  # fix seed for fixed (random) initial rotations

        camera_mode = None if camera_mode.lower() == "none" else camera_mode
        if camera_mode is not None and save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []
        self.camera_mode = camera_mode

        self.cost_infos = {}

        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.xy_range = gym.spaces.Box(
            config.ABS_POSE_RANGE_LIMITS[0],
            config.ABS_POSE_RANGE_LIMITS[1],
            dtype=np.float64,
        )
        self.mrp_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self.last_action = np.zeros(self.action_space.shape)

        image_space_definition = {}
        if camera_mode in ["rgb", "grey", "both"]:
            channel = 1 if camera_mode == "grey" else 3
            if "wrist" in config.REALSENSE_CAMERAS.keys():
                image_space_definition["wrist"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, channel), dtype=np.uint8
                )
            if "wrist_2" in config.REALSENSE_CAMERAS.keys():
                image_space_definition["wrist_2"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, channel), dtype=np.uint8
                )

        if camera_mode in ["depth", "both"]:
            if "wrist" in config.REALSENSE_CAMERAS.keys():
                image_space_definition["wrist_depth"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, 1), dtype=np.uint8
                )
            if "wrist_2" in config.REALSENSE_CAMERAS.keys():
                image_space_definition["wrist_2_depth"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, 1), dtype=np.uint8
                )

        if camera_mode in ["pointcloud"]:
            image_space_definition["wrist_pointcloud"] = gym.spaces.Box(
                0, 255, shape=(50, 50, 40), dtype=np.uint8
            )
        if camera_mode is not None and camera_mode not in [
            "rgb",
            "both",
            "depth",
            "pointcloud",
            "grey",
        ]:
            raise NotImplementedError(f"camera mode {camera_mode} not implemented")

        state_space = gym.spaces.Dict(
            {
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),  # xyz + quat
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_state": gym.spaces.Box(-1.0, 1.0, shape=(2,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "action": gym.spaces.Box(-1.0, 1.0, shape=self.action_space.shape),
            }
        )

        obs_space_definition = {"state": state_space}
        if self.camera_mode in ["rgb", "both", "depth", "pointcloud", "grey"]:
            obs_space_definition["images"] = gym.spaces.Dict(image_space_definition)

        self.observation_space = gym.spaces.Dict(obs_space_definition)

        self.cycle_count = 0
        self.controller = None
        self.cap = None

        if fake_env:
            print("[UR5Env] is fake!")
            return

        self.controller = UrImpedanceController(
            robot_ip=config.ROBOT_IP,
            frequency=config.CONTROLLER_HZ,
            kp=15000,
            kd=3300,
            config=config,
            verbose=False,
            plot=False,
        )
        self.controller.start()  # start Thread

        if self.camera_mode is not None:
            self.init_cameras(config.REALSENSE_CAMERAS)
            self.img_queue = queue.Queue()
            if self.camera_mode in ["pointcloud"]:
                self.displayer = (
                    PointCloudDisplayer()
                )  # o3d displayer cannot be threaded :/
            else:
                self.displayer = ImageDisplayer(self.img_queue)
                self.displayer.start()
            print("[CAM] Cameras are ready!")

        while not self.controller.is_ready():  # wait for controller
            time.sleep(0.1)
        print("[RIC] Controller has started and is ready!")

        if self.camera_mode in ["pointcloud"]:
            voxel_grid_shape = np.array(
                self.observation_space["images"]["wrist_pointcloud"].shape
            )
            # voxel_grid_shape[-1] *= 8     # do not use compacting for now
            # voxel_grid_shape *= 2
            print(f"pointcloud resolution set to: {voxel_grid_shape}")
            self.pointcloud_fusion = PointCloudFusion(
                angle=30.5,
                x_distance=0.185,
                y_distance=-0.01,
                voxel_grid_shape=voxel_grid_shape,
            )

            # load pre calibrated, else calibrate
            if not self.pointcloud_fusion.load_finetuned():
                # TODO make calibration more robust!
                self.calibration_thread = CalibrationTread(
                    pc_fusion=self.pointcloud_fusion, verbose=True
                )
                self.calibration_thread.start()

                self.calibrate_pointcloud_fusion(visualize=True)

    def clip_safety_box(self, next_pos: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        next_pos[:3] = np.clip(
            next_pos[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        orientation_diff = (
            R.from_quat(next_pos[3:]) * R.from_quat(self.curr_reset_pose[3:]).inv()
        ).as_mrp()
        orientation_diff = np.clip(
            orientation_diff, self.mrp_bounding_box.low, self.mrp_bounding_box.high
        )
        next_pos[3:] = (
            R.from_mrp(orientation_diff) * R.from_quat(self.curr_reset_pose[3:])
        ).as_quat()

        return next_pos

    def get_cost_infos(self, done):
        if not done:
            return {}
        cost_infos = self.cost_infos.copy()
        self.cost_infos = {}
        return cost_infos

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # position
        next_pos = self.curr_pos.copy()
        next_pos[:3] = next_pos[:3] + action[:3] * self.action_scale[0]

        next_pos[3:] = (
            R.from_mrp(action[3:6] * self.action_scale[1] / 4.0)
            * R.from_quat(next_pos[3:])
        ).as_quat()  # c * r  --> applies c after r

        gripper_action = action[6] * self.action_scale[2]

        safe_pos = self.clip_safety_box(next_pos)
        self._send_pos_command(safe_pos)
        self._send_gripper_command(gripper_action)

        self.curr_path_length += 1

        obs = self._get_obs(action)

        reward = self.compute_reward(obs, action)
        truncated = self._is_truncated()
        reward = reward if not truncated else reward - 10.0  # truncation penalty
        done = (
            self.curr_path_length >= self.max_episode_length
            or self.reached_goal_state(obs)
            or truncated
        )

        dt = time.time() - start_time
        to_sleep = max(0, (1.0 / self.hz) - dt)
        if to_sleep == 0:
            warnings.warn(
                f"environment could not be within {self.hz} Hz, took {dt:.4f}s!"
            )
        time.sleep(to_sleep)

        return obs, reward, done, truncated, self.get_cost_infos(done)

    def compute_reward(self, obs, action) -> float:
        return 0.0  # overwrite for each task

    def reached_goal_state(self, obs) -> bool:
        return False  # overwrite for each task

    def go_to_rest(self):
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
            reset_Q[:] = self.resetQ[0, :].copy()  # make random guess
            self.resetQ[:] = np.roll(self.resetQ, -1, axis=0)  # roll one (not random)
        else:
            raise ValueError(f"invalid resetQ dimension: {self.resetQ.shape}")

        self._send_reset_command(reset_Q)

        while not self.controller.is_reset():
            time.sleep(0.1)  # wait for the reset operation

        self._update_currpos()
        reset_pose = self.controller.get_target_pos()

        if self.random_reset:  # randomize reset position in xy plane
            reset_shift = np.random.uniform(
                np.negative(self.random_xy_range), self.random_xy_range, (2,)
            )
            reset_pose[:2] += reset_shift

            if self.random_rot_range[0] > 0.0:
                random_rot = np.random.triangular(
                    np.negative(self.random_rot_range),
                    0.0,
                    self.random_rot_range,
                    size=(3,),
                )
            else:
                random_rot = np.zeros((3,))
            reset_pose[3:][:] = (
                R.from_quat(reset_pose[3:]) * R.from_mrp(random_rot)
            ).as_quat()

            self.curr_reset_pose[:] = reset_pose

            self.controller.set_target_pos(
                reset_pose
            )  # random movement after resetting
            time.sleep(0.1)
            while self.controller.is_moving():
                time.sleep(0.1)
        else:
            self.curr_reset_pose[:] = reset_pose

    def go_to_detected_box(self):
        """ "
        function for the demo
        """
        if self.gripper_state[0] > 0.01:
            reset_Q = self.curr_Q.copy()
            reset_Q[:4] = [0.0, -np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0]
            self._send_reset_command(reset_Q)
            while not self.controller.is_reset():
                time.sleep(0.1)  # wait for the reset operation

            reset_Q[:4] = [np.pi / 2, -np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0]
            self._send_reset_command(reset_Q)
            while not self.controller.is_reset():
                time.sleep(0.1)  # wait for the reset operation

            # release the box
            self._send_gripper_command(np.array(-1))
            time.sleep(0.1)

        # go back on top
        reset_Q = [0.0, -np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0, 0.0]
        self._send_reset_command(reset_Q)
        while not self.controller.is_reset():
            time.sleep(0.1)  # wait for the reset operation
        time.sleep(0.5)

        def get_request(i=10):
            if i == 0:
                raise Exception("err")
            try:
                r = requests.get("http://192.168.1.204:5000/api/data")
                r.raise_for_status()
                boxes = r.json()
                if len(boxes) == 0:
                    time.sleep(0.1)
                    return get_request(i)
                else:
                    return boxes

            except (json.decoder.JSONDecodeError, requests.exceptions.HTTPError):
                return get_request(i=i - 1)

        boxes = get_request()

        highest = list(boxes.keys())[
            np.argmax([b["world2box"]["pos"][1] for b in boxes.values()])
        ]
        box = boxes[highest]["world2box"]
        print(
            f"pose: {[round(b, 2) for b in box['pos']]} {[round(b, 2) for b in box['rot']]}"
        )

        t = R.from_euler("xyz", [-np.pi / 2.0, np.pi, 0.0])
        pos = t.apply(
            np.array(box["pos"])
            + np.array([0.0, 0.1 + boxes[highest]["size"][1] / 2.0, 0.0])
        )
        rot = (
            R.from_euler("xyz", t.apply(box["rot"]))
            * R.from_euler("xyz", [np.pi, 0.0, 0.0])
        ).as_rotvec()

        init_pose = np.concatenate((pos, rot))

        print(f"moving to {init_pose}")
        self._send_taskspace_command(init_pose)
        while not self.controller.is_reset():
            time.sleep(0.1)  # wait for the reset operation

        self._update_currpos()
        self.curr_reset_pose[:] = self.curr_pos

    def reset(self, **kwargs):
        self.cycle_count += 1
        if self.save_video:
            self.save_video_recording()

        self.go_to_rest()
        self.curr_path_length = 0

        obs = self._get_obs(np.zeros_like(self.last_action))
        return obs

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'./videos/{datetime.now().strftime("%m-%d_%H-%M")}.mp4',
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
            rgb = self.camera_mode in ["rgb", "both", "grey"]
            depth = self.camera_mode in ["depth", "both"]
            pointcloud = self.camera_mode in ["pointcloud"]
            cap = VideoCapture(
                RSCapture(
                    name=cam_name,
                    serial_number=cam_serial,
                    rgb=rgb,
                    depth=depth,
                    pointcloud=pointcloud,
                )
            )
            self.cap[cam_name] = cap

    def crop_image(self, name, image) -> np.ndarray:
        """Crop realsense images to be a square."""
        if name == "wrist":
            return image[:, 124:604, :]
        elif name == "wrist_2":
            return image[:, 124:604, :]
        else:
            raise ValueError(f"Camera {name} not recognized in cropping")

    def get_image(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        if self.camera_mode == "pointcloud":
            self.pointcloud_fusion.clear()
        for key, cap in self.cap.items():
            try:
                image = cap.read()
                if self.camera_mode in ["rgb", "both", "grey"]:
                    rgb = image[..., :3].astype(np.uint8)
                    cropped_rgb = self.crop_image(key, rgb)
                    resized = cv2.resize(
                        cropped_rgb,
                        self.observation_space["images"][key].shape[:2][::-1],
                    )
                    # convert to grayscale here
                    if self.camera_mode == "grey":
                        grey = np.array([0.2989, 0.5870, 0.1140])
                        resized = np.dot(resized, grey)[..., None]
                        resized = resized.astype(np.uint8)
                        display_images[key] = np.repeat(resized, 3, axis=-1)
                    else:
                        display_images[key] = resized

                    images[key] = resized[..., ::-1]
                    display_images[key + "_full"] = cropped_rgb

                if self.camera_mode in ["depth", "both"]:
                    depth_key = key + "_depth"
                    depth = image[..., -1:]
                    cropped_depth = self.crop_image(key, depth)

                    resized = cv2.resize(
                        cropped_depth,
                        np.array(self.observation_space["images"][depth_key].shape[:2])
                        * 3,
                        # (128 * 3, 128 * 3) image
                    )[..., None]

                    resized = resized.reshape((128, 3, 128, 3, 1)).max(
                        (1, 3)
                    )  # max pool with 3x3

                    images[depth_key] = resized
                    display_images[depth_key] = cv2.applyColorMap(
                        resized, cv2.COLORMAP_JET
                    )
                    display_images[depth_key + "_full"] = cv2.applyColorMap(
                        cropped_depth, cv2.COLORMAP_JET
                    )

                if self.camera_mode in ["pointcloud"]:
                    pointcloud = image
                    self.pointcloud_fusion.append(pointcloud)

            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_image()

        if self.camera_mode in ["pointcloud"]:
            (
                voxel_grid,
                voxel_indices,
            ) = self.pointcloud_fusion.get_pointcloud_representation(voxelize=True)

            # downsample on 2x2x2 grid with sum of points (8 as max)
            # vs = self.observation_space["images"]["wrist_pointcloud"].shape
            # voxel_grid = np.sum(np.reshape(voxel_grid, (vs[0], 2, vs[1], 2, vs[2], 2)), axis=(1, 3, 5))
            images["wrist_pointcloud"] = voxel_grid.astype(np.uint8)

            self.displayer.display(voxel_indices)

        # self.recording_frames.append(
        #     np.concatenate([image for key, image in display_images.items() if "full" in key], axis=0)
        # )
        self.img_queue.put(display_images)

        return images

    def calibrate_pointcloud_fusion(self, save=True, visualize=False, num_samples=20):
        self.reset()
        import open3d as o3d

        assert self.camera_mode in ["pointcloud"]
        print("calibrating pointcloud fusion...")
        # calibrate pc fusion here

        obs, reward, done, truncated, _ = self.step(np.zeros((7,)))
        pc = o3d.geometry.PointCloud()
        fused = self.pointcloud_fusion.fuse_pointclouds(voxelize=False, cropped=False)
        pc.points = o3d.utility.Vector3dVector(fused)
        o3d.visualization.draw_geometries([pc])

        # get samples
        for i in range(num_samples):
            # action = [np.sin(i * np.pi / 10.), np.cos(i * np.pi / 10.), 0., -.3 * np.sin(i * np.pi / 10.),
            #           -.3 * np.cos(i * np.pi / 10.), 0., 0.]
            action = [
                -1.0 if i % 4 < 2 else 1,
                -1.0 if i % 4 in [1, 2] else 1,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]

            print(action)
            obs, reward, done, truncated, _ = self.step(np.array(action))
            time.sleep(0.1)

            self.calibration_thread.append_backlog(
                *self.pointcloud_fusion.get_original_pcds()
            )

        # calibrate()
        self.controller.stop()
        time.sleep(1)
        self.calibration_thread.calibrate()

        if save:
            self.pointcloud_fusion.save_finetuned()

        if visualize:
            pc = o3d.geometry.PointCloud()
            for i in range(num_samples):
                pc.clear()
                pcs = self.calibration_thread.pc_backlog[i]
                self.pointcloud_fusion.clear()
                self.pointcloud_fusion.append(pcs[0])
                self.pointcloud_fusion.append(pcs[1])
                fused = self.pointcloud_fusion.fuse_pointclouds(
                    voxelize=False, cropped=False
                )
                pc.points = o3d.utility.Vector3dVector(fused)
                o3d.visualization.draw_geometries([pc])

        self.calibration_thread.join()
        exit(f"restart the program to use the calibrated values")

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

    def _send_taskspace_command(self, target_pos):
        self.controller.set_reset_pose(target_pos)

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        state = self.controller.get_state()

        self.curr_pos[:] = state["pos"]
        self.curr_vel[:] = state["vel"]
        self.curr_force[:] = state["force"]
        self.curr_torque[:] = state["torque"]
        self.curr_Q[:] = state["Q"]
        self.curr_Qd[:] = state["Qd"]
        self.gripper_state[:] = state["gripper"]

    def _is_truncated(self):
        return self.controller.is_truncated()

    def _get_obs(self, action) -> dict:
        # get image before state observation, so they match better in time

        images = None
        if self.camera_mode is not None:
            images = self.get_image()

        self._update_currpos()
        state_observation = {
            "tcp_pose": self.curr_pos,
            "tcp_vel": self.curr_vel,
            "gripper_state": self.gripper_state,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
            "action": action,
        }

        if images is not None:
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))

    def close(self):
        if self.controller:
            self.controller.stop()
        super().close()
