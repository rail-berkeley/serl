"""Gym Interface for Robotiq"""

import numpy as np
import gymnasium as gym
import copy
import time
from typing import Dict
from scipy.spatial.transform import Rotation as R


from robotiq_env.utils.rotations import rotvec_2_quat, quat_2_rotvec
from robot_controllers.robotiq_controller import RobotiqImpedanceController


##############################################################################


class DefaultEnvConfig:
    """Default configuration for RobotiqEnv. Fill in the values below."""

    TARGET_POSE: np.ndarray = np.zeros((6,))  # might change as well
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    RESET_Q = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,), dtype=np.float32)

    ROBOT_IP: str = "localhost"
    CONTROLLER_HZ: int = 0
    ERROR_DELTA: float = 0.
    FORCEMODE_DAMPING: float = 0.1
    FORCEMODE_TASK_FRAME = np.zeros(6, )
    FORCEMODE_SELECTION_VECTOR = np.ones(6, )
    FORCEMODE_LIMITS = np.zeros(6, )

    # not used for now
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }


##############################################################################


class RobotiqEnv(gym.Env):
    def __init__(
            self,
            hz: int = 10,
            config=DefaultEnvConfig,
            max_episode_length: int = 100
    ):
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.max_episode_length = max_episode_length
        self.action_scale = config.ACTION_SCALE

        self.config = config

        self.resetQ = config.RESET_Q

        self.currpos = np.zeros((7, ), dtype=np.float32)
        self.currvel = np.zeros((7,), dtype=np.float32)
        self.Q = np.zeros((6,), dtype=np.float32)  # TODO is (7,) for some reason in franka?? same in dq
        self.Qd = np.zeros((6,), dtype=np.float32)
        self.currforce = np.zeros((3,), dtype=np.float32)
        self.currtorque = np.zeros((3,), dtype=np.float32)

        self.currpressure = np.zeros((1, ), dtype=np.float32)
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = 200  # reset the robot joint every 200 cycles  # TODO needed?

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
                        "gripper_pressure": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                )
                # "images": gym.spaces.Dict(        # Images are ignored for now
                #     {
                #         "wrist_1": gym.spaces.Box(
                #             0, 255, shape=(128, 128, 3), dtype=np.uint8
                #         ),
                #         "wrist_2": gym.spaces.Box(
                #             0, 255, shape=(128, 128, 3), dtype=np.uint8
                #         ),
                #     }
                # ),
            }
        )
        self.cycle_count = 0

        self.controller = RobotiqImpedanceController(
            robot_ip=config.ROBOT_IP,
            frequency=config.CONTROLLER_HZ,
            kp=10000,
            kd=2200,
            config=config,
            verbose=True,
            plot=False
        )

        self.controller.start()  # start Thread

        while not self.controller.is_ready():       # wait for contoller
            time.sleep(0.1)

    def pose_r2q(self, pose: np.ndarray) -> np.ndarray:
        return np.concatenate([pose[:3], rotvec_2_quat(pose[3:])])

    def pose_q2r(self, pose: np.ndarray) -> np.ndarray:
        return np.concatenate([pose[:3], quat_2_rotvec(pose[3:])])

    def clip_safety_box(self, next_pos: np.ndarray) -> np.ndarray:
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
        # next_pos = self.currpos.copy()
        next_pos = self.controller.get_target_pos()
        next_pos[:3] = next_pos[:3] + action[:3] * self.action_scale[0]

        # orientation (leave for now)
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
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward
        return ob, int(reward), done, False, {}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to axis angle representation first
        current_pose = self.pose_q2r(current_pose)
        delta = np.abs(current_pose - self._TARGET_POSE)
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {_REWARD_THRESHOLD}')
            return False

    def go_to_rest(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        if joint_reset:
            pass
            # TODO joint reset

        # Perform Carteasian reset
        if self.randomreset[0]:  # randomize reset position in xy plane TODO is most likely bug in codebase, no ...[0]
            # reset_pose = self.resetpos.copy()ss
            # reset_pose[:2] += np.random.uniform(
            #     np.negative(self.random_xy_range), self.random_xy_range, (2,)
            # )
            # euler_random = self._TARGET_POSE[3:].copy()
            # euler_random[-1] += np.random.uniform(
            #     np.negative(self.random_rz_range), self.random_rz_range
            # )
            # reset_pose[3:] = euler_2_quat(euler_random)
            # self.move_to(reset_pose)
            pass
        else:
            reset_Q = self.resetQ.copy()
            self._send_gripper_command(np.ones((1,))*-1)            # disable vacuum gripper
            self._send_reset_command(reset_Q)

            while True:
                time.sleep(0.1)
                if self.controller.is_reset():
                    break       # wait for reset

            self._update_currpos()

    def reset(self, joint_reset=False, **kwargs):
        self.cycle_count += 1
        if self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self.go_to_rest(joint_reset=joint_reset)
        # self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()

        return obs, {}

    def _recover(self):
        """Internal function to recover the robot from error state."""
        # TODO make recover function
        pass

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

        self.currpos[:] = state['pos']
        self.currvel[:] = state['vel']
        self.currforce[:] = state['force']
        self.currtorque[:] = state['torque']
        self.Q[:] = state['Q']
        self.Qd[:] = state['Qd']
        self.currpressure[:] = state['pressure']

    def _get_obs(self) -> dict:
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pressure": self.currpressure,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(state=state_observation))

    def close(self):
        self.controller.stop()
        super().close()
