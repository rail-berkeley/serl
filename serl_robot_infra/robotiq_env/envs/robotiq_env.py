"""Gym Interface for Robotiq"""

import numpy as np
import gymnasium as gym
import copy
import time
from typing import Dict
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from robotiq_env.utils.rotations import rotvec_2_quat, quat_2_rotvec

##############################################################################


class DefaultEnvConfig:
    """Default configuration for RobotiqEnv. Fill in the values below."""

    TARGET_POSE: np.ndarray = np.zeros((6,))           # might change as well
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))

    ROBOT_IP: str = "localhost"
    FORCEMODE_DAMPING: float = 0.1
    FORCEMODE_TASK_FRAME = np.zeros(6, )
    FORCEMODE_SELECTION_VECTOR = np.ones(6, )
    FORCEMODE_FORCE_TYPE: int = 2
    FORCEMODE_LIMITS = np.zeros(6, )
    FORCEMODE_SCALING = np.ones(6, )

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
            save_video: bool = False,
            config=DefaultEnvConfig,
            max_episode_length: int = 100
    ):
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.max_episode_length = max_episode_length

        self.robot_ip = config.ROBOT_IP
        self.FM_DAMPING = config.FORCEMODE_DAMPING
        self.FM_TASK_FRAME = config.FORCEMODE_TASK_FRAME
        self.FM_SELECTION_VECTOR = config.FORCEMODE_SELECTION_VECTOR
        self.FM_FORCE_TYPE = config.FORCEMODE_FORCE_TYPE
        self.FM_LIMITS = config.FORCEMODE_LIMITS
        self.FM_SCALING = config.FORCEMODE_SCALING

        self.robotiq_control = None
        self.robotiq_receive = None
        self.robotiq_gripper = None

        # convert last 3 elements from axis angle to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], rotvec_2_quat(config.RESET_POSE[3:])]
        ).astype(np.float32)

        self.currpos = self.resetpos.copy().astype(np.float32)
        self.currvel = np.zeros((6,), dtype=np.float32)
        self.q = np.zeros((6,), dtype=np.float32)         # TODO is (7,) for some reason in franka?? same in dq
        self.dq = np.zeros((6,), dtype=np.float32)
        self.currforce = np.zeros((3,), dtype=np.float32)
        self.currtorque = np.zeros((3,), dtype=np.float32)
        self.currjacobian = np.zeros((6, 7), dtype=np.float32)

        self.curr_gripper_pos = 0
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
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
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

        self.start_robotiq_interfaces()

    def start_robotiq_interfaces(self):
        self.robotiq_control = RTDEControl(self.robot_ip)
        self.robotiq_receive = RTDEReceive(self.robot_ip)
        # self.robotiq_gripper = VacuumGripper(self.robot_ip)
        # TODO gripper
        print("UR-RTDE interfaces ready")


    def pose_r2q(self, pose: np.ndarray) -> np.ndarray:
        return np.concatenate([pose[:3], rotvec_2_quat(pose[3:])])

    def pose_q2r(self, pose: np.ndarray) -> np.ndarray:
        return np.concatenate([pose[:3], quat_2_rotvec(pose[3:])])

    def clip_safety_box(self, action: np.ndarray) -> np.ndarray:
        """Clip the action to not move outside the safety box."""

        # check for position limits (prevent, but do not stop)
        adverse_move = 0.1
        for i, (low, high) in enumerate(zip(self.xyz_bounding_box.low, self.xyz_bounding_box.high)):
            if low and self.currpos[i] < low and action[i] < adverse_move:
                action[i] = adverse_move
                print(f"lower {i} set new")
            elif high and self.currpos[i] > high and action[i] > -adverse_move:
                action[i] = -adverse_move
                print("upper set new")

        # TODO apply quaternion limits
        # TODO test
        return action

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        t_start_robotiq = self.robotiq_control.initPeriod()
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        safe_action = self.clip_safety_box(action)
        self._send_force_command(safe_action[:6])
        self._send_gripper_command(safe_action[6])

        self.curr_path_length += 1
        self.robotiq_control.waitPeriod(t_start_robotiq)

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

    # TODO adapt get_im(), crop_image(), init_cameras(), close_cameras(),

    def move_to(self, goal: np.ndarray):
        """Move the robot to the goal position with moveL (linear in tool-space)"""
        goal_ur = self.pose_q2r(goal)
        self.robotiq_control.moveL(list(goal_ur), speed=0.05, acceleration=0.3)   # command will block until finished
        self._update_currpos()

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
            reset_pose = self.resetpos.copy()
            self.move_to(reset_pose)

    def reset(self, joint_reset=False, **kwargs):
        if self.save_video:
            # TODO adapt save_video_recording()
            pass

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

    def _recover(self):
        """Internal function to recover the robot from error state."""
        # TODO make recover function
        pass

    def _send_force_command(self, action: np.ndarray):
        """Internal function to send force command to the robot."""
        force = action * self.FM_SCALING
        self.robotiq_control.forceModeSetDamping(self.FM_DAMPING)
        self.robotiq_control.forceMode(
            self.FM_TASK_FRAME,
            self.FM_SELECTION_VECTOR,
            force,
            self.FM_FORCE_TYPE,
            self.FM_LIMITS
        )

    def _send_gripper_command(self, pos: float):
        """Internal function to send gripper command to the robot."""
        if (pos >= -1) and (pos <= -0.9):  # close gripper
            pass  # TODO gripper command
        elif (pos >= 0.9) and (pos <= 1):  # open gripper
            pass
        else:  # do nothing to the gripper
            return

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        pose = self.robotiq_receive.getActualTCPPose()
        vel = self.robotiq_receive.getActualTCPSpeed()
        pose_quat = self.pose_r2q(pose)

        force = self.robotiq_receive.getActualTCPForce()
        q = self.robotiq_receive.getActualQ()
        qd = self.robotiq_receive.getActualQd()

        self.currpos[:] = np.array(pose_quat)
        self.currvel[:] = np.array(vel)

        self.currforce[:] = np.array(force[:3])
        self.currtorque[:] = np.array(force[3:])
        # self.currjacobian[:] = np.reshape(np.array(ps["jacobian"]), (6, 7)) # TODO jacobian?

        self.q[:] = np.array(q)
        self.dq[:] = np.array(qd)

        # TODO get gripper pos
        self.curr_gripper_pos = np.zeros(1, dtype=np.float32)

    def _get_obs(self) -> dict:
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(state=state_observation))

    def close(self):
        self.robotiq_control.forceModeStop()
        print("force mode stopped")
        super().close()
