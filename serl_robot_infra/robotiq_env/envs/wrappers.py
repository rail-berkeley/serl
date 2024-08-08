import gymnasium as gym
import numpy as np
from agentlace import action

from robotiq_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import time
from scipy.spatial.transform import Rotation as R

from robotiq_env.utils.rotations import quat_2_euler, quat_2_mrp

ROT90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
ROT_GENERAL = np.array([np.eye(3), ROT90, ROT90 @ ROT90, ROT90.transpose()])


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, gripper_action_span=3):
        super().__init__(env)

        self.gripper_enabled = True

        self.expert = SpaceMouseExpert()
        self.last_intervene = 0
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()

        self.invert_axes = [-1, -1, 1, 1, 1, -1]
        self.deadspace = 0.15

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a = self.get_deadspace_action()

        if np.linalg.norm(
                expert_a) > 0.001 or self.left.any() or self.right.any():  # also read buttons with no movement
            self.last_intervene = time.time()

        if self.gripper_enabled:
            gripper_action = np.zeros((1,)) + int(self.left.any()) - int(self.right.any())
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.5:
            expert_a = self.adapt_spacemouse_output(expert_a)
            return expert_a

        return action

    def get_deadspace_action(self) -> np.ndarray:
        expert_a, buttons = self.expert.get_action()

        positive = np.clip((expert_a - self.deadspace) / (1. - self.deadspace), a_min=0.0, a_max=1.0)
        negative = np.clip((expert_a + self.deadspace) / (1. - self.deadspace), a_min=-1.0, a_max=0.0)
        expert_a = positive + negative

        self.left, self.right = np.roll(self.left, -1), np.roll(self.right, -1)  # shift them one to the left
        self.left[-1], self.right[-1] = tuple(buttons)

        return np.array(expert_a, dtype=np.float32)

    def adapt_spacemouse_output(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: spacemouse raw output
        Output:
        - expert_a: spacemouse output adapted to force space (action)
        """

        # position = super().get_wrapper_attr("curr_pos")  # get position from robotiq_env
        position = self.unwrapped.curr_pos
        z_angle = np.arctan2(position[1], position[0])  # get first joint angle

        z_rot = R.from_rotvec(np.array([0, 0, z_angle]))
        action[:6] *= self.invert_axes  # if some want to be inverted
        action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[3:6] = z_rot.inv().apply(action[3:6])  # z rotation invariant rotation

        return action

    def step(self, action):
        new_action = self.action(action)
        # print(f"new action: {new_action}")
        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_action"] = new_action
        info["left"] = self.left.any()
        info["right"] = self.right.any()
        return obs, rew, done, truncated, info


class Quat2EulerWrapper(gym.ObservationWrapper):  # not used anymore (stay away from euler angles!)
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_euler(tcp_pose[3:]))
        )
        return observation


class Quat2MrpWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_mrp(tcp_pose[3:]))
        )
        return observation


def rotate_state(state: np.ndarray, num_rot: int):
    assert len(state.shape) == 1 and state.shape[0] % 3 == 0
    state = state.reshape((-1, 3)).transpose()
    rotated = np.dot(ROT_GENERAL[num_rot % 4], state).transpose()
    return rotated.reshape((-1))


class ObservationRotationWrapper(gym.Wrapper):
    """
    Convert every observation into the first quadrant of the Relative Frame
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_rot_quadrant = -1

    def step(self, action: np.ndarray):
        action = self.rotate_action(action=action)
        obs, reward, done, truncated, info = self.env.step(action)
        print(self.num_rot_quadrant)
        rotated_obs = self.rotate_observation(obs)
        return rotated_obs, reward, done, truncated, info

    def rotate_observation(self, observation):
        x, y = (observation["state"]["tcp_pose"][:2])
        self.num_rot_quadrant = int(x < 0.) * 2 + int(x * y < 0.)  # save quadrant info

        for state in observation["state"].keys():
            if state == "gripper_state":
                continue
            observation["state"][state][:] = rotate_state(observation["state"][state], self.num_rot_quadrant)  # rotate

        for image_keys in observation["images"].keys():
            observation["images"][image_keys][:] = np.rot90(
                observation["images"][image_keys],
                axes=(0, 1),
                k=self.num_rot_quadrant
            )
        return observation

    def rotate_action(self, action):
        action[:6] = rotate_state(action[:6], 4 - self.num_rot_quadrant)        # rotate
        return action
