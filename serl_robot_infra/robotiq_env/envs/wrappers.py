import gymnasium as gym
import numpy as np
from robotiq_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import time
from scipy.spatial.transform import Rotation as R

from robotiq_env.utils.rotations import quat_2_euler


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.gripper_enabled = True

        self.expert = SpaceMouseExpert()
        self.last_intervene = 0
        self.left, self.right = False, False

        self.invert_axes = [-1, -1, 1, 1, -1, -1]
        self.deadspace = 0.15

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a = self.get_deadspace_action()

        if np.linalg.norm(expert_a) > 0.001 or self.left or self.right:     # also read buttons with no movement
            self.last_intervene = time.time()

        if self.gripper_enabled:
            gripper_action = np.zeros((1,)) + int(self.left) - int(self.right)
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

        self.left, self.right = tuple(buttons)

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
        action[3:6] = z_rot.inv().apply(action[3:6])  # z rotation invariant rotation

        return action

    def step(self, action):
        new_action = self.action(action)
        # print(f"new action: {new_action}")
        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info


class Quat2EulerWrapper(gym.ObservationWrapper):
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
