import time
from gym import Env, spaces
import gym
import numpy as np
from gym.spaces import Box
import copy
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from franka_env.utils.rotations import quat_2_euler

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class FWBWFrontCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the RL policy's observation space. This is used for the
    forward backward reset-free bin picking task, where there are two classifiers,
    one for classifying success + failure for the forward and one for the
    backward task. Here we also use these two classifiers to decide which
    task to transition into next at the end of the episode to maximize the
    learning efficiency.
    """

    def __init__(self, env: Env, fw_reward_classifier_func, bw_reward_classifier_func):
        # check if env.task_id exists
        assert hasattr(env, "task_id"), "fwbw env must have task_idx attribute"
        assert hasattr(env, "task_graph"), "fwbw env must have a task_graph method"

        super().__init__(env)
        self.reward_classifier_funcs = [
            fw_reward_classifier_func,
            bw_reward_classifier_func,
        ]

    def task_graph(self, obs):
        """
        predict the next task to transition into based on the current observation
        if the current task is not successful, stay in the current task
        else transition to the next task
        """
        success = self.compute_reward(obs)
        if success:
            return (self.task_id + 1) % 2
        return self.task_id

    def compute_reward(self, obs):
        reward = self.reward_classifier_funcs[self.task_id](obs).item()
        return (sigmoid(reward) >= 0.5) * 1

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        success = self.compute_reward(self.env.get_front_cam_obs())
        rew += success
        done = done or success
        return obs, rew, done, truncated, info


class FrontCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1
        return 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        success = self.compute_reward(self.env.get_front_cam_obs())
        rew += success
        done = done or success
        return obs, rew, done, truncated, info


class BinaryRewardClassifierWrapper(gym.Wrapper):
    """
    Compute reward with custom binary reward classifier fn
    """

    def __init__(self, env: Env, reward_classifier_func):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1
        return 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        success = self.compute_reward(obs)
        rew += success
        done = done or success
        return obs, rew, done, truncated, info


class ZOnlyWrapper(gym.ObservationWrapper):
    """
    Removal of X and Y coordinates
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space["state"] = spaces.Box(-np.inf, np.inf, shape=(14,))

    def observation(self, observation):
        observation["state"] = np.concatenate(
            (
                observation["state"][:4],
                np.array(observation["state"][6])[..., None],
                observation["state"][10:],
            ),
            axis=-1,
        )
        return observation


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_euler(tcp_pose[3:]))
        )
        return observation


class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.last_intervene = 0
        self.left, self.right = False, False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)

        if np.linalg.norm(expert_a) > 0.001:
            self.last_intervene = time.time()

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                self.last_intervene = time.time()
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                self.last_intervene = time.time()
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info
