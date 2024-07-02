import numpy as np
from collections import deque
import gymnasium as gym


class ObservationStatisticsWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """
    This wrapper will keep track of the observation statistics.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``obsStat``.
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        self.buffer = {}

        # make buffer
        for name, space in self.env.observation_space["state"].items():
            self.buffer[name] = np.zeros(shape=(self.max_episode_length, space.shape[0]))

        # may not be used
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."

        for name, obs in observations["state"].items():
            self.buffer[name][self.curr_path_length - 1, :] = obs

        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            calc_buffs = {}
            calc_buffs.update({
                name + "_mean": np.mean(obs[:self.curr_path_length], axis=0) for name, obs in self.buffer.items()
            })
            calc_buffs.update({
                name + "_std": np.std(obs[:self.curr_path_length], axis=0) for name, obs in self.buffer.items()
            })
            buff = {}
            for name, value in calc_buffs.items():
                for i in range(value.shape[0]):
                    buff[name + f"_{['x', 'y', 'z', 'rx', 'ry', 'rz'][i]}"] = value[i]
            infos["obsStat"] = buff
            # print(buff)

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)

        # reset buffer to zero
        for name, value in self.buffer.items():
            value[...] = 0

        return obs, info
