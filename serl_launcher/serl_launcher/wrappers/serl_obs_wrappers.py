import gym
from gym.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.env.observation_space["state"]),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        obs = {
            "state": flatten(self.env.observation_space["state"], obs["state"]),
            **(obs["images"]),
        }
        return obs

class ManipulatorEnvObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper is to be used with the ManipulatorEnv.
    from:
        https://github.com/rail-berkeley/manipulator_gym
        
    The will convert the observation space to be compatible with the SERL
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {   
                # convert (N) to (1, N)
                "state": gym.spaces.Box(
                    low=self.env.observation_space["state"].low[None],
                    high=self.env.observation_space["state"].high[None],
                    shape=(1, *self.env.observation_space["state"].shape),
                    dtype=self.env.observation_space["state"].dtype,
                ),
                # convert "image*" from (H, W, C) to (1, H, W, C)
                **{
                    key: gym.spaces.Box(
                        low=self.env.observation_space[key].low[None],
                        high=self.env.observation_space[key].high[None],
                        shape=(1, *self.env.observation_space[key].shape),
                        dtype=self.env.observation_space[key].dtype,
                    )
                    for key in self.observation_space.keys() if key.startswith("image")
                },
            }
        )

    def observation(self, obs):
        obs = {
            "state": obs["state"][None],
            **{
                key: obs[key][None] for key in obs if key.startswith("image")
            },
        }
        return obs
