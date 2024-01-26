import gymnasium as gym
from gymnasium.core import Env
from copy import deepcopy


class FrontCameraWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        front_obs_space = {
            k: space for k, space in self.observation_space.items() if "wrist" not in k
        }

        self.front_observation_space = gym.spaces.Dict(front_obs_space)
        # self.observation_space = gym.spaces.Dict(new_obs_space)
        self.front_obs = None

    def observation(self, observation):
        # cache a copy of observation with only the front camera image
        new_obs = deepcopy(observation)
        new_obs.pop("wrist_1")
        self.front_obs = new_obs

        return observation

    def get_front_cam_obs(self):
        return self.front_obs
