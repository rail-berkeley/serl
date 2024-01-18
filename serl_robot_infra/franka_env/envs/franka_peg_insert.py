import numpy as np
import gymnasium as gym

from franka_env.envs.franka_env import FrankaEnv


class FrankaPegInsert(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def go_to_rest(self, jpos=False):
        # TODO: Implement this
        return NotImplementedError
