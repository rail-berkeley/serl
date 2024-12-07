import numpy as np
from typing import Tuple

from ur_env.envs.ur5_env import UR5Env
from ur_env.envs.basic_env.config import UR5BasicConfig


class BoxPickingBasicEnv(UR5Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=UR5BasicConfig)

    def compute_reward(self, obs, action) -> float:
        # huge action gives negative reward (like in mountain car)
        action_cost = 0.1 * np.sum(np.power(action, 2))
        step_cost = 0.01

        gripper_state = obs["state"]["gripper_state"]
        suction_cost = 0.1 * float(np.isclose(gripper_state[0], 0.99, atol=1e-4))

        if self.reached_goal_state(obs):
            return 10.0 - action_cost - step_cost - suction_cost
        else:
            return 0.0 - action_cost - step_cost - suction_cost

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        return (
            0.1 < state["gripper_state"][0] < 0.85 and state["tcp_pose"][2] > 0.15
        )  # new min height with box
