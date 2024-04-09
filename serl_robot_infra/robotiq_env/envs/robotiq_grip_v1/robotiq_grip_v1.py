import numpy as np

from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.robotiq_grip_v1.config import RobotiqCornerConfig


class RobotiqGripV1(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RobotiqCornerConfig)

    def compute_reward(self, obs, action) -> float:
        # huge action gives negative reward (like in mountain car)
        neg_rew = -0.1 * np.sum(np.power(action, 2))
        # neg_rew = 0.

        if self.reached_goal_state(obs):
            return 10. + neg_rew
        else:
            return 0.0 + neg_rew

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        return 0.1 < state['gripper_state'][0] < 0.4 and state['tcp_force'][2] < -3.
