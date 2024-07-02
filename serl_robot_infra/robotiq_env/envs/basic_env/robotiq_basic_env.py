import numpy as np
from typing import Tuple

from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.basic_env.config import RobotiqCornerConfig


# used for float value comparisons (pressure of vacuum-gripper)
def is_close(value, target):
    return abs(value - target) < 1e-4


class RobotiqBasicEnv(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RobotiqCornerConfig)

    def compute_reward(self, obs, action) -> float:
        # huge action gives negative reward (like in mountain car)
        action_cost = 0.1 * np.sum(np.power(action, 2))
        step_cost = 0.01

        gripper_state = obs["state"]['gripper_state']
        suck_cost = 0.1 * float(is_close(gripper_state[0], 0.99))

        pose = obs["state"]["tcp_pose"]
        # box_xy = np.array([0.009, -0.5437])     # TODO replace with camera / pointcloud info of box
        # xy_cost = 5 * np.sum(np.power(pose[:2] - box_xy, 2))        # TODO can be ignored

        # print(f"action_cost: {action_cost}, xy_cost: {xy_cost}")
        if self.reached_goal_state(obs):
            return 10. - action_cost - step_cost - suck_cost
        else:
            return 0.0 - action_cost - step_cost - suck_cost

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        # return 0.1 < state['gripper_state'][0] < 0.8 and state['tcp_force'][2] < -3.
        return 0.1 < state['gripper_state'][0] < 0.85 and state['tcp_pose'][2] > 0.15  # new min height with box
