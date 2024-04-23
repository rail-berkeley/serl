import numpy as np

from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.camera_env.config import RobotiqCameraConfig


# used for float value comparisons (pressure of vacuum-gripper)
def is_close(value, target):
    return abs(value - target) < 1e-4


class RobotiqCameraEnv(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RobotiqCameraConfig)

    def compute_reward(self, obs, action) -> float:
        # huge action gives negative reward (like in mountain car)
        action_cost = 0.1 * np.sum(np.power(action, 2))
        step_cost = 0.01

        gripper_state = obs["state"]['gripper_state']
        suction_cost = 0.1 * float(is_close(gripper_state[0], 0.99))

        torque = obs["state"]['tcp_torque']
        non_central_cost = 0.5 * max(np.linalg.norm(torque[:2]) - 0.07, 0.)

        # print(f"action_cost: {action_cost}, suction_cost: {suction_cost}   non_central_cost: {non_central_cost}")
        if self.reached_goal_state(obs):
            box_is_central = np.sum(np.power(torque[:2], 2)) - 0.01 < 0.
            return (10. if box_is_central else 5.) - action_cost - step_cost - suction_cost
        else:
            return 0.0 - action_cost - step_cost - suction_cost - non_central_cost

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        return 0.1 < state['gripper_state'][0] < 0.85 and state['tcp_pose'][2] > 0.15  # new min height with box
