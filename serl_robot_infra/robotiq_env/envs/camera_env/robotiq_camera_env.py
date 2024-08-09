import numpy as np
from typing import Tuple

from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.camera_env.config import RobotiqCameraConfigFinal, RobotiqCameraConfigFinalTests


class RobotiqCameraEnv(RobotiqEnv):
    def __init__(self, load_config=True, **kwargs):
        if load_config:
            super().__init__(**kwargs, config=RobotiqCameraConfigFinal)
        else:
            super().__init__(**kwargs)

    def compute_reward(self, obs, action) -> float:
        action_cost = 0.1 * np.sum(np.power(action, 2))
        step_cost = 0.1

        suction_reward = 0.3 * float(obs["state"]["gripper_state"][1] > 0.5)
        suction_cost = 3. * float(obs["state"]["gripper_state"][1] < -0.5)

        orientation_cost = 1. - sum(obs["state"]["tcp_pose"][3:] * self.curr_reset_pose[3:]) ** 2
        orientation_cost *= 25.

        max_pose_diff = 0.03  # set to 3cm
        pos_diff = obs["state"]["tcp_pose"][:2] - self.curr_reset_pose[:2]
        position_cost = 10. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )

        cost_info = dict(
            action_cost=-action_cost,
            step_cost=-step_cost,
            suction_reward=suction_reward,
            suction_cost=-suction_cost,
            orientation_cost=-orientation_cost,
            position_cost=-position_cost,
            total_cost=-action_cost - step_cost + suction_reward - suction_cost - orientation_cost - position_cost
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])

        if self.reached_goal_state(obs):
            return 100. - action_cost - orientation_cost - position_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        return 0.1 < state['gripper_state'][0] < 1. and state['tcp_pose'][2] > self.curr_reset_pose[2] + 0.01  # +1cm

    def close(self):
        super().close()


class RobotiqCameraEnvTest(RobotiqCameraEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, load_config=False, config=RobotiqCameraConfigFinalTests)
