import numpy as np

from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.camera_env.config import RobotiqCameraConfig


class RobotiqCameraEnv(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RobotiqCameraConfig)
        self.plot_costs_yes = False
        if self.plot_costs_yes:
            self.reward_hist = dict(action_cost=[], suction_cost=[], non_central_cost=[], suction_reward=[],
                                    downward_force_cost=[])

    """def compute_reward(self, obs, action) -> float:
        # huge action gives negative reward (like in mountain car)
        action_cost = 0.1 * np.sum(np.power(action, 2))
        step_cost = 0.01

        gripper_state = obs["state"]['gripper_state']
        suction_cost = 0.2 * float(is_close(gripper_state[0], 0.99))
        suction_reward = 0.1 * float(0.1 < gripper_state[0] < 0.85)
        downward_force_cost = 0.4 * max(obs["state"]["tcp_force"][2] - 5, 0.)

        torque = obs["state"]['tcp_torque']
        non_central_cost = 0.5 * max(np.linalg.norm(torque[:2]) - 0.07, 0.)

        if self.plot_costs_yes:
            self.reward_hist['action_cost'].append(-action_cost)
            self.reward_hist['suction_cost'].append(-suction_cost)
            self.reward_hist['non_central_cost'].append(-non_central_cost)
            self.reward_hist['suction_reward'].append(suction_reward)
            self.reward_hist['downward_force_cost'].append(-downward_force_cost)

        total_cost = action_cost + step_cost + suction_cost + non_central_cost + downward_force_cost
        if self.reached_goal_state(obs):
            box_is_central = np.sum(np.power(torque[:2], 2)) - 0.1 < 0.
            # return (100. if box_is_central else 50.) - action_cost - step_cost - suction_cost
            return 100. - total_cost
        else:
            return 0.0 - total_cost + suction_reward"""

    def compute_reward(self, obs, action) -> float:
        action_cost = 0.1 * np.sum(np.power(action, 2))
        step_cost = 0.01

        downward_force_cost = 0.1 * max(obs["state"]["tcp_force"][2] - 10., 0.)
        suction_reward = 0.5 * float(obs["state"]["gripper_state"][1] > 0.9)
        suction_cost = 0.3 * float(np.isclose(obs["state"]["gripper_state"][0], 0.99))

        orientation_cost = 1. - sum(obs["state"]["tcp_pose"][3:] * self.curr_reset_pose[3:]) ** 2
        orientation_cost *= 25.

        max_pose_diff = 0.05  # set to 5cm
        pos_diff = obs["state"]["tcp_pose"][:2] - self.curr_reset_pose[:2]
        position_cost = 10. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0))

        if self.reached_goal_state(obs):
            return 100. - action_cost - orientation_cost - position_cost
        else:
            return 0. + suction_reward - action_cost - downward_force_cost - orientation_cost - position_cost

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        return 0.1 < state['gripper_state'][0] < 0.85 and state['tcp_pose'][2] > 0.10  # new min height with box

    def close(self):
        if self.plot_costs_yes:
            self.plot_costs()
        super().close()

    def plot_costs(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(6, 1, figsize=(12, 10), sharey=True, sharex=True)
        y = np.arange(len(self.reward_hist["action_cost"]))

        ax1, ax2, ax3, ax4, ax5, ax6 = ax
        ax1.plot(y, self.reward_hist["action_cost"], label="action_cost")
        ax1.legend()
        ax2.plot(y, self.reward_hist["suction_cost"], label="suction_cost")
        ax2.legend()
        ax3.plot(y, self.reward_hist["non_central_cost"], label="non_central_cost")
        ax3.legend()
        ax4.plot(y, self.reward_hist["suction_reward"], label="suction_reward")
        ax4.legend()
        ax5.plot(y, self.reward_hist["downward_force_cost"], label="downward_force_cost")
        ax5.legend()

        total = [sum(i) for i in zip(*self.reward_hist.values())]
        ax6.plot(y, total, label="total")
        ax6.legend()

        plt.show()
