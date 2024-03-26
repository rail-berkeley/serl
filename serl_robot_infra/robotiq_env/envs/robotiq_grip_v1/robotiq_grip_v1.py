from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.robotiq_grip_v1.config import RobotiqCornerConfig


class RobotiqGripV1(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RobotiqCornerConfig)

    def compute_reward(self, obs) -> float:
        if int(self.gripper_state[1]) == 1 and 10 < self.gripper_state[0] < 30 and self.curr_force[2] < -1.:
            return 1.
        else:
            return 0.0
