from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.robotiq_grip_v1.config import RobotiqCornerConfig


class RobotiqGripV1(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RobotiqCornerConfig)

    def compute_reward(self, obs) -> float:
        if int(self.gripper_state[1]) == 1:
            if 10 < self.gripper_state[0] < 30:
                if self.curr_force[2] < -1.:
                    return 10.
                return 5.
            return 2.5
        if self.curr_force[2] > 5.:
            return (5. - self.curr_force[2]) * 0.2          # return 0 if force=5, -1 if force=10 (max))
        else:
            return 0.0
