from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.robotiq_grip_v1.config import RobotiqCornerConfig


class RobotiqGripV1(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RobotiqCornerConfig)

    def compute_reward(self, obs) -> float:
        if 0.1 < self.gripper_state[0] < 0.3 and self.curr_force[2] < -3.:
            return 1.
        if self.curr_force[2] > 5.:
            return (5. - self.curr_force[2]) * 0.1          # return 0 if force=5, -0.5 if force=10 (max))
        else:
            return 0.0
