from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.robotiq_grip_v1.config import RobotiqCornerConfig


class RobotiqGripV1(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RobotiqCornerConfig)
