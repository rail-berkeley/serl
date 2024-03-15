from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.test_env.config import TestConfig


class RobotiqTest(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=TestConfig)
