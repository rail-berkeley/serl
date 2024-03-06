from robotiq_env.envs.robotiq_env import RobotiqEnv
from robotiq_env.envs.test_env.config import TestConfig


class RobotiqTest(RobotiqEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=TestConfig)

    def go_to_rest(self, joint_reset=False):
        # how to overwrite without blocking the old function
        super().go_to_rest(joint_reset)
