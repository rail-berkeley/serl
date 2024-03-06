from gymnasium.envs.registration import register
import numpy as np

register(
    id="robotiq_test",
    entry_point="robotiq_env.envs.test_env:RobotiqTest",
    max_episode_steps=100,
)