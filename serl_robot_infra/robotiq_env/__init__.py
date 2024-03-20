from gymnasium.envs.registration import register
import numpy as np

register(
    id="robotiq-grip-v1",
    entry_point="robotiq_env.envs.robotiq_grip_v1:RobotiqGripV1",
    max_episode_steps=100,
)