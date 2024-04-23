from gymnasium.envs.registration import register
import numpy as np

register(
    id="robotiq-grip-v1",
    entry_point="robotiq_env.envs.basic_env:RobotiqBasicEnv",
    max_episode_steps=200,
)

register(
    id="robotiq_camera_env",
    entry_point="robotiq_env.envs.camera_env:RobotiqCameraEnv",
    max_episode_steps=300,
)