from gymnasium.envs.registration import register
import numpy as np

register(
    id="box_picking_basic_env",
    entry_point="ur_env.envs.basic_env:BoxPickingBasicEnv",
    max_episode_steps=200,
)

register(
    id="box_picking_camera_env",
    entry_point="ur_env.envs.camera_env:BoxPickingCameraEnv",
    max_episode_steps=100,
)

