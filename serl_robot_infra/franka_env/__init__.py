from gymnasium.envs.registration import register
import numpy as np

register(
    id="FrankaEnv-Vision-v0",
    entry_point="franka_env.envs:FrankaEnv",
    max_episode_steps=100,
)

register(
    id="FrankaPegInsert-Vision-v0",
    entry_point="franka_env.envs:FrankaPegInsert",
    max_episode_steps=100,
    kwargs={
        "randomReset": True,
        "random_xy_range": 0.05,
        "random_rz_range": np.pi / 9,
    },
)

register(
    id="FrankaPCBInsert-Vision-v0",
    entry_point="franka_env.envs:FrankaPCBInsert",
    max_episode_steps=100,
    kwargs={
        "randomReset": True,
        "random_xy_range": 0.05,
        "random_rz_range": np.pi / 9,
    },
)

register(
    id="FrankaCableRoute-Vision-v0",
    entry_point="franka_env.envs:FrankaCableRoute",
    max_episode_steps=100,
    kwargs={
        "randomReset": True,
        "random_xy_range": 0.1,
        "random_rz_range": np.pi / 6,
    },
)
