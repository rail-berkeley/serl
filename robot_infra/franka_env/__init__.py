from gymnasium.envs.registration import register
import numpy as np

register(
    id="FrankaRobotiq-Vision-v0",
    entry_point="franka_env.envs:FrankaRobotiq",
    max_episode_steps=100,
)

register(
    id="FrankaRobotiqPegInsert-Vision-v0",
    entry_point="franka_env.envs:FrankaRobotiqPegInsert",
    max_episode_steps=100,
    kwargs={
        "randomReset": True,
        "random_xy_range": 0.05,
        "random_rz_range": np.pi / 9,
    },
)

register(
    id="FrankaRobotiqPCBInsert-Vision-v0",
    entry_point="franka_env.envs:FrankaRobotiqPCBInsert",
    max_episode_steps=100,
    kwargs={
        "randomReset": True,
        "random_xy_range": 0.05,
        "random_rz_range": np.pi / 9,
    },
)

register(
    id="FrankaRobotiqCableRoute-Vision-v0",
    entry_point="franka_env.envs:FrankaRobotiqCableRoute",
    max_episode_steps=100,
    kwargs={
        "randomReset": True,
        "random_xy_range": 0.1,
        "random_rz_range": np.pi / 6,
    },
)
