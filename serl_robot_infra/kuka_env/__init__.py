from gym.envs.registration import register
import numpy as np

register(
    id="KukaEnv-Vision-v0",
    entry_point="kuka_env.envs:FrankaEnv",
    max_episode_steps=100,
)

register(
    id="KukaPegInsert-Vision-v0",
    entry_point="kuka_env.envs.peg_env:KukaPegInsert",
    max_episode_steps=100,
)
