import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from jaxrl_m.envs.wrappers.chunking import ChunkingWrapper

env = gym.make("FrankaRobotiqPegInsert-Vision-v0")
env = GripperCloseEnv(env)
env = SpacemouseIntervention(env)
env = RelativeFrame(env)
env = Quat2EulerWrapper(env)
env = SERLObsWrapper(env)
env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

obs, _ = env.reset()

transitions = []
success_count = 0
success_needed = 20
pbar = tqdm(total=success_needed)

while success_count < success_needed:
    next_obs, rew, done, truncated, info = env.step(action=np.zeros((6,)))
    actions = info["intervene_action"]

    transition = copy.deepcopy(
        dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=rew,
            masks=1.0 - done,
            dones=done,
        )
    )
    transitions.append(transition)

    obs = next_obs

    if done:
        obs, _ = env.reset()
        success_count += 1
        pbar.update(1)


import pickle as pkl
import datetime

uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"peg_insert_{success_needed}_demos_{uuid}.pkl"
with open(file_name, "wb") as f:
    pkl.dump(transitions, f)
    print(f"saved {success_needed} demos to {file_name}")
