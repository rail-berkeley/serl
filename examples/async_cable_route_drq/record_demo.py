import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    BinaryRewardClassifierWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
import jax

if __name__ == "__main__":
    env = gym.make("FrankaCableRoute-Vision-v0", save_video=False)
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [k for k in env.observation_space.keys() if "state" not in k]

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    classifier_func = load_classifier_func(
        key=key,
        sample=env.observation_space.sample(),
        image_keys=image_keys,
        checkpoint_path="/home/undergrad/code/serl_dev/examples/async_cable_route_drq/classifier_ckpt/",
    )
    env = BinaryRewardClassifierWrapper(env, classifier_func)

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    success_needed = 70
    total_count = 0

    pbar = tqdm(total=success_needed)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./bc_demos/cable_route_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    while success_count < success_needed:
        actions = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action=actions)
        if "intervene_action" in info:
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
            success_count += rew
            total_count += 1
            print(
                f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
            )
            pbar.update(rew)
            obs, _ = env.reset()

    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(
            f"saved {success_needed} demos and {len(transitions)} transitions to {file_path}"
        )

    env.close()
    pbar.close()
