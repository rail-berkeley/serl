import os
import datetime
import numpy as np
import copy
import pickle as pkl
from tqdm import tqdm
import gymnasium as gym
from pprint import pprint

from robotiq_env.envs.wrappers import SpacemouseIntervention, Quat2EulerWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper     # TODO has no images


if __name__ == "__main__":
    env = gym.make("robotiq_test")
    env = SpacemouseIntervention(env)
    env = Quat2EulerWrapper(env)
    # env = SERLObsWrapper(env)

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    success_needed = 20
    total_count = 0
    pbar = tqdm(total=success_needed)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"robotiq_test_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        while success_count < success_needed:
            next_obs, rew, done, truncated, info = env.step(action=np.zeros((7,)))
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
            # pprint(transition)

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
            print(f"saved {success_needed} demos to {file_path}")

    finally:
        env.close()
