import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from serl_launcher.wrappers.chunking import ChunkingWrapper

if __name__ == "__main__":
    env = gym.make("FrankaPCBInsert-Vision-v0", save_video=False)
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    success_needed = 40
    total_count = 0
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
            success_count += rew
            total_count += 1
            print(
                f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
            )
            pbar.update(rew)
            obs, _ = env.reset()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./pcb_insert_{success_needed}_demos_{uuid}.pkl"
    try:
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_needed} demos to {file_name}")
    except Exception as e:
        print(f"failed to save demos to {file_name}")
        print(e)
        f_temp = f"/tmp/recovered_serl_demos_{uuid}.pkl"
        print(f"attempting to save to {f_temp} instead...")
        with open(f_temp, "wb") as f:
            pkl.dump(transitions, f)
            print(f"successfully saved to {f_temp}. PLEASE MOVE TO A SAFE LOCATION!")

    env.close()
    pbar.close()
