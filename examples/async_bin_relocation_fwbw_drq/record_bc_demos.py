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
    SpacemouseIntervention,
    Quat2EulerWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper

from serl_launcher.wrappers.chunking import ChunkingWrapper

if __name__ == "__main__":
    from pynput import keyboard

    env = gym.make(
        "FrankaBinRelocation-Vision-v0", save_video=False, max_episode_length=200
    )
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)

    env.set_task_id(0)
    obs, _ = env.reset()

    demos_count = 0
    demos_needed = 20

    is_success = False
    is_fail = False
    trajectories = []

    def on_press(key):
        global is_success
        if key == keyboard.Key.space and not is_success:
            is_success = True

    def on_esc(key):
        global is_fail
        if key == keyboard.Key.esc and not is_fail:
            is_fail = True

    # Collect all event until released
    listener_1 = keyboard.Listener(on_press=on_press)
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc)
    listener_2.start()

    pbar = tqdm(total=demos_needed, desc="bc_demos")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"bc_bin_relocate_{demos_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    transitions = []
    while demos_count < demos_needed:

        actions = np.zeros((7,))
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

        if is_fail or done:
            obs, _ = env.reset()
            is_fail = False
            transitions = []

        elif is_success:
            done = True
            obs, _ = env.reset()
            is_success = False
            demos_count += 1
            pbar.update(1)
            trajectories += transitions
            transitions = []

    with open(file_path, "wb") as f:
        pkl.dump(trajectories, f)
        print(f"saved {len(trajectories)} transitions to {file_path}")

    listener_1.stop()
    listener_2.stop()
    env.close()
    pbar.close()
