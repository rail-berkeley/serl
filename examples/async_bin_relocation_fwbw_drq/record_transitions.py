"""
Script to record goals and failed transitions for the bin relocation task.

Usage:
    python record_transitions.py --transitions_needed 400

add `--record_failed_only` to only record failed transitions
"""

import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from pynput import keyboard

import franka_env
import argparse

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    SpacemouseIntervention,
    Quat2EulerWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper

from serl_launcher.wrappers.chunking import ChunkingWrapper

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--transitions_needed",
        type=int,
        default=400,
        help="number of transitions to collect",
    )
    arg_parser.add_argument(
        "--record_failed_only",
        action="store_true",
        help="only collect failed transitions",
    )
    args = arg_parser.parse_args()

    env = gym.make("FrankaBinRelocation-Vision-v0", save_video=False)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)

    env.set_task_id(0)
    obs, _ = env.reset()
    obs = env.get_front_cam_obs()

    is_success = False
    fw_success_transitions = []
    fw_failed_transitions = []
    bw_success_transitions = []
    bw_failed_transitions = []

    def on_press(key):
        global is_success
        if key == keyboard.Key.space:
            is_success = True

    def on_release(key):
        global is_success
        if key == keyboard.Key.space:
            is_success = False

    # Collect all event until released
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    fw_pbar = tqdm(total=args.transitions_needed, desc="fw")
    bw_pbar = tqdm(total=args.transitions_needed, desc="bw")

    if args.record_failed_only:

        def check_all_done():
            return (
                len(fw_failed_transitions) >= args.transitions_needed
                or len(bw_failed_transitions) >= args.transitions_needed
            )

    else:

        def check_all_done():
            return (
                len(fw_success_transitions) >= args.transitions_needed
                or len(bw_success_transitions) >= args.transitions_needed
            )

    # Loop until we have enough transitions
    while not check_all_done():
        actions = np.zeros((7,))
        next_obs, rew, done, truncated, info = env.step(action=actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]
        next_obs = env.get_front_cam_obs()

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

        # Append data to buffer list
        if is_success and not args.record_failed_only:
            if (
                env.task_id == 0
                and len(fw_success_transitions) < args.transitions_needed
            ):
                fw_success_transitions.append(transition)
                fw_pbar.update(1)
            elif (
                env.task_id == 1
                and len(bw_success_transitions) < args.transitions_needed
            ):
                bw_success_transitions.append(transition)
                bw_pbar.update(1)
        else:
            if env.task_id == 0:
                fw_failed_transitions.append(transition)
                if args.record_failed_only:
                    fw_pbar.update(1)
            else:
                bw_failed_transitions.append(transition)
                if args.record_failed_only:
                    bw_pbar.update(1)

        obs = next_obs

        if done:
            env.set_task_id(env.task_graph())
            print(f"current task id: {env.task_id}")
            obs, _ = env.reset()
            obs = env.get_front_cam_obs()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # save success transitions
    if not args.record_failed_only:
        file_name = (
            f"fw_bin_relocate_{args.transitions_needed}_front_cam_goal_{uuid}.pkl"
        )
        with open(file_name, "wb") as f:
            pkl.dump(fw_success_transitions, f)
            print(f"saved {len(fw_success_transitions)} transitions to {file_name}")
        file_name = (
            f"bw_bin_relocate_{args.transitions_needed}_front_cam_goal_{uuid}.pkl"
        )
        with open(file_name, "wb") as f:
            pkl.dump(bw_success_transitions, f)
            print(f"saved {len(bw_success_transitions)} transitions to {file_name}")

    # save failed transitions
    file_name = (
        f"fw_bin_relocate_{len(fw_failed_transitions)}_front_cam_failed_{uuid}.pkl"
    )
    with open(file_name, "wb") as f:
        pkl.dump(fw_failed_transitions, f)
        print(f"saved {len(fw_failed_transitions)} transitions to {file_name}")
    file_name = (
        f"bw_bin_relocate_{len(bw_failed_transitions)}_front_cam_failed_{uuid}.pkl"
    )
    with open(file_name, "wb") as f:
        pkl.dump(bw_failed_transitions, f)
        print(f"saved {len(bw_failed_transitions)} transitions to {file_name}")

    listener.stop()
