import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime

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

    success_needed = 2000

    is_success = False
    fw_failed_transitions = []
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

    fw_pbar = tqdm(total=success_needed, desc="fw")
    bw_pbar = tqdm(total=success_needed, desc="bw")

    while (
        len(fw_failed_transitions) < success_needed
        or len(bw_failed_transitions) < success_needed
    ):
        next_obs, rew, done, truncated, info = env.step(action=np.zeros((7,)))
        next_obs = env.get_front_cam_obs()
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

        if not is_success:
            if env.task_id == 0:
                fw_failed_transitions.append(transition)
                fw_pbar.update(1)
            else:
                bw_failed_transitions.append(transition)
                bw_pbar.update(1)

        obs = next_obs

        if done:
            print(rew)
            env.set_task_id(env.task_graph())
            print(f"current task id: {env.task_id}")
            obs, _ = env.reset()
            obs = env.get_front_cam_obs()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
