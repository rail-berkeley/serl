import gym
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
    FWBWFrontCameraBinaryRewardClassifierWrapper,
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

    image_keys = [k for k in env.front_observation_space.keys() if "state" not in k]

    from serl_launcher.networks.reward_classifier import load_classifier_func
    import jax

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    fw_classifier_func = load_classifier_func(
        key=key,
        sample=env.front_observation_space.sample(),
        image_keys=image_keys,
        checkpoint_path="/home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/fw_classifier_ckpt",
    )
    rng, key = jax.random.split(rng)
    bw_classifier_func = load_classifier_func(
        key=key,
        sample=env.front_observation_space.sample(),
        image_keys=image_keys,
        checkpoint_path="/home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bw_classifier_ckpt",
    )
    env = FWBWFrontCameraBinaryRewardClassifierWrapper(
        env, fw_classifier_func, bw_classifier_func
    )

    transitions_needed = 2000
    fw_transitions = []
    bw_transitions = []

    fw_pbar = tqdm(total=transitions_needed, desc="fw")
    bw_pbar = tqdm(total=transitions_needed, desc="bw")

    while (
        len(fw_transitions) < transitions_needed
        or len(bw_transitions) < transitions_needed
    ):
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

        if env.task_id == 0 and len(fw_transitions) < transitions_needed:
            fw_transitions.append(transition)
            fw_pbar.update(1)
        elif env.task_id == 1 and len(bw_transitions) < transitions_needed:
            bw_transitions.append(transition)
            bw_pbar.update(1)

        obs = next_obs

        if done:
            print(rew)
            next_task_id = env.task_graph(env.get_front_cam_obs())
            print(f"transition from {env.task_id} to next task: {next_task_id}")
            env.set_task_id(next_task_id)
            obs, _ = env.reset()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"fw_bin_{len(fw_transitions)}_demo_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(fw_transitions, f)
        print(f"saved {len(fw_transitions)} transitions to {file_name}")

    file_name = f"bw_bin_{len(bw_transitions)}_demo_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(bw_transitions, f)
        print(f"saved {len(bw_transitions)} transitions to {file_name}")
