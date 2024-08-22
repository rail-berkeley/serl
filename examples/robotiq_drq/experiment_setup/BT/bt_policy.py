import numpy as np
from BehaviorTree import BehaviorTree

import copy
import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pynput
import threading
import tqdm
from absl import app, flags
from flax.training import checkpoints
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper
from serl_launcher.wrappers.observation_statistics_wrapper import ObservationStatisticsWrapper
from robotiq_env.envs.relative_env import RelativeFrame
from robotiq_env.envs.wrappers import Quat2MrpWrapper, ObservationRotationWrapper

import robotiq_env

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "robotiq_camera_env", "Name of environment.")
flags.DEFINE_string("exp_name", "BT agent", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of trajectories for evaluation.")


def main(_):
    env = gym.make(
        FLAGS.env,
        camera_mode="none",
        fake_env=False,
        max_episode_length=FLAGS.max_traj_length,
    )
    env = RelativeFrame(env)
    env = Quat2MrpWrapper(env)
    env = ScaleObservationWrapper(env)  # scale obs space (after quat2mrp, but before serlobs)
    env = ObservationStatisticsWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    agent = BehaviorTree()

    success_counter = 0
    time_list = []
    trajectories = []
    for episode in range(FLAGS.eval_n_trajs):
        trajectory = []
        obs, _ = env.reset()
        agent.reset()
        done = False
        start_time = time.time()
        while not done:
            actions = agent.sample_actions(
                observations=obs
            )

            next_obs, reward, done, truncated, info = env.step(actions)
            transition = dict(
                observations={"state": obs["state"].copy()},  # do not save voxel grid or images
                actions=actions,
                next_observations={"state": next_obs["state"].copy()},
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            trajectory.append(transition)
            obs = next_obs

            if done:
                dt = time.time() - start_time
                if reward > 50.:
                    time_list.append(dt)
                    print(f"time: {dt}")

                success_counter += (reward > 50.)
                print(f"{success_counter}/{episode + 1}")
                trajectories.append({"traj": trajectory, "time": dt, "success": (reward > 50.)})


if __name__ == "__main__":
    app.run(main)