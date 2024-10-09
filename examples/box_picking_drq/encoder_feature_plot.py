#!/usr/bin/env python3

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

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.wrappers import SpacemouseIntervention, Quat2EulerWrapper

import ur_env

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "box_picking_camera_env", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", "DRQ first tests", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_string("camera_mode", "rgb", "Camera mode, one of (rgb, depth, both)")

flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")
flags.DEFINE_integer("utd_ratio", 4, "UTD ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 100000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 0, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 0, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 10, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 1000, "Evaluation period.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", '/home/nico/real-world-rl/serl/examples/box_picking_drq/checkpoints',
                    "Path to save checkpoints.")

flags.DEFINE_integer("eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

flags.DEFINE_string("log_rlds_path", '/home/nico/real-world-rl/serl/examples/box_picking_sac/rlds',
                    "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    FLAGS.checkpoint_path = FLAGS.checkpoint_path + " " + datetime.now().strftime("%m%d-%H:%M")

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env = gym.make(
        FLAGS.env,
        camera_mode="rgb",
        fake_env=True,
        max_episode_length=100,
    )
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    demo_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=2000,
        image_keys=image_keys,
    )

    sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())

    import pickle as pkl

    with open("/examples/box_picking_drq/old_demos/box_picking_20_demos_mai3_streamlined.pkl", "rb") as f:
        trajs = pkl.load(f)
        for traj in trajs:
            demo_buffer.insert(traj)
    print(f"demo buffer size: {len(demo_buffer)}")

    # encode here
    # encode(demo_buffer, agent)

    # plot here
    plot(demo_buffer)


def encode(demo_buffer, agent):
    for i in range(len(demo_buffer) // 10):
        index = np.arange(10) + i * 10
        obs = demo_buffer.sample(batch_size=10, indx=index)

        actions = agent.sample_actions(
            observations=jax.device_put(obs["observations"]),
            argmax=True
        )


def plot(demo_buffer):
    import sys
    sys.path.append("/home/nico/real-world-rl/spacemouse_tests")
    from spacemouse_tests.jax_feature_plotter import generate_and_save_images
    feature_file = "/spacemouse_tests/feature_plots_meanstd/features_meanstd.npy"

    with open(feature_file, 'rb') as f:
        features = np.load(f)

    print("features shape ", features.shape)
    shoulder_f = features[:, :256].reshape((800, 16, 16))  # shoulder comes first
    wrist_f = features[:, 256:512].reshape((800, 16, 16))

    # get images here
    shoulder, wrist = [], []
    s_next, w_next = [], []
    for i in range(len(demo_buffer) // 10):
        index = np.arange(10) + i * 10
        obs = demo_buffer.sample(batch_size=10, indx=index)
        wrist.append(obs["observations"]["wrist"])
        shoulder.append(obs["observations"]["shoulder"])

        w_next.append(obs["next_observations"]["wrist"])
        s_next.append(obs["next_observations"]["shoulder"])

    w_next = np.array(w_next).reshape((800, 128, 128, 3))
    s_next = np.array(s_next).reshape((800, 128, 128, 3))
    wrist = np.array(wrist).reshape((800, 128, 128, 3))
    shoulder = np.array(shoulder).reshape((800, 128, 128, 3))

    output_folder = '/home/nico/real-world-rl/spacemouse_tests/feature_plots_meanstd'
    generate_and_save_images(output_folder, [shoulder, wrist, shoulder_f, wrist_f])


if __name__ == "__main__":
    app.run(main)
