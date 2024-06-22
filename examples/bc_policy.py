from tqdm import tqdm
from absl import app, flags
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import pickle as pkl
import numpy as np
from copy import deepcopy
import time

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.agents.continuous.bc import BCAgent

from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_wandb_logger,
)
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    populate_data_store,
    populate_data_store_with_z_axis_only,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    ZOnlyWrapper,
    BinaryRewardClassifierWrapper,
)

import franka_env

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "FrankaEnv-Vision-v0", "Name of environment.")
flags.DEFINE_string("agent", "bc", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", True, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

flags.DEFINE_integer("max_steps", 100, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 10000, "Replay buffer capacity.")
flags.DEFINE_bool(
    "gripper",
    False,
    "Whether to use GripperClose Env wrapper.",
)
flags.DEFINE_bool(
    "remove_xy",
    False,
    "Whether to remove x y cartesian coordinates from state and next_state to avoid causal confusion.",
)
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_multi_string("demo_paths", None, "paths to demos")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step"
)
flags.DEFINE_integer("eval_n_trajs", 100, "Number of trajectories for evaluation.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging
flags.DEFINE_string(
    "reward_classifier_ckpt_path",
    None,
    "Path to reward classifier checkpoint. Default: None",
)

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env = gym.make(
        FLAGS.env,
        fake_env=not FLAGS.eval_checkpoint_step,
        save_video=FLAGS.eval_checkpoint_step,
        max_episode_length=200,
    )
    if not FLAGS.gripper:
        env = GripperCloseEnv(env)
    if FLAGS.eval_checkpoint_step:
        env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    if FLAGS.remove_xy:
        env = ZOnlyWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    # use custom trained image-based reward classifier
    if FLAGS.reward_classifier_ckpt_path:
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        classifier_func = load_classifier_func(
            key=key,
            sample=env.observation_space.sample(),
            image_keys=image_keys,
            checkpoint_path=FLAGS.reward_classifier_ckpt_path,
        )
        env = BinaryRewardClassifierWrapper(env, classifier_func)

    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: BCAgent = make_bc_agent(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        encoder_type=FLAGS.encoder_type,
        image_keys=image_keys,
    )

    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    if not FLAGS.eval_checkpoint_step:
        """
        Training Mode
        """
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        # load demos and populate to current replay buffer
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
        )
        if FLAGS.remove_xy:
            # load demos, remove x y cartesian coordinates from state and next_state to avoid causal confusion
            replay_buffer = populate_data_store_with_z_axis_only(
                replay_buffer, FLAGS.demo_paths
            )
        else:
            replay_buffer = populate_data_store(replay_buffer, FLAGS.demo_paths)

        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

        for step in tqdm(range(FLAGS.max_steps)):
            batch = next(replay_iterator)
            agent, info = agent.update(batch)
            wandb_logger.log(info, step=step)

            if (step + 1) % 10000 == 0 and FLAGS.save_model:
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path,
                    agent.state,
                    step=step + 1,
                    keep=100,
                    overwrite=True,
                )

    else:
        """
        Evaluation Mode
        """
        from pynput import keyboard

        is_failure = False
        is_success = False

        def esc_on_press(key):
            nonlocal is_failure
            if key == keyboard.Key.esc:
                is_failure = True

        def space_on_press(key):
            nonlocal is_success
            if key == keyboard.Key.space and not is_success:
                is_success = True

        esc_listener = keyboard.Listener(on_press=esc_on_press)
        esc_listener.start()
        space_listener = keyboard.Listener(on_press=space_on_press)
        space_listener.start()

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        success_counter = 0
        time_list = []

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            is_failure = False
            is_success = False
            start_time = time.time()
            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if is_failure:
                    done = True
                    print("terminated by user")

                if is_success:
                    reward = 1
                    done = True
                    print("success, reset now")

                if done:
                    if not is_failure:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

            wandb_logger.log(info, step=episode)

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")


if __name__ == "__main__":
    app.run(main)
