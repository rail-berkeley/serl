from tqdm import tqdm
from absl import app, flags
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import pickle as pkl
import numpy as np
from copy import deepcopy
import time

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.agents.continuous.bc_noimg import BCAgentNoImg

from serl_launcher.utils.launcher import (
    make_bc_agent_no_img,
    make_wandb_logger,
    make_replay_buffer,
)
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    populate_data_store,
    populate_data_store_with_z_axis_only,
)
from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages
from serl_launcher.networks.reward_classifier import load_classifier_func
# from franka_env.envs.relative_env import RelativeFrame
from robotiq_env.envs.wrappers import SpacemouseIntervention, Quat2EulerWrapper

import robotiq_env

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "robotiq-grip-v1", "Name of environment.")
flags.DEFINE_string("agent", "bc_noimg", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", True, "Whether to save model.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")

flags.DEFINE_integer("max_steps", 100000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 100000, "Replay buffer capacity.")

flags.DEFINE_multi_string("demo_paths", "robotiq_grip_v1/robotiq_test_20_demos_2024-03-25_16-39-22.pkl",
                          "paths to demos")
flags.DEFINE_string("checkpoint_path", "/home/nico/real-world-rl/serl/examples/checkpoints",
                    "Path to save checkpoints.")

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
    print(FLAGS.env)
    env = gym.make(
        FLAGS.env,
        fake_env=not FLAGS.eval_checkpoint_step,
    )
    env = SpacemouseIntervention(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SerlObsWrapperNoImages(env)
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: BCAgentNoImg = make_bc_agent_no_img(  # replace with no img one
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
    )

    wandb_logger = make_wandb_logger(
        project="real-world-rl",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    if not FLAGS.eval_checkpoint_step:
        """
        Training Mode
        """
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        # load demos and populate to current replay buffer
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            type="replay_buffer",
            # rlds_logger_path=FLAGS.log_rlds_path,         # can be added to log
            # preload_rlds_path=FLAGS.preload_rlds_path,
        )

        print(f"loaded demos from {FLAGS.demo_paths}")
        replay_buffer = populate_data_store(replay_buffer, FLAGS.demo_paths)

        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size,
            },
            device=sharding.replicate(),
        )
        try:
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
        except KeyboardInterrupt:
            print("interrupted by user")

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

        try:
            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset()
                done = False
                is_failure = False
                is_success = False
                start_time = time.time()
                while not done:
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=False,
                        seed=rng,
                    )
                    actions = np.asarray(jax.device_get(actions))
                    print(f"sampled actions: {actions}")

                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

                    if is_failure:
                        done = True
                        print("terminated by user")

                    if is_success:
                        reward = 1
                        done = True
                        print("success, reset now")

                    if truncated:
                        reward = 0
                        done = True
                        print("truncated, reset now")

                    if done:
                        if not is_failure:
                            dt = time.time() - start_time
                            time_list.append(dt)
                            print(dt)

                        success_counter += reward
                        print(reward)
                        print(f"{success_counter}/{episode + 1}")

                wandb_logger.log(info, step=episode)

        except KeyboardInterrupt:
            print("interrupted by user, exiting...")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")

    env.close()


if __name__ == "__main__":
    app.run(main)
