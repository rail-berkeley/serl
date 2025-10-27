#!/usr/bin/env python3
import sys
import os
import getpass

username = getpass.getuser()
env = "serl"
print("Ensure to change anaconda3 or miniconda3 and change your environment name")
print("Conda Environment name is: ", env)
sys.path.append(
    "/home/" + username + "/anaconda3/envs/" + env + "/lib/python3.10/site-packages"
)
import rclpy
import rclpy.duration
from rclpy.executors import MultiThreadedExecutor
import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

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
from kuka_server.robot_interface import RobotInterfaceNode
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

import os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(MAIN_DIR)
from serl_robot_infra.kuka_env.envs.relative_env import RelativeFrame
from serl_robot_infra.kuka_env.envs.wrappers import (
    Quat2EulerWrapper,
)

import serl_robot_infra.franka_env
import serl_robot_infra.kuka_env

# sys.argv = sys.argv[:1]

# # `app.run` calls `sys.exit`
# try:
#   app.run(lambda argv: None)
# except:
#   pass

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "KukaPegInsert-Vision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")

## TODO :: NISARA :: change max trajectory length to at least 300 (or see what the new data wants)
flags.DEFINE_integer("max_traj_length", 200, "Maximum length of trajectory.")

flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")
flags.DEFINE_boolean(
    "load_checkpoint", False, "Whether to start from previous checkpoint or not."
)
flags.DEFINE_string("load_checkpoint_path", None, "Checkpoint to start training from.")

flags.DEFINE_integer("batch_size", 256, "Batch size for training the policy.")
flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("loaded_checkpoint_step", 1000, "Loaded Checkpoint Step.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step"
)
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        print("Evaluating the policy")
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in tqdm.tqdm(range(FLAGS.eval_n_trajs), "Evaluation Trajectory"):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    # argmax=True,
                    seed=key,
                    # deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    done = False

    # training loop
    print("Entering Training Loop")
    timer = Timer()
    running_return = 0.0

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="actor"):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))
                # nisara : Comment
                # print("Actions in Training Loop: ", actions)

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")

            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                stats = {"train": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent: DrQAgent, replay_buffer, demo_buffer):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    # wandb_logger = None
    wandb_logger = make_wandb_logger(
        project="serl_testing",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=200
            )

        update_steps += 1


##############################################################################


def main(_):
    rclpy.init(args=_)
    print("ROS2 initialized.")
    robot_interface_node = RobotInterfaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(robot_interface_node)
    robot_interface_node.get_logger().info("Robot interface node started.")
    # executor.spin_once()

    assert FLAGS.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    print("Initializing environment")
    # create env and load dataset
    print("Value of learner: ", FLAGS.learner)

    env = gym.make(
        FLAGS.env,
        fake_env=FLAGS.learner,
        save_video=FLAGS.eval_checkpoint_step,
        robot_interface_node=robot_interface_node,
    )
    print("Environment initialized")
    # env = GripperCloseEnv(env)
    # if FLAGS.actor:
    #     env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)
    print("Environment wrapped")

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    rng, sampling_rng = jax.random.split(rng)
    print("Creating agent")
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    if FLAGS.load_checkpoint:
        print(f"Loading Checkpoint from Previous Run:{FLAGS.load_checkpoint_path}")

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.load_checkpoint_path,
            agent.state,
            step=FLAGS.loaded_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

    print("Agent created")
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    print("Replicating agent")
    agent: DrQAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        print("Learner code executed")
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
        )
        print("Replay buffer initialized")
        print("Checkpoint Path: ", FLAGS.checkpoint_path)
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=10000,
            image_keys=image_keys,
        )
        import pickle5 as pkl

        with open(FLAGS.demo_path, "rb") as f:
            trajs = pkl.load(f, fix_imports=True)
            for traj in trajs:
                demo_buffer.insert(traj)
        print(f"demo buffer size: {len(demo_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(sampling_rng, agent=agent, replay_buffer=replay_buffer, demo_buffer=demo_buffer)

    elif FLAGS.actor:
        print("Initializing Actor Node")
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")

    rclpy.shutdown()


if __name__ == "__main__":
    app.run(main)
