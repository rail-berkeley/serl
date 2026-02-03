#!/usr/bin/env python3

import time
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)

from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer

import franka_sim

# from demos.demoHandling import DemoHandling

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "HalfCheetah-v4", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_string("run_name", None, "Name of run for wandb logging")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic to actor update ratio.")
flags.DEFINE_integer("port_number", 5488, "Port for server")
flags.DEFINE_integer("broadcast_port", 5489, "Port for server")
flags.DEFINE_boolean("wandb_offline", False, "Save locally to be synced with 'wandb sync <wandb_dir>")
flags.DEFINE_string("wandb_output_dir", None, "Where to save local wandb files")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a learner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

# flags for replay buffer
flags.DEFINE_string("replay_buffer_type", "replay_buffer", "Which replay buffer to use")
flags.DEFINE_string("branch_method", None, "Method for how many branches to generate")
flags.DEFINE_string("split_method", None, "Method for when to change number of branches generated")
flags.DEFINE_float("workspace_width", 0.5, "Workspace width in meters")
flags.DEFINE_integer("max_depth",None,"Maximum layers of depth")
flags.DEFINE_integer("starting_branch_count", None, "Initial number of branches")
flags.DEFINE_integer("branching_factor", None, "Rate of change of branches per dimension (x,y)") # For fractal_branch and fractal_contraction
flags.DEFINE_float("alpha",None,"alpha value")
flags.DEFINE_enum("disassociated_type", None, ["octahedron", "hourglass"], 
                  "Type of disassociated fracal rollout. Octahedron: expand from min to max then contract to min,"
                   + " Hourglass: Contract from max to min then expand to max")
flags.DEFINE_integer("min_branch_count", None, "Minimum number of branches for disassociated fractal rollout")
flags.DEFINE_integer("max_branch_count", None, "Maximum number of branches for disassociated fractal rollout")

# Debug
flags.DEFINE_boolean("debug", False, "Debug mode.")  # debug mode will disable wandb logging

# Logging
flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")


# Load demonstation data
flags.DEFINE_boolean("load_demos", False, "Whether to load demo dataset.")
flags.DEFINE_string("demo_dir", "/data/data/serl/demos", "Path to demo dataset.")
flags.DEFINE_string("file_name", "data_franka_reach_random_20.npz", "Name of the demo file to load.")

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: SACAgent, data_store, env, sampling_rng, demos_handler=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(port_number=FLAGS.port_number, broadcast_port=FLAGS.broadcast_port),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    eval_env = gym.make(FLAGS.env)
    #if FLAGS.env == "PandaPickCube-v0":
    eval_env = gym.wrappers.FlattenObservation(eval_env) ## Note!! 
    eval_env = RecordEpisodeStatistics(eval_env)

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0

    # Load demos: handler.run will insert all transition demo data into the data store.
    if FLAGS.load_demos:
        with timer.context("sample and step into env with loaded demos"):
            
            # Insert complete demonstration into the data store 
            print(f"Inserting {demos_handler.data['transition_ctr']} transitions into the data store.")
            demos_handler.insert_data_to_buffer(data_store)
            FLAGS.random_steps = 0  # Set random steps to 0 since we have demo data
    # For subsequent steps, sample actions from the agent
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
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

            # Step environment
            with timer.context("step_env"):

                next_obs, reward, done, truncated, info = env.step(actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                reward = np.asarray(reward, dtype=np.float32)

                running_return += reward

                data_store.insert(
                    dict(
                        observations=obs,
                        actions=actions,
                        next_observations=next_obs,
                        rewards=reward,
                        masks=1.0 - done,
                        dones=done or truncated,
                    )
                )

                obs = next_obs
                if done or truncated:
                    running_return = 0.0
                    obs, _ = env.reset()

        if FLAGS.render:
            env.render()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################

 
def learner(rng, agent: SACAgent, replay_buffer, replay_iterator):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project=FLAGS.exp_name,
        name=FLAGS.run_name,
        description=FLAGS.exp_name or FLAGS.env,
        # wandb_output_dir=FLAGS.wandb_output_dir,
        debug=FLAGS.debug,
        # offline=FLAGS.wandb_offline,
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
    server = TrainerServer(make_trainer_config(port_number=FLAGS.port_number, broadcast_port=FLAGS.broadcast_port), request_callback=stats_callback)
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

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # Train the networks
        with timer.context("sample_replay_buffer"):
            batch = next(replay_iterator)

        with timer.context("train"):
            agent, update_info = agent.update_high_utd(batch, utd_ratio=FLAGS.critic_actor_ratio)
            agent = jax.block_until_ready(agent)

            # publish the updated network
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.render:
        env = gym.make(FLAGS.env, render_mode="human")
    else:
        env = gym.make(FLAGS.env)
    
    if FLAGS.env in {"PandaPickCube-v0", "PandaReachCube-v0", "PandaPickSparseCube-v0", "PandaReachSparseCube-v0"}:
        x_obs_idx=np.array([0,4])
        y_obs_idx=np.array([1,5])
    else:
        raise NotImplementedError(f"Unknown observation layout for {FLAGS.env}")
    
    env = gym.wrappers.FlattenObservation(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: SACAgent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    # Demo Data
    if FLAGS.load_demos:
        print_green("Setting demo parameters")
        # Create a handler for the demo data
        demos_handler = DemoHandling(
            demo_dir=FLAGS.demo_dir,
            file_name=FLAGS.file_name,
        )

        # 1. Modify actor data_store size
        # Extract number of demo transitions
        demo_transitions = demos_handler.get_num_transitions()
        
        if demo_transitions > 2000:
            qds_size = demo_transitions + 1000  # Increment the queue size on the actor
        else:
           qds_size = 2000  # the original queue size on the actor

        # 2. Modify training starts (since we have good data)
        FLAGS.training_starts = 1

    else:
        demos_handler = None        
        qds_size = 2000  # the original queue size on the actor


    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type=FLAGS.replay_buffer_type,
            branch_method=FLAGS.branch_method,
            split_method=FLAGS.split_method,
            branching_factor=FLAGS.branching_factor,
            starting_branch_count=FLAGS.starting_branch_count,
            workspace_width=FLAGS.workspace_width,
            max_traj_length=FLAGS.max_traj_length,
            x_obs_idx=x_obs_idx,
            y_obs_idx=y_obs_idx,
            preload_rlds_path=FLAGS.preload_rlds_path,
            max_depth=FLAGS.max_depth,
            alpha=FLAGS.alpha,
            disassociated_type=FLAGS.disassociated_type,
            min_branch_count=FLAGS.min_branch_count,
            max_branch_count=FLAGS.max_branch_count,
        )
        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.critic_actor_ratio,
            },
            device=sharding.replicate(),
        )
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            replay_iterator=replay_iterator,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

        if FLAGS.load_demos:
            print_green("loading demo data")            

            # Create a data store for the actor
            data_store = QueuedDataStore(qds_size)  # the queue size on the actor
        else:
            print_green("no demo data, using empty data store")
            # Create a data store for the actor
            data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng, demos_handler)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
