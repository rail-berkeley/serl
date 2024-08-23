#!/usr/bin/env python3
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

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.sampling_utils import TemporalActionEnsemble
from serl_launcher.utils.train_utils import (
    print_agent_params,
    parameter_overview,
    plot_feature_kernel_histogram,
    find_zero_weights,
    plot_conv3d_kernels,
)

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper
from serl_launcher.wrappers.observation_statistics_wrapper import ObservationStatisticsWrapper
from robotiq_env.envs.relative_env import RelativeFrame
from robotiq_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper, ObservationRotationWrapper
from serl_launcher.vision.data_augmentations import batched_random_rot90_state, batched_random_rot90_voxel, \
    batched_random_rot90_action

import robotiq_env

# used to debug nan errors (also in jit-ed functions)
# jax.config.update("jax_debug_nans", True)

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "robotiq_camera_env", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", "DRQ agent", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_string("camera_mode", "rgb", "Camera mode, one of (rgb, depth, both)")

flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("utd_ratio", 4, "UTD ratio.")

flags.DEFINE_string("state_mask", "no_ForceTorque",
                    "if all the states should be considered, possible: (all, none, no_ForceTorque, gripper, position_gripper)")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_integer("encoder_bottleneck_dim", 128, "bottleneck dimension of the encoder")
# flags.DEFINE_integer("proprio_latent_dim", 64,
#                     "the latent dimension for the state, will be concatenated with encoder bottleneck dim before being passed onward")
flags.DEFINE_multi_string("encoder_kwargs", None, "Encoder kwargs in the form ['dict key', 'dict value']")
flags.DEFINE_bool("enable_obs_rotation_wrapper", False,
                  "Whether to enable observation rotation wrapper (train in one quaternion)")
flags.DEFINE_bool("enable_obs_rotation_augmentation", False,
                  "Whether to enable observation rotation augmentation (90 deg)")
flags.DEFINE_bool("enable_temporal_ensemble_sampling", False,
                  "Whether to enable sampling the action from a temporal ensemble: action = 0.5*a0 + 0.3*a-1 + 0.2*a-2 + 0.1*a-3")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 10000,
                     "Replay buffer capacity.")  # quite low to forget demo trajectories

flags.DEFINE_integer("random_steps", 0, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 0, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 10, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 1000, "Evaluation period in seconds")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", '/home/nico/real-world-rl/serl/examples/robotiq_drq/checkpoints',
                    "Path to save checkpoints.")

flags.DEFINE_integer("eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step")
flags.DEFINE_string("log_rlds_path", '/home/nico/real-world-rl/serl/examples/robotiq_drq/rlds',
                    "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


PAUSE_EVENT_FLAG = threading.Event()
PAUSE_EVENT_FLAG.clear()  # clear() to continue the actor/learner loop, set() to pause


def pause_callback(key):
    """Callback for when a key is pressed"""
    global PAUSE_EVENT_FLAG
    try:
        # chosen a rarely used key to avoid conflicts. this listener is always on, even when the program is not in focus
        if not PAUSE_EVENT_FLAG.is_set() and key == pynput.keyboard.Key.pause:
            print("Requested pause training")
            # set the PAUSE FLAG to pause the actor/learner loop
            PAUSE_EVENT_FLAG.set()
    except AttributeError:
        # print(f'{key} pressed')
        pass


listener = pynput.keyboard.Listener(
    on_press=pause_callback
)  # to enable keyboard based pause
listener.start()


##############################################################################


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    global PAUSE_EVENT_FLAG

    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        find_zero_weights(agent.state.params, print_all=False)
        action_ensemble = TemporalActionEnsemble(activated=FLAGS.enable_temporal_ensemble_sampling)

        # examine model parameters if trajs==0
        if FLAGS.eval_n_trajs == 0:
            # parameter_overview(agent)
            # plot_feature_kernel_histogram(agent)
            plot_conv3d_kernels(agent.state.params)

        trajectories = []
        for episode in range(FLAGS.eval_n_trajs):
            trajectory = []
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            action_ensemble.reset()
            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )

                ensembled_action = action_ensemble.sample(actions)
                next_obs, reward, done, truncated, info = env.step(ensembled_action)
                transition = dict(
                    observations={"state": obs["state"].copy()},  # do not save voxel grid or images
                    actions=ensembled_action,
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

            # if pause event is requested, pause the actor
            if PAUSE_EVENT_FLAG.is_set():
                print("Actor eval loop interrupted")
                response = input("Do you want to continue (c), or exit (e)? ")
                if response == "c":
                    # update PAUSE FLAG to continue training
                    PAUSE_EVENT_FLAG.clear()
                    print("Continuing")
                else:
                    print("Stopping actor eval")
                    break

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs if FLAGS.eval_n_trajs else 0.}")
        print(f"average time: {np.mean(time_list)}")
        with open("trajectories.pkl", "wb") as f:
            import pickle
            pickle.dump(trajectories, f)
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

    # training loop
    timer = Timer()
    running_return = 0.0

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            elif not agent.config["activate_batch_rotation"]:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))
            else:
                sampling_rng, rot_rng, key = jax.random.split(sampling_rng, 3)

                rotated_obs = copy.deepcopy(obs)
                rotated_obs["state"] = batched_random_rot90_state(obs["state"], rot_rng)
                rotated_obs["wrist_pointcloud"] = batched_random_rot90_voxel(obs["wrist_pointcloud"], rot_rng)

                actions = agent.sample_actions(
                    observations=jax.device_put(rotated_obs),
                    seed=key,
                    deterministic=False,
                )
                for _ in range(3):
                    actions = batched_random_rot90_action(actions[None, ...], rot_rng)[0, ...]  # rotate back

                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")

            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return = running_return * 0.99 + reward
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
                print(f"running return: {running_return}")
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        timer.tock("total")

        if FLAGS.eval_period and step % FLAGS.eval_period == 0 and step:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

        if PAUSE_EVENT_FLAG.is_set():
            print_green("Actor loop interrupted")
            response = input(
                "Do you want to continue (c), save replay buffer and exit (s) or simply exit (e)? "
            )
            if response == "c":
                print("Continuing")
                PAUSE_EVENT_FLAG.clear()
            else:
                if response == "s":
                    print("Saving replay buffer")
                    data_store.save(
                        "replay_buffer_actor.npz"
                    )  # not yet supported for QueuedDataStore
                else:
                    print("Replay buffer not saved")
                print("Stopping actor client")
                client.stop()
                break


##############################################################################


def learner(rng, agent: DrQAgent, replay_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # To track the step in the training loop
    update_steps = 0
    global PAUSE_EVENT_FLAG

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

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        timer.tick("learner_total")

        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.utd_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(batch, )

        with timer.context("train"):
            batch = next(replay_iterator)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        timer.tock("learner_total")

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
            wandb_logger.log({"replay_buffer_size": len(replay_buffer)})

        update_steps += 1

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100
            )

        if PAUSE_EVENT_FLAG.is_set():
            print("Learner loop interrupted")
            response = input(
                "Do you want to continue (c), save training state and exit (s) or simply exit (e)? "
            )
            if "c" in response:
                print("Continuing")
                PAUSE_EVENT_FLAG.clear()
            else:
                if response == "s":
                    print("Saving learner state")
                    agent_ckpt = checkpoints.save_checkpoint(
                        FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100
                    )
                    replay_buffer.save(
                        "replay_buffer_learner.npz"
                    )  # not yet supported for QueuedDataStore
                    # TODO: save other parts of training state
                else:
                    print("Training state not saved")
                print("Stopping learner client")
                break

    server.stop()
    parameter_overview(agent)  # print end state


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    if FLAGS.checkpoint_path.split('/')[-1] == "checkpoints":
        FLAGS.checkpoint_path = FLAGS.checkpoint_path + " " + FLAGS.exp_name + " " + datetime.now().strftime(
            "%m%d-%H:%M")

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env = gym.make(
        FLAGS.env,
        camera_mode=FLAGS.camera_mode,
        fake_env=FLAGS.learner,
        max_episode_length=FLAGS.max_traj_length,
    )
    # if FLAGS.actor:
    #     env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2MrpWrapper(env)
    env = ScaleObservationWrapper(env)  # scale obs space (after quat2mrp, but before serlobs)
    env = ObservationStatisticsWrapper(env)
    if FLAGS.enable_obs_rotation_wrapper:
        env = ObservationRotationWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    print(f"image keys: {image_keys}")

    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.encoder_kwargs is None or len(FLAGS.encoder_kwargs) % 2 == 0
    encoder_kwargs = {
        "bottleneck_dim": FLAGS.encoder_bottleneck_dim,
        **(dict(zip(*[iter(FLAGS.encoder_kwargs)] * 2)) if FLAGS.encoder_kwargs else {}),
    }
    encoder_kwargs = {k: (int(v) if str(v).isdigit() else v) for k, v in encoder_kwargs.items()}

    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
        state_mask=FLAGS.state_mask,
        # proprio_latent_dim=FLAGS.proprio_latent_dim,
        encoder_kwargs=encoder_kwargs
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    # print useful info
    print_agent_params(agent, image_keys)
    parameter_overview(agent)
    # plot_conv3d_kernels(agent.state.params)

    agent.config["activate_batch_rotation"] = FLAGS.enable_obs_rotation_augmentation  # obs batch rotation control
    if FLAGS.enable_obs_rotation_augmentation:
        print("Batch Observation Rotation enabled!")
    assert not FLAGS.enable_obs_rotation_augmentation or not FLAGS.enable_obs_rotation_wrapper  # both is pointless

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="paper_experiments",
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        import pickle as pkl
        with open(FLAGS.demo_path, "rb") as f:
            trajs = pkl.load(f)

            # check which observations can be ignored for this run
            to_pop = []
            for obs_name in [i for i in trajs[0]["observations"].keys()]:
                if obs_name not in env.observation_space.spaces:
                    to_pop.append(obs_name)
            print(f"ignored {to_pop} observation in the demo trajectories")

            for traj in trajs:
                for obs_name in to_pop:
                    traj["observations"].pop(obs_name)
                    traj["next_observations"].pop(obs_name)

                # convert to grey here
                if FLAGS.camera_mode == "grey":
                    gray = np.array([0.2989, 0.5870, 0.1140])
                    traj["observations"]["wrist"] = np.dot(traj["observations"]["wrist"], gray)[..., None]
                    traj["next_observations"]["wrist"] = np.dot(traj["next_observations"]["wrist"], gray)[..., None]

                replay_buffer.insert(traj)
        print(f"replay buffer size: {len(replay_buffer)}")

        # learner loop
        print_green("starting learner loop")
        try:
            learner(
                sampling_rng,
                agent,
                replay_buffer=replay_buffer,
                wandb_logger=wandb_logger,
            )
        except KeyboardInterrupt:
            print_green("leraner loop interrupted")
        finally:
            # Wrap up the learner loop
            env.close()
            print("Learner loop finished")

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        try:
            actor(agent, data_store, env, sampling_rng)
            print_green("actor loop finished")
        except (KeyboardInterrupt, RuntimeError) as e:
            print_green("actor loop interrupted: " + str(e))
        finally:
            env.close()

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
