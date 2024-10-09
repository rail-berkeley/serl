from tqdm import tqdm
from absl import app, flags
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import pickle as pkl
import numpy as np
from copy import deepcopy
import time
from datetime import datetime

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
from ur_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper
from ur_env.envs.relative_env import RelativeFrame
from serl_launcher.utils.sampling_utils import TemporalActionEnsemble


import ur_env

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "box_picking_basic_env", "Name of environment.")
flags.DEFINE_string("agent", "bc_noimg", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", True, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

flags.DEFINE_integer("max_steps", 100000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 100000, "Replay buffer capacity.")

flags.DEFINE_multi_string("demo_paths", None, "paths to demos")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("checkpoint_period", 10000, "Period to save checkpoints.")

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
        camera_mode="none",
        max_episode_length=100,
    )
    # env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2MrpWrapper(env)
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
        project="paper_experiments",  # TODO only temporary
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

                if step and step % FLAGS.checkpoint_period == 0 and FLAGS.save_model:
                    checkpoints.save_checkpoint(
                        FLAGS.checkpoint_path,
                        agent.state,
                        step=step,
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

        wandb_logger = make_wandb_logger(
            project="paper_evaluation_unseen",
            description=FLAGS.exp_name or FLAGS.env,
            debug=False,
        )
        action_ensemble = TemporalActionEnsemble(activated=False)
        success_counter = 0

        trajectories = []
        traj_infos = []
        for episode in range(FLAGS.eval_n_trajs):
            trajectory = []
            obs, _ = env.reset()
            done = False
            action_ensemble.reset()

            if len(trajectories) == 0:
                input("ready? record robot view as well!")

            start_time = time.time()

            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                    # seed=rng,
                )
                actions = np.asarray(actions)

                ensembled_action = action_ensemble.sample(actions)  # will return actions if not activated
                next_obs, reward, done, truncated, info = env.step(ensembled_action)
                transition = dict(
                    observations=obs.copy(),  # do not save voxel grid or images
                    actions=ensembled_action,
                    next_observations=next_obs.copy(),
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                trajectory.append(transition)
                obs = next_obs

                if done or truncated:
                    success_counter += (reward > 50.)
                    dt = time.time() - start_time
                    running_reward = np.sum(np.asarray([t["rewards"] for t in trajectory]))
                    running_reward = max(running_reward, -100.)

                    print(f"{success_counter}/{episode + 1} ", end=' ')
                    print(f"time: {dt:.3f}s  running_rew: {running_reward:.2f}")

                    trajectories.append({"traj": trajectory, "time": dt, "success": (reward > 50.)})
                    infos = {
                        "running_reward": running_reward,
                        "time": dt,
                        "success_rate": float(reward > 50.),
                        "action_cost": np.linalg.norm(np.asarray([t["actions"] for t in trajectory]), axis=1, ord=2).mean()
                    }
                    traj_infos.append(infos)
                    wandb_logger.log(infos, step=episode)

        traj_infos = {k: [d[k] for d in traj_infos] for k in traj_infos[0]}  # list of dicts to dict of lists
        mean_infos = {"mean_" + key: np.mean(val) for key, val in traj_infos.items()}
        wandb_logger.log(mean_infos)
        for key, value in mean_infos.items():
            print(f"{key}: {value:.3f}")

        with open(f"trajectories {datetime.now().strftime('%m-%d %H%M')}.pkl", "wb") as f:
            import pickle
            pickle.dump(trajectories, f)

    env.close()


if __name__ == "__main__":
    app.run(main)
