import gym
from tqdm import tqdm
import numpy as np
import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    BinaryRewardClassifierWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

import jax
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "reward_classifier_ckpt_path", None, "Path to reward classifier ckpt."
)


def main(_):
    env = gym.make("FrankaCableRoute-Vision-v0", save_video=False)
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [k for k in env.observation_space.keys() if "state" not in k]

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    classifier_func = load_classifier_func(
        key=key,
        sample=env.observation_space.sample(),
        image_keys=image_keys,
        checkpoint_path=FLAGS.reward_classifier_ckpt_path,
    )
    env = BinaryRewardClassifierWrapper(env, classifier_func)
    obs, _ = env.reset()

    for i in tqdm(range(1000)):
        actions = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action=actions)

        if "intervene_action" in info:
            actions = info["intervene_action"]

        obs = next_obs

        if done:
            print("Reward: ", rew)
            obs, _ = env.reset()


if __name__ == "__main__":
    app.run(main)
