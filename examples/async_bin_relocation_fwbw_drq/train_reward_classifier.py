import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.core import frozen_dict
from flax.training import checkpoints
import optax
from tqdm import tqdm
import gymnasium as gym
import os
from absl import app, flags

from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    populate_data_store,
)
from serl_launcher.networks.reward_classifier import create_classifier

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
)
from franka_env.envs.relative_env import RelativeFrame

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("positive_demo_paths", None, "paths to positive demos")
flags.DEFINE_multi_string("negative_demo_paths", None, "paths to negative demos")
flags.DEFINE_string("classifier_ckpt_path", None, "Path to classifier checkpoint")


def main(_):
    num_epochs = 100
    batch_size = 256

    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)

    env = gym.make("FrankaBinRelocation-Vision-v0", save_video=False)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)
    image_keys = [k for k in env.front_observation_space.keys() if "state" not in k]

    pos_buffer = MemoryEfficientReplayBufferDataStore(
        env.front_observation_space,
        env.action_space,
        capacity=2000,
        image_keys=image_keys,
    )
    pos_buffer = populate_data_store(pos_buffer, FLAGS.positive_demo_paths)

    neg_buffer = MemoryEfficientReplayBufferDataStore(
        env.front_observation_space,
        env.action_space,
        capacity=5000,
        image_keys=image_keys,
    )
    neg_buffer = populate_data_store(neg_buffer, FLAGS.negative_demo_paths)

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": batch_size // 2,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": batch_size // 2,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, sample["next_observations"], image_keys)

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    # Define the training step
    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["data"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["data"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    # Training Loop
    for epoch in tqdm(range(num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        sample = concat_batches(
            pos_sample["next_observations"], neg_sample["observations"], axis=0
        )
        rng, key = jax.random.split(rng)
        sample = data_augmentation_fn(key, sample)
        labels = jnp.concatenate(
            [jnp.ones((batch_size // 2, 1)), jnp.zeros((batch_size // 2, 1))], axis=0
        )
        batch = {"data": sample, "labels": labels}

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    checkpoints.save_checkpoint(
        FLAGS.classifier_ckpt_path,
        classifier,
        step=num_epochs,
        overwrite=True,
    )


if __name__ == "__main__":
    app.run(main)
