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

from jaxrl_m.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from jaxrl_m.common.encoding import EncodingWrapper
from jaxrl_m.envs.wrappers.chunking import ChunkingWrapper
from jaxrl_m.utils.train_utils import concat_batches
from jaxrl_m.vision.data_augmentations import batched_random_crop

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    Quat2EulerWrapper,
)

import franka_env
from franka_env.envs.relative_env import RelativeFrame


class BinaryClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def create_classifier(key, sample, image_keys):
    pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True,
        name="pretrained_encoder",
    )
    encoders = {
        image_key: PreTrainedResNetEncoder(
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=256,
            pretrained_encoder=pretrained_encoder,
            name=f"encoder_{image_key}",
        )
        for image_key in image_keys
    }
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=True,
        image_keys=image_keys,
    )

    classifier_def = BinaryClassifier(encoder_def=encoder_def)
    params = classifier_def.init(key, sample)["params"]
    classifier_def = BinaryClassifier(encoder_def=encoder_def)
    classifier = TrainState.create(
        apply_fn=classifier_def.apply,
        params=params,
        tx=optax.adam(learning_rate=1e-4),
    )

    file_name = "./resnet10_params.pkl"
    with open(file_name, "rb") as f:
        encoder_params = pkl.load(f)
    param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )
    new_params = classifier.params.unfreeze()
    for image_key in image_keys:
        if "pretrained_encoder" in new_params["encoder_def"][f"encoder_{image_key}"]:
            for k in new_params["encoder_def"][f"encoder_{image_key}"][
                "pretrained_encoder"
            ]:
                if k in encoder_params:
                    new_params["encoder_def"][f"encoder_{image_key}"][
                        "pretrained_encoder"
                    ][k] = encoder_params[k]
                    print(f"replaced {k} in encoder_{image_key}")
    from flax.core.frozen_dict import freeze

    new_params = freeze(new_params)
    classifier = classifier.replace(params=new_params)
    return classifier


def load_classifier_func(key, sample, image_keys):
    classifier = create_classifier(key, sample, image_keys)
    classifier = checkpoints.restore_checkpoint(
        "/home/undergrad/code/serl_dev/examples/async_cable_route_drq/classifier_ckpt",
        target=classifier,
        step=100,
    )
    func = lambda obs: classifier.apply_fn(
        {"params": classifier.params}, obs, train=False
    )
    func = jax.jit(func)
    return func


if __name__ == "__main__":
    num_epochs = 100
    batch_size = 256

    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)

    env = gym.make("FrankaRobotiqCableRoute-Vision-v0", fake_env=True, save_video=False)
    env = GripperCloseEnv(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [k for k in env.observation_space.keys() if "state" not in k]

    success_data = pkl.load(
        open("cable_route_200_success_2024-01-03_22-17-54.pkl", "rb")
    )
    pos_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=5000,
        image_keys=image_keys,
    )
    for traj in success_data:
        pos_buffer.insert(traj)

    neg_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=10000,
        image_keys=image_keys,
    )
    demos = pkl.load(open("cable_route_10_demos_2024-01-03_22-22-56.pkl", "rb"))
    failed_data = [d for d in demos if not d["dones"]]
    for traj in failed_data:
        neg_buffer.insert(traj)
    success_data = [d for d in demos if d["dones"]]
    for traj in success_data:
        pos_buffer.insert(traj)

    demos = pkl.load(open("cable_route_10_demos_2024-01-03_22-25-50.pkl", "rb"))
    failed_data = [d for d in demos if not d["dones"]]
    success_data = [d for d in demos if d["dones"]]
    for traj in failed_data:
        neg_buffer.insert(traj)
    for traj in success_data:
        pos_buffer.insert(traj)

    demos = pkl.load(open("cable_route_20_failed_2024-01-04_12-24-11.pkl", "rb"))
    failed_data = [d for d in demos if not d["dones"]]
    for traj in failed_data:
        neg_buffer.insert(traj)

    demos = pkl.load(open("cable_route_20_failed_2024-01-04_12-50-24.pkl", "rb"))
    failed_data = [d for d in demos if not d["dones"]]
    for traj in failed_data:
        neg_buffer.insert(traj)

    demos = pkl.load(open("cable_route_20_demos_2024-01-04_12-10-54.pkl", "rb"))
    failed_data = [d for d in demos if not d["dones"]]
    success_data = [d for d in demos if d["dones"]]
    for traj in failed_data:
        neg_buffer.insert(traj)
    for traj in success_data:
        pos_buffer.insert(traj)

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
        "/home/undergrad/code/serl_dev/examples/async_cable_route_drq/classifier_ckpt",
        classifier,
        step=num_epochs,
        overwrite=True,
    )
