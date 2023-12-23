# !/usr/bin/env python3

# NOTE: this requires jaxrl_m to be installed:
#       https://github.com/rail-berkeley/jaxrl_minimal

from collections import deque
from functools import partial
from threading import Lock
from typing import Union, Iterable
import pickle as pkl
import os
import urllib.request as request

import gymnasium as gym
import jax
from serl_launcher.data.serl_replay_buffer import ReplayBuffer
from serl_launcher.data.serl_memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)

from edgeml.data.data_store import DataStoreBase
from edgeml.trainer import TrainerConfig

from jax import nn
from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.agents.continuous.drq import DrQAgent
from jaxrl_m.vision.small_encoders import SmallEncoder
from jaxrl_m.vision.mobilenet import MobileNetEncoder
from jaxrl_m.vision.resnet_v1 import resnetv1_configs


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity, pixel_keys=image_keys
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


##############################################################################


def make_sac_agent(seed, sample_obs, sample_action):
    return SACAgent.create_states(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.99,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )


def make_drq_agent(
    seed, sample_obs, sample_action, image_keys=("image",), encoder_type="small"
):
    if encoder_type == "small":
        encoder_defs = {
            image_key: SmallEncoder(
                features=(32, 64, 128, 256),
                kernel_sizes=(3, 3, 3, 3),
                strides=(2, 2, 2, 2),
                padding="VALID",
                pool_method="avg",
                bottleneck_dim=256,
                spatial_block_size=8,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    elif encoder_type == "mobilenet":
        from jeffnet.linen import create_model

        encoder, encoder_params = create_model(
            "tf_mobilenetv3_small_minimal_100", pretrained=True
        )
        encoder_defs = {
            image_key: MobileNetEncoder(
                encoder=encoder,
                params=encoder_params,
                pool_method="spatial_learned_embeddings",
                bottleneck_dim=256,
                spatial_block_size=8,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    elif encoder_type == "resnet":
        encoder_defs = {
            image_key: resnetv1_configs["resnetv1-10"](
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pre_trained_frozen=True,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    else:
        raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

    agent = DrQAgent.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder=encoder_defs,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.99,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )

    if encoder_type == "resnet":  # load pretrained weights for ResNet-10
        # agent = load_resnet10_params(agent, image_keys)
        # the auto-download would only work if we make the repo public.
        # We can uncomment the above line once we make the repo public.
        with open("resnet10_params.pkl", "rb") as f:
            encoder_params = pkl.load(f)
            param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
            print(
                f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
            )

        new_params = agent.state.params.unfreeze()
        for image_key in image_keys:
            for k in new_params["modules_actor"]["encoder"][f"encoder_{image_key}"]:
                if k in encoder_params:
                    new_params["modules_actor"]["encoder"][f"encoder_{image_key}"][
                        k
                    ] = encoder_params[k]
                    print(f"replaced {k} in encoder_{image_key}")
        from flax.core.frozen_dict import freeze

        new_params = freeze(new_params)
        agent = agent.replace(state=agent.state.replace(params=new_params))

    return agent


def load_resnet10_params(agent, image_keys=("image",)):
    file_name = "resnet10_params.pkl"
    # Construct the full path to the file
    file_path = os.path.expanduser(f"~/.serl/{file_name}")

    # Check if the file exists
    if os.path.exists(file_path):
        print(f"The ResNet-10 weights already exists at '{file_path}'.")
    else:
        url = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
        print(f"Downloading file from {url}")
        try:
            request.urlretrieve(url, file_path)
        except Exception as e:
            raise RuntimeError(e)
        print("Download complete!")

    with open(file_path, "rb") as f:
        encoder_params = pkl.load(f)
        param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
        print(
            f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
        )

    new_params = agent.state.params.unfreeze()
    for image_key in image_keys:
        for k in new_params["modules_actor"]["encoder"][f"encoder_{image_key}"]:
            if k in encoder_params:
                new_params["modules_actor"]["encoder"][f"encoder_{image_key}"][
                    k
                ] = encoder_params[k]
                print(f"replaced {k} in encoder_{image_key}")
    from flax.core.frozen_dict import freeze

    new_params = freeze(new_params)
    agent = agent.replace(state=agent.state.replace(params=new_params))


def make_trainer_config():
    return TrainerConfig(
        port_number=5488, broadcast_port=5489, request_types=["send-stats"]
    )


def make_wandb_logger(
    project: str = "edgeml",
    description: str = "jaxrl_m",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
