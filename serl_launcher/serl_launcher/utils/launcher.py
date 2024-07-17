# !/usr/bin/env python3

import jax
from jax import nn

from typing import Optional
import tensorflow_datasets as tfds

from agentlace.trainer import TrainerConfig
from agentlace.data.tfds import populate_datastore

from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.agents.continuous.vice import VICEAgent
from serl_launcher.agents.continuous.bc_noimg import BCAgentNoImg

from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    ReplayBufferDataStore,
)


##############################################################################


def make_bc_agent(
        seed, sample_obs, sample_action, image_keys=("image",), encoder_type="small"
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": False,
            "hidden_dims": [256, 256],
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
    )


def make_bc_agent_no_img(
        seed, sample_obs, sample_action
):
    return BCAgentNoImg.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": False,
            "hidden_dims": [256, 256],
            # "hidden_dims": [128, 64],
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
    )


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
            # "hidden_dims": [128, 64],         # simpler network
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
            # "hidden_dims": [128, 64],
        },
        temperature_init=1e-2,  # 1e-2
        discount=0.99,  # or try values lower, not lower than 0.95
        backup_entropy=False,
        critic_ensemble_size=10,  # isREDQ with these
        critic_subsample_size=2,
    )


def make_drq_agent(
        seed, sample_obs, sample_action, image_keys=("image",), encoder_type="small"
):
    agent = DrQAgent.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs=dict(
            tanh_squash_distribution=True,
            std_parameterization="exp",
            std_min=1e-5,
            std_max=5,
        ),
        critic_network_kwargs=dict(
            activations=nn.tanh,        # TODO test relu
            use_layer_norm=True,
            hidden_dims=[256, 256],
            dropout_rate=0.1           # TODO test dropout
        ),
        policy_network_kwargs=dict(
            activations=nn.tanh,
            use_layer_norm=True,
            hidden_dims=[256, 256],
            dropout_rate=0.1           # TODO test dropout
        ),
        temperature_init=1e-2,
        discount=0.99,  # 0.99
        backup_entropy=True,  # default: False
        critic_ensemble_size=10,
        critic_subsample_size=2,
        encoder_kwargs=dict(
            # pooling_method="spatial_softmax",        # default "spatial_learned_embeddings"
            bottleneck_dim=64,
            # num_spatial_blocks=8,
            # num_kp=64,
        ),
        actor_optimizer_kwargs={
            "learning_rate": 3e-3,  # 3e-4
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-3,  # 3e-4
        },
    )
    return agent


def make_vice_agent(
        seed,
        sample_obs,
        sample_action,
        sample_vice_obs,
        image_keys=("image",),
        vice_image_keys=("image",),
        encoder_type="small",
):
    agent = VICEAgent.create_vice(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        sample_vice_obs,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        vice_image_keys=vice_image_keys,
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
        vice_network_kwargs={
            "activations": nn.leaky_relu,
            "use_layer_norm": True,
            "hidden_dims": [
                256,
            ],
            "dropout_rate": 0.1,
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.96,  # 0.99
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )
    return agent


def make_trainer_config(port_number: int = 5488, broadcast_port: int = 5489):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
        project: str = "agentlace",
        description: str = "serl_launcher",
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


def make_replay_buffer(
        env,
        capacity: int = 1000000,
        rlds_logger_path: Optional[str] = None,
        type: str = "replay_buffer",
        image_keys: list = [],  # used only type=="memory_efficient_replay_buffer"
        preload_rlds_path: Optional[str] = None,
):
    """
    This is the high-level helper function to
    create a replay buffer for the given environment.

    Args:
    - env: gym or gymasium environment
    - capacity: capacity of the replay buffer
    - rlds_logger_path: path to save RLDS logs
    - type: support only for "replay_buffer" and "memory_efficient_replay_buffer"
    - image_keys: list of image keys, used only "memory_efficient_replay_buffer"
    - preload_rlds_path: path to preloaded RLDS trajectories
    """
    print("shape of observation space and action space")
    print(env.observation_space)
    print(env.action_space)

    # init logger for RLDS
    if rlds_logger_path:
        # from: https://github.com/rail-berkeley/oxe_envlogger
        from oxe_envlogger.rlds_logger import RLDSLogger

        rlds_logger = RLDSLogger(
            observation_space=env.observation_space,
            action_space=env.action_space,
            dataset_name="serl_rlds_dataset",
            directory=rlds_logger_path,
            max_episodes_per_file=5,  # TODO: arbitrary number
        )
    else:
        rlds_logger = None

    if type == "replay_buffer":
        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            rlds_logger=rlds_logger,
        )
    elif type == "memory_efficient_replay_buffer":
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            rlds_logger=rlds_logger,
            image_keys=image_keys,
        )
    else:
        raise ValueError(f"Unsupported replay_buffer_type: {type}")

    if preload_rlds_path:
        print(f" - Preloaded {preload_rlds_path} to replay buffer")
        dataset = tfds.builder_from_directory(preload_rlds_path).as_dataset(split="all")
        populate_datastore(replay_buffer, dataset, type="with_dones")
        print(f" - done populated {len(replay_buffer)} samples to replay buffer")

    return replay_buffer
