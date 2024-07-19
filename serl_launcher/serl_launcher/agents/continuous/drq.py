import copy
from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import frozen_dict

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.vision.voxel_grid_encoders import MLPEncoder, VoxNet
from serl_launcher.utils.train_utils import _unpack, concat_batches
from serl_launcher.vision.data_augmentations import (
    batched_random_crop,
    batched_random_shift_voxel,
    batched_random_rot90_action,
    batched_random_rot90_state,
    batched_random_rot90_voxel
)


class DrQAgent(SACAgent):
    @classmethod
    def create(
            cls,
            rng: PRNGKey,
            observations: Data,
            actions: jnp.ndarray,
            # Models
            actor_def: nn.Module,
            critic_def: nn.Module,
            temperature_def: nn.Module,
            # Optimizer
            actor_optimizer_kwargs={
                "learning_rate": 3e-4,  # 3e-4
            },
            critic_optimizer_kwargs={
                "learning_rate": 3e-4,  # 3e-4
            },
            temperature_optimizer_kwargs={
                "learning_rate": 3e-4,
            },
            # Algorithm config
            discount: float = 0.95,
            soft_target_update_rate: float = 0.005,
            target_entropy: Optional[float] = None,
            entropy_per_dim: bool = False,
            backup_entropy: bool = False,
            critic_ensemble_size: int = 2,
            critic_subsample_size: Optional[int] = None,
            image_keys: Iterable[str] = ("image",),
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            # target_entropy = -actions.shape[-1] / 2
            from numpy import prod
            target_entropy = -prod(actions.shape)

        print(f"config: discount: {discount}, target_entropy: {target_entropy}")

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
            ),
        )

    @classmethod
    def create_drq(
            cls,
            rng: PRNGKey,
            observations: Data,
            actions: jnp.ndarray,
            # Model architecture
            encoder_type: str = "small",
            use_proprio: bool = False,
            critic_network_kwargs: dict = {
                "hidden_dims": [256, 256],
            },
            policy_network_kwargs: dict = {
                "hidden_dims": [256, 256],
            },
            policy_kwargs: dict = {
                "tanh_squash_distribution": True,
                "std_parameterization": "uniform",
            },
            encoder_kwargs: dict = {
                "pooling_method": "spatial_learned_embeddings",
                "num_spatial_blocks": 8,
                "bottleneck_dim": 256,
            },
            critic_ensemble_size: int = 2,
            critic_subsample_size: Optional[int] = None,
            temperature_init: float = 1.0,
            image_keys: Iterable[str] = ("image",),
            **kwargs,
    ):
        """
        Create a new pixel-based agent.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder

            encoders = {
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
        elif encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    name=f"encoder_{image_key}",
                    **encoder_kwargs
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )

            use_single_channel = [value for key, value in observations.items() if key != "state"][0].shape[-3:] == (
                128, 128, 1)
            print(f"use single channel only: {use_single_channel}")

            encoders = {
                image_key: PreTrainedResNetEncoder(
                    rng=rng,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                    use_single_channel=use_single_channel,
                    **encoder_kwargs
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained-18":
            # pretrained ResNet18 from pytorch
            from serl_launcher.vision.resnet_v1_18 import resnetv1_18_configs
            from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder

            pretrained_encoder = resnetv1_18_configs["resnetv1-18-frozen"](
                name="pretrained_encoder",
            )

            use_single_channel = [value for key, value in observations.items() if key != "state"][0].shape[-3:] == (
                128, 128, 1)
            print(f"use single channel only: {use_single_channel}")

            encoders = {
                image_key: PreTrainedResNetEncoder(
                    rng=rng,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                    use_single_channel=use_single_channel,
                    **encoder_kwargs
                )
                for image_key in image_keys
            }
        elif encoder_type == "distance-sensor":
            from serl_launcher.vision.range_sensor import RangeSensorEncoder
            # use depth image as range-like sensor
            assert [value for key, value in observations.items() if key != "state"][0].shape[-3:] == (128, 128, 1)
            import numpy as np

            # 3x3 points centered in the middle
            keypoints = [tuple(k) for k in np.stack(np.meshgrid([32, 64, 96], [32, 64, 96])).reshape((-1, 2))]
            keypoint_size = (5, 5)

            encoders = {
                image_key: RangeSensorEncoder(
                    name=f"encoder_{image_key}",
                    keypoints=keypoints,
                    keypoint_size=keypoint_size,
                )
                for image_key in image_keys
            }
        elif encoder_type == "voxel-mlp":  # not used, too many weights...
            encoders = {
                image_key: MLPEncoder(
                    mlp=MLP(
                        hidden_dims=[64],
                        activations=nn.relu,
                        activate_final=True,
                        use_layer_norm=True,
                    ),
                    bottleneck_dim=encoder_kwargs["bottleneck_dim"],
                )
                for image_key in image_keys
            }
        elif encoder_type == "voxnet":
            encoders = {
                image_key: VoxNet(
                    bottleneck_dim=encoder_kwargs["bottleneck_dim"],
                    use_conv_bias=False,
                    final_activation=nn.tanh,
                )
                for image_key in image_keys
            }
        elif encoder_type.lower() == "none":
            encoders = None
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        return agent

    def batch_augmentation_fn(self, observations, next_observations, actions, rng):
        for pixel_key in self.config["image_keys"]:
            if not "pointcloud" in pixel_key:
                continue        # skip if not pointcloud

            # rotation of state, action and voxel grid (use the same rng for all of them, so same rotation)
            # jax.debug.print("before {}  {}  {}", observations["state"][0, 0, :], next_observations["state"][0, 0, :], actions[0, :])
            # jax.debug.print("voxel: \n{}", jnp.mean(observations[pixel_key][0, 0, ...].reshape((5, 10, 5, 10, 40)), axis=(1, 3, 4)))
            observations = observations.copy(
                add_or_replace={
                    "state": batched_random_rot90_state(
                        observations["state"], rng, num_batch_dims=2
                    ),
                    pixel_key: batched_random_rot90_voxel(
                        observations[pixel_key], rng, num_batch_dims=2
                    ),
                }
            )
            next_observations = next_observations.copy(
                add_or_replace={
                    "state": batched_random_rot90_state(
                        next_observations["state"], rng, num_batch_dims=2
                    ),
                    pixel_key: batched_random_rot90_voxel(
                        next_observations[pixel_key], rng, num_batch_dims=2
                    )
                }
            )
            # actions = batched_random_rot90_action(
            #     actions, rng,
            # )     # maybe action are the problem
            # jax.debug.print("after {}  {}  {}\n", observations["state"][0, 0, :], next_observations["state"][0, 0, :], actions[0, :])
            # jax.debug.print("voxel after: \n{}", jnp.mean(observations[pixel_key][0, 0, ...].reshape((5, 10, 5, 10, 40)), axis=(1, 3, 4)))
            return observations, next_observations, actions

    def image_augmentation_fn(self, obs_rng, observations, next_obs_rng, next_observations):
        # TODO make it configurable: see https://github.com/rail-berkeley/serl/pull/67

        for pixel_key in self.config["image_keys"]:
            # pointcloud augmentation
            if "pointcloud" in pixel_key:
                observations = observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_shift_voxel(
                            observations[pixel_key], obs_rng, padding=3, num_batch_dims=2
                        )
                    }
                )
                next_observations = next_observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_shift_voxel(
                            next_observations[pixel_key], next_obs_rng, padding=3, num_batch_dims=2
                        )
                    }
                )

            # image augmentation
            else:
                observations = observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_crop(
                            observations[pixel_key], obs_rng, padding=4, num_batch_dims=2
                        )
                    }
                )
                next_observations = next_observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_crop(
                            next_observations[pixel_key], next_obs_rng, padding=4, num_batch_dims=2
                        )
                    }
                )
        return observations, next_observations

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
            self,
            batch: Batch,
            *,
            utd_ratio: int,
            pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        Fast JITted high-UTD version of `.update`.

        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.

        Batch dimension must be divisible by `utd_ratio`.

        It also performs data augmentation on the observations and next_observations
        before updating the network.
        """
        new_agent = self
        if len(self.config["image_keys"]) and self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng = new_agent.state.rng
        rng, obs_rng, next_obs_rng, rot90_rng = jax.random.split(rng, 4)
        obs, next_obs = self.image_augmentation_fn(
            obs_rng=obs_rng,
            observations=batch["observations"],
            next_obs_rng=next_obs_rng,
            next_observations=batch["next_observations"]
        )
        obs, next_obs, actions = self.batch_augmentation_fn(
            observations=obs,
            next_observations=next_obs,
            actions=batch["actions"],
            rng=rot90_rng
        )
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )

        # TODO implement K=2 and M=2

        new_state = self.state.replace(rng=rng)

        new_agent = self.replace(state=new_state)
        return SACAgent.update_high_utd(
            new_agent, batch, utd_ratio=utd_ratio, pmap_axis=pmap_axis
        )

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_critics(
            self,
            batch: Batch,
            *,
            pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        new_agent = self
        if len(self.config["image_keys"]) and self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        # TODO implement K=2 and M=2

        rng = new_agent.state.rng
        rng, obs_rng, next_obs_rng, rot90_rng = jax.random.split(rng, 4)
        obs, next_obs = self.image_augmentation_fn(
            obs_rng=obs_rng,
            observations=batch["observations"],
            next_obs_rng=next_obs_rng,
            next_observations=batch["next_observations"]
        )
        obs, next_obs, actions = self.batch_augmentation_fn(
            observations=obs,
            next_observations=next_obs,
            actions=batch["actions"],
            rng=rot90_rng
        )

        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
                "actions": actions,
            }
        )

        new_state = self.state.replace(rng=rng)
        new_agent = self.replace(state=new_state)
        new_agent, critic_infos = new_agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"critic"}),
        )
        del critic_infos["actor"]
        del critic_infos["temperature"]

        return new_agent, critic_infos
