import copy
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Iterable, Dict
from functools import partial
from flax.core import frozen_dict
import flax.linen as nn
from collections import OrderedDict
import optax

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.networks.classifier import BinaryClassifier

from serl_launcher.utils.train_utils import _unpack, concat_batches


class VICEAgent(DrQAgent):
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
        vice_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        vice_optimizer_kwargs={
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
            "vice": vice_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
            "vice": make_optimizer(**vice_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
            vice=[observations],
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
            target_entropy = -actions.shape[-1] / 2

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
    def create_vice(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "small",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        vice_network_kwargs: dict = {
            "hidden_dims": [
                256,
            ],
            "activations": nn.leaky_relu,
            "use_layer_norm": True,
            "dropout_rate": 0.1,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True
        vice_network_kwargs["activate_final"] = True

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
            vice_encoders = {
                image_key: SmallEncoder(
                    features=(32, 64, 128, 256),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 2, 2),
                    padding="VALID",
                    pool_method="avg",
                    bottleneck_dim=256,
                    spatial_block_size=8,
                    name=f"vice_encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pre_trained_frozen=False,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
            vice_encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pre_trained_frozen=False,
                    name=f"vice_encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                resnetv1_configs,
                PreTrainedResNetEncoder,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    freeze_encoder=True,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
            vice_encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=512,
                    freeze_encoder=True,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
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

        vice_encoder_def = EncodingWrapper(
            encoder=vice_encoders,
            use_proprio=False,
            enable_stacking=True,
            image_keys=image_keys,
        )

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

        vice_def = BinaryClassifier(
            pretrained_encoder=pretrained_encoder,
            encoder=vice_encoder_def,
            network=MLP(**vice_network_kwargs),
            enable_stacking=True,
            name="vice",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            vice_def=vice_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        return agent

    def data_augmentation_fn(self, rng, observations):
        for pixel_key in self.config["image_keys"]:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    def encode_images(
        self,
        images: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        """
        Forward pass for pre-trained encoder network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            images,
            name="vice",
            rngs={"dropout": rng} if train else {},
            return_encoded=True,
            train=train,
        )

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
            "vice": lambda params, rng: (0.0, {}),
        }

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_vice(
        self,
        batch,
        pmap_axis: Optional[str] = None,
    ):
        """
        update the VICE reward classifier using the BCE loss.
        addtional regularization techniques are also used: mixup, label smoothing, and gradient penalty regularization
        to prevent GAN mode collapse.

        NOTE: assumes that the second half of the batch contains the goal images, so labels = 1
        """
        new_agent = self
        rng = new_agent.state.rng
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        def mixup_data_rng(key_0, key_1, x: jnp.ndarray, y: jnp.ndarray, alpha=1):
            """
            performs mixup regularization on the input images and labels
            """
            if alpha > 0:
                lam = jax.random.beta(key_0, alpha, alpha)
            else:
                lam = 1
            batch_size = x.shape[0]
            index = jax.random.permutation(key_1, batch_size)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        observations = batch["next_observations"]
        key, rng = jax.random.split(rng)
        # data augmentation on images
        aug_observations = self.data_augmentation_fn(key, observations)
        mix_observations = observations.unfreeze()
        gp_observations = observations.unfreeze()
        all_info = {}

        for image_key in self.config["image_keys"]:
            pixels = observations[image_key]
            batch_size = pixels.shape[0]

            pixels = observations[image_key][: batch_size // 2]
            aug_pixels = aug_observations[image_key][: batch_size // 2]
            goal_pixels = observations[image_key][batch_size // 2 :]
            aug_goal_pixels = aug_observations[image_key][batch_size // 2 :]

            # concatenate all images for update
            all_obs_pixels = jnp.concatenate([pixels, aug_pixels], axis=0)
            all_goal_pixels = jnp.concatenate([goal_pixels, aug_goal_pixels], axis=0)
            all_pixels = jnp.concatenate([all_goal_pixels, all_obs_pixels], axis=0)

            # create labels
            ones = jnp.ones((batch_size, 1))
            zeros = jnp.zeros((batch_size, 1))
            y_batch = jnp.concatenate([ones, zeros], axis=0)
            y_batch = y_batch.squeeze(-1)

            # label smoothing, help with bad numerical issue with large negative logits
            y_batch = y_batch * (1 - 0.2) + 0.5 * 0.2

            # encode images into embeddings
            key, rng = jax.random.split(rng)
            encoded = self.encode_images(all_pixels, key, train=True)

            # perform mixup
            key_0, key_1, rng = jax.random.split(rng, 3)
            mix_encoded, y_a_0, y_b_0, lam_0 = mixup_data_rng(
                key_0, key_1, encoded, y_batch
            )
            mix_observations[image_key] = mix_encoded

            # interpolate for Gradient Penalty regularization
            key, rng = jax.random.split(rng)
            """
            generate random epsilon for each sample in the batch, the shape here depends on the shape of the encoded embeddings.
            Here are some examples:
            epsilon = jax.random.uniform(key, shape=(all_obs_pixels.shape[0], 1, 1, 1, 1))
            epsilon = jax.random.uniform(key, shape=(all_obs_pixels.shape[0], 1))
            """
            # epsilon = jax.random.uniform(key, shape=(all_obs_pixels.shape[0], 1, 1, 1))
            epsilon = jax.random.uniform(
                key,
                shape=(len(mix_encoded) // 2, *([1] * (len(mix_encoded.shape[1:])))),
            )
            gp_encoded = (
                epsilon * mix_encoded[: len(mix_encoded) // 2]
                + (1 - epsilon) * mix_encoded[len(mix_encoded) // 2 :]
            )
            gp_observations[image_key] = gp_encoded

        # remove all non pixel inputs keys from the batch
        remove_keys = [
            k for k in gp_observations.keys() if k not in self.config["image_keys"]
        ]
        for k in remove_keys:
            gp_observations.pop(k)

        mix_observations = frozen_dict.freeze(mix_observations)
        gp_observations = frozen_dict.freeze(gp_observations)

        key, rng = jax.random.split(rng, 2)

        def mixup_loss_fn(params, rng) -> Tuple[jnp.ndarray, Dict[str, float]]:
            y_hat = new_agent.state.apply_fn(
                {
                    "params": params,
                },
                mix_observations,
                name="vice",
                train=True,
                classify_encoded=True,
                rngs={"dropout": rng},
            )
            bce_loss_a = jnp.mean(optax.sigmoid_binary_cross_entropy(y_hat, y_a_0))
            bce_loss_b = jnp.mean(optax.sigmoid_binary_cross_entropy(y_hat, y_b_0))
            bce_loss = lam_0 * bce_loss_a + (1 - lam_0) * bce_loss_b
            return bce_loss, {
                "bce_loss": bce_loss,
            }

        def gp_loss_fn(params, rng) -> Tuple[jnp.ndarray, Dict[str, float]]:
            bce_loss, info = mixup_loss_fn(params, key)
            helper_fn = lambda x: new_agent.state.apply_fn(
                {
                    "params": params,
                },
                x,
                name="vice",
                train=True,
                classify_encoded=True,
                rngs={"dropout": rng},
            )

            grad_wrt_input = jax.vmap(jax.grad(helper_fn), in_axes=0, out_axes=0)
            gradients = grad_wrt_input(gp_observations)
            gradients = jnp.concatenate([grads for grads in gradients.values()], axis=0)
            gradients = gradients.reshape((gradients.shape[0], -1))
            grad_norms = jnp.sqrt(jnp.sum((gradients**2 + 1e-6), axis=1))
            grad_penalty = jnp.mean((grad_norms - 1) ** 2)

            return bce_loss + 10 * grad_penalty, {
                "bce_loss": bce_loss,
                "grad_norm": grad_norms.mean(),
            }

        loss_fns = {
            "actor": lambda params, rng: (0.0, {}),
            "critic": lambda params, rng: (0.0, {}),
            "temperature": lambda params, rng: (0.0, {}),
            "vice": gp_loss_fn,
        }
        new_state, info = new_agent.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )
        new_state = new_state.replace(rng=rng)
        all_info.update(info)

        return new_agent.replace(state=new_state), all_info

    @partial(jax.jit)
    def vice_reward(self, observation):
        rews = nn.sigmoid(
            self.state.apply_fn(
                {"params": self.state.params},
                observation,
                name="vice",
                train=False,
            )
        )
        return rews

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_critics(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        new_agent = self
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng = new_agent.state.rng
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = self.data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
        rewards = (self.vice_reward(next_obs) >= 0.5) * 1.0
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
                "rewards": rewards,
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
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng = new_agent.state.rng
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = self.data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
        rewards = (self.vice_reward(next_obs) >= 0.5) * 1.0
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
                "rewards": rewards,
            }
        )

        new_state = self.state.replace(rng=rng)

        new_agent = self.replace(state=new_state)
        new_agnet, info = SACAgent.update_high_utd(
            new_agent, batch, utd_ratio=utd_ratio, pmap_axis=pmap_axis
        )
        info["vice_rewards"] = rewards.mean()
        return new_agent, info
