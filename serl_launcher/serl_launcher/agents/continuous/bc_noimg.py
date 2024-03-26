from functools import partial
from typing import Any, Iterable, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from serl_launcher.common.common import JaxRLTrainState, ModuleDict
from serl_launcher.common.typing import Batch, PRNGKey, Data
from serl_launcher.networks.actor_critic_nets import Policy
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop

"""
Behavioral cloning without using image data as observation, only state data
"""


class BCAgentNoImg(flax.struct.PyTreeNode):
    state: JaxRLTrainState

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        # rng = self.state.rng
        # rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        # obs = self.data_augmentation_fn(obs_rng, batch["observations"])
        # batch = batch.copy(add_or_replace={"observations": obs})

        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn(
                {"params": params},
                batch["observations"],
                temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            return actor_loss, {
                "actor_loss": actor_loss,
                "mse": mse.mean(),
                "log_probs": log_probs,
                "pi_actions": pi_actions,
                "mean_std": actor_std.mean(),
                "max_std": actor_std.max(),
            }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")     # no img data present
    def sample_actions(
            self,
            observations: np.ndarray,
            *,
            seed: Optional[PRNGKey] = None,
            temperature: float = 1.0,
            argmax=False,
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            temperature=temperature,
            name="actor",
        )
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        return {
            "mse": mse,
            "log_probs": log_probs,
            "pi_actions": pi_actions,
        }

    @classmethod
    def create(
            cls,
            rng: PRNGKey,
            observations: Data,
            actions: jnp.ndarray,
            # Model architecture
            network_kwargs: dict = dict(hidden_dims=[256, 256]),
            policy_kwargs: dict = dict(tanh_squash_distribution=False),
            # Optimizer
            learning_rate: float = 3e-4,
    ):
        """
        create agent with no encoders
        """

        network_kwargs["activate_final"] = True
        networks = {
            "actor": Policy(
                None,
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs,
            )
        }

        model_def = ModuleDict(networks)

        tx = optax.adam(learning_rate)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, actor=[observations])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        agent = cls(state)
        return agent
