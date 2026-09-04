"""Regression tests for continuous-control SAC targets."""

import jax.numpy as jnp

from serl_launcher.agents.continuous.sac import _compute_target_q


def test_entropy_backup_is_discounted_and_masked() -> None:
    rewards = jnp.array([1.0, 2.0])
    masks = jnp.array([1.0, 0.0])
    target_next_min_q = jnp.array([4.0, 4.0])
    entropy_term = jnp.array([0.5, 0.5])

    target_q = _compute_target_q(
        rewards,
        masks,
        discount=0.9,
        target_next_min_q=target_next_min_q,
        entropy_term=entropy_term,
    )

    expected = jnp.array([1.0 + 0.9 * (4.0 - 0.5), 2.0])
    assert jnp.allclose(target_q, expected)


def test_target_q_without_entropy_backup_is_unchanged() -> None:
    rewards = jnp.array([1.0, 2.0])
    masks = jnp.array([1.0, 0.0])
    target_next_min_q = jnp.array([4.0, 4.0])

    target_q = _compute_target_q(
        rewards,
        masks,
        discount=0.9,
        target_next_min_q=target_next_min_q,
    )

    expected = jnp.array([1.0 + 0.9 * 4.0, 2.0])
    assert jnp.allclose(target_q, expected)
