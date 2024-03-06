import gymnasium as gym
import numpy as np
import copy

from robotiq_env.envs.wrappers import SpacemouseIntervention
from pprint import pprint


if __name__ == "__main__":
    env = gym.make("robotiq_test")
    env = SpacemouseIntervention(env)

    obs, _ = env.reset()
    transitions = []

    while True:
        next_obs, rew, done, truncated, info = env.step(action=np.zeros((7,)))
        actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        transitions.append(transition)
        pprint(transition)

        obs = next_obs

        if done:
            break

    env.close()
