import gymnasium as gym

# Correct method to list environments
for env in gym.envs.registry.keys():
    print(env)
