from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

# Register environments from franka_sim.envs where specific classes are found in the PandaXXX.py scripts
from gym.envs.registration import register

register(
    id="PandaPickCube-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=200,
    kwargs={"image_obs": True},
)
register(
    id="PandaReachCube-v0",
    entry_point="franka_sim.envs:PandaReachCubeGymEnv", 
    max_episode_steps=100,
)
register(
    id="PandaReachSparseCube-v0",
    entry_point="franka_sim.envs:PandaReachSparseCubeGymEnv",
    max_episode_steps=200,
)
    
