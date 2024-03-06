from robotiq_env.envs.robotiq_env import DefaultEnvConfig
import numpy as np


class TestConfig(DefaultEnvConfig):
    TARGET_POSE: np.ndarray = np.array([-0.23454916629572226, -0.6939331362168063, 0.1548973281273407, 2.9025739504570782, 1.1983948447880342, -0.08076374785512226])
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    RESET_POSE = np.array([-0.2, -0.7, 0.133, 2.58, -1.63, -0.03])
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([1., 1., 1., 3., 1., 1.])
    ABS_POSE_LIMIT_LOW = np.array([-1., -1., 0.05, -3., -1., -1.])

    ROBOT_IP: str = "172.22.22.3"
    FORCEMODE_DAMPING: float = 0.1
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_FORCE_TYPE: int = 2
    FORCEMODE_LIMITS = np.array([8., 8., 8., 4., 4., 4.])
    FORCEMODE_SCALING = np.array([1e3, 1e3, 1e3, 5, 5, 5])
