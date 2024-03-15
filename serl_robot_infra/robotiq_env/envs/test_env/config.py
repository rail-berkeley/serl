from robotiq_env.envs.robotiq_env import DefaultEnvConfig
import numpy as np


class TestConfig(DefaultEnvConfig):
    TARGET_POSE: np.ndarray = np.array([-0.23454916629572226, -0.6939331362168063, 0.1548973281273407, 2.9025739504570782, 1.1983948447880342, -0.08076374785512226])
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    RESET_Q = np.array([1.0464833974838257, -0.9013996881297608, 1.3149092833148401, -1.9843775234618128, -1.566035572682516, -0.5265157858477991])
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([1., 1., 1., np.inf, np.inf, np.inf])
    ABS_POSE_LIMIT_LOW = np.array([-1., -1., 0.05, -np.inf, -np.inf, -np.inf])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "172.22.22.3"
    CONTROLLER_HZ = 100
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0      # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
