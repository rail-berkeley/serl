from robotiq_env.envs.robotiq_env import DefaultEnvConfig
import numpy as np


class RobotiqCornerConfig(DefaultEnvConfig):
    TARGET_POSE: np.ndarray = np.array([-0.23454916629572226, -0.6939331362168063, 0.1548973281273407, 2.9025739504570782, 1.1983948447880342, -0.08076374785512226])
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    RESET_Q = np.array([[1.38228, -1.24648, 1.9504, -2.2732, -1.5645, -0.18799]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.14, 0.07, 1., 3.3, 3.3, 3.3])            # TODO euler rotations suck :/
    ABS_POSE_LIMIT_LOW = np.array([-0.45, -0.78, -0.006, -3.3, -3.3, -3.3])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "172.22.22.2"
    CONTROLLER_HZ = 200
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0      # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
