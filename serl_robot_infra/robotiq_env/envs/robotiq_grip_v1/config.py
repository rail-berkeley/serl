from robotiq_env.envs.robotiq_env import DefaultEnvConfig
import numpy as np


class RobotiqCornerConfig(DefaultEnvConfig):
    # RESET_Q = np.array([[1.38228, -1.24648, 1.9504, -2.2732, -1.5645, -0.18799]])     # old one
    RESET_Q = np.array([[1.34231, -1.24585, 1.94961, -2.27267, -1.56428, -0.22641]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.05,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])            # TODO euler rotations suck :/
    ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
    # ABS_POSE_LIMIT_HIGH = np.array([0.0, -0.5, 0.15, 3.2, 0.1, 3.2])            # more conservative for tests
    # ABS_POSE_LIMIT_LOW = np.array([-0.2, -0.6, 0.1, 3.0, -0.1, -3.2])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "172.22.22.2"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000      # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0      # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
