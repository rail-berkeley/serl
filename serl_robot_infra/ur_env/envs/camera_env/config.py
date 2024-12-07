from ur_env.envs.ur5_env import DefaultEnvConfig
import numpy as np


class UR5CameraConfig(DefaultEnvConfig):
    """Set the configuration for UR5Env."""

    RESET_Q = np.array(
        [  # reset poses in joint space (multiple if preferred)
            [2.6331, -1.5022, 2.1151, -2.183, -1.5664, -0.4762],
            [1.983, -1.2533, 1.9069, -2.2314, -1.5495, 0.4462],
        ]
    )
    RANDOM_RESET = True
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.04,)
    ABS_POSE_LIMIT_HIGH = np.array([0.6, 0.1, 0.25, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.7, -0.85, -0.006, -0.05, -0.05, -0.2])
    ABS_POSE_RANGE_LIMITS = np.array([0.36, 0.83])
    ACTION_SCALE = np.array([0.02, 0.1, 1.0], dtype=np.float32)

    ROBOT_IP: str = "172.22.22.2"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.02
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    REALSENSE_CAMERAS = {"wrist": "218622277164", "wrist_2": "218622279756"}
