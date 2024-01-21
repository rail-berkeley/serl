import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PCBEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    TARGET_POSE = np.array(
        [
            0.5668657154487453,
            0.002050321710641817,
            0.05362772570641611,
            3.1279511,
            0.0176617,
            0.0212859,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.04, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD = [0.003, 0.003, 0.001, 0.1, 0.1, 0.1]
    ACTION_SCALE = (0.02, 0.2, 1)
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 9
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2] - 0.005,
            TARGET_POSE[3] - 0.05,
            TARGET_POSE[4] - 0.05,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.05,
            TARGET_POSE[3] + 0.05,
            TARGET_POSE[4] + 0.05,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
