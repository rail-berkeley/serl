import numpy as np
from serl_robot_infra.franka_env.envs.franka_env import DefaultEnvConfig


class ExampleEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    WRIST_CAM1_SERIAL: str = "130322274175"
    WRIST_CAM2_SERIAL: str = "127122270572"
    TARGET_POSE = np.array(
        [
            0.5668657154487453,
            0.002050321710641817,
            0.05462772570641611,
            3.1279511,
            0.0176617,
            0.0212859,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD = [0.005, 0.005, 0.001, 0.1, 0.1, 0.1]
    ACTION_SCALE = (0.02, 0.2, 1)
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - 0.05,
            TARGET_POSE[1] - 0.05,
            0.005,
            TARGET_POSE[3] - 0.05,
            TARGET_POSE[4] - 0.05,
            TARGET_POSE[5] - np.pi / 9,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + 0.05,
            TARGET_POSE[1] + 0.05,
            0.1,
            TARGET_POSE[3] + 0.05,
            TARGET_POSE[4] + 0.05,
            TARGET_POSE[5] + np.pi / 9,
        ]
    )
