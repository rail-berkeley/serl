import numpy as np
from serl_robot_infra.franka_env.envs.franka_env import DefaultEnvConfig


class ExampleEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    WRIST_CAM1_SERIAL: str = "130322274175"
    WRIST_CAM2_SERIAL: str = "127122270572"
    TARGET_POSE = np.array(
        [
            0.5906439143742067,
            0.07771711953459341,
            0.0937835826958042,
            3.1099675,
            0.0146619,
            -0.0078615,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    ACTION_SCALE = np.array([0.02, 0.1, 1])
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - 0.05,
            TARGET_POSE[1] - 0.05,
            0.0,
            TARGET_POSE[3] - 0.05,
            TARGET_POSE[4] - 0.05,
            TARGET_POSE[5] - np.pi / 9,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + 0.01,
            TARGET_POSE[1] + 0.01,
            0.1,
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + np.pi / 9,
        ]
    )
