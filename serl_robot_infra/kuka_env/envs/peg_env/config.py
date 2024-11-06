import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PegEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    ROBOT_IP: str = "192.168.10.122"
    REALSENSE_CAMERAS = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
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
    RESET_POSE = TARGET_POSE + np.array([0.5, 0.1, 0.3, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    APPLY_GRIPPER_PENALTY = False
    ACTION_SCALE = np.array([0.02, 0.1, 1])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 6
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2],
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.1,
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 0.89,
        "rotational_stiffness": 150,
        "rotational_damping": 0.7,
        "translational_Ki": 0,
        "translational_clip_x": 0.003,
        "translational_clip_y": 0.003,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.003,
        "translational_clip_neg_y": 0.003,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 0.89,
        "rotational_stiffness": 300,
        "rotational_damping": 0.9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
