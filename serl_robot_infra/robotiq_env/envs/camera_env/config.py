from robotiq_env.envs.robotiq_env import DefaultEnvConfig
import numpy as np


class RobotiqCameraConfig(DefaultEnvConfig):
    RESET_Q = np.array([[1.3502, -1.2897, 1.9304, -2.2098, -1.5661, 1.4027]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_RZ_RANGE = (0.78,)
    ABS_POSE_LIMIT_HIGH = np.array([0.2, -0.4, 0.22, 3.2, 0.18, 3.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.2, -0.7, - 0.006, 2.8, -0.18, -3.2])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "172.22.22.2"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.1, 1., 1., 1.])

    REALSENSE_CAMERAS = {
        "wrist": "218622277164",
        "wrist2": "218622279756"
    }


class RobotiqCameraConfigBox5(DefaultEnvConfig):
    RESET_Q = np.array([
        [1.3776, -1.0603, 1.6296, -2.1462, -1.5704, -0.2019],
        [0.9104, -0.9716, 1.3539, -1.9824, -1.545, -0.662],
        [0.4782, -1.4072, 2.1258, -2.3129, -1.5816, -1.1417],
        [1.2083, -1.656, 2.272, -2.202, -1.5828, -0.4231],
        [-0.0388, -1.754, 2.2969, -2.1271, -1.5423, -1.7011]
    ])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.05, 0.1, 0.22, 3.2, 0.18, 3.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.49, -0.75, -0.006, 2.8, -0.18, -3.2])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "172.22.22.2"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.1, 1., 1., 1.])

    REALSENSE_CAMERAS = {
        "wrist": "218622277164",
        # "shoulder": "218622279756"
    }


class RobotiqCameraConfigFinal(DefaultEnvConfig):
    RESET_Q = np.array([        # TODO new ones and more
        [1.3776, -1.0603, 1.6296, -2.1462, -1.5704, -0.2019],
        [0.9104, -0.9716, 1.3539, -1.9824, -1.545, -0.662],
        [0.4782, -1.4072, 2.1258, -2.3129, -1.5816, -1.1417],
        [1.2083, -1.656, 2.272, -2.202, -1.5828, -0.4231],
        [-0.0388, -1.754, 2.2969, -2.1271, -1.5423, -1.7011]
    ])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.6, 0.1, 0.25, 3.2, 0.18, 3.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.7, -0.85, -0.006, 2.8, -0.18, -3.2])
    ABS_POSE_RANGE_LIMITS = np.array([0.35, 0.85])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "172.22.22.2"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.1, 1., 1., 1.])

    REALSENSE_CAMERAS = {
        "wrist": "218622277164",
        "wrist2": "218622279756"
    }


