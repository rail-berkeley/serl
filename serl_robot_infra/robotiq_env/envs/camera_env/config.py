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
        "wrist_2": "218622279756"
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
    RESET_Q = np.array([
        [2.6259, - 1.5196, 2.1287, - 2.1784, - 1.5665, - 0.4741],
        [2.0131, - 1.271, 1.9316, - 2.2306, - 1.566, 0.4537],
        [1.8937, - 0.8222, 1.239, - 1.9868, - 1.565, 0.3668],
        [1.4173, - 1.6375, 2.2516, - 2.1841, - 1.5668, - 0.1285],
        [1.4715, - 0.8744, 1.2767, - 1.9723, - 1.5653, - 0.0874],
        [1.12, - 0.7634, 1.1184, - 1.9248, - 1.5653, - 0.4334],
        [1.0242, - 1.3104, 2.0986, - 2.358, - 1.5664, - 2.0496],
        [0.7431, - 1.0746, 1.6486, - 2.1441, - 1.5655, - 0.7738],
        [0.4391, - 1.5926, 2.3356, - 2.3129, - 1.5668, - 1.1115],
        [0.1815, - 1.2945, 1.8964, - 2.1719, - 1.5658, - 1.3841],
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
        "wrist_2": "218622279756"
    }
