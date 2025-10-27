import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PegEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    ROBOT_IP: str = "192.168.10.122"
    REALSENSE_CAMERAS = {
        "wrist_1": "932122060300",
        "wrist_2": "023322060732",
    }
    MESH_FILE_PATH = "/home/rp/SERL/src/examples/async_peg_insert_drq/rect_peg/SERL_rectangle v1.stl"
    TARGET_POSE = np.array(
        [
        0.77049,
        -0.04725,
        0.09370,
        -1.5875515,
        0.009599311,
        -3.13583307
    ]
    )
    # nisara - P2
    RESET_POSE = np.array(
        [
           0.77644,
        -0.04044,
        0.10319,
        -1.5510741,
        0.009599311,
        -3.13583307
        ]
    )
    # RESET JOINT POSITION
    # MAKE SURE IT IS: A1, A2, A4, A3, A5, A6, A7
    # Radians
    ## From the topic:
    # RESET_JOINT_POSITION = np.array(
    #     [
    #         -0.010134220158701582,
    #         1.2094114255907469,
    #         -0.6516944804629864,
    #         -0.14664185220364415,
    #         0.15364255700526772,
    #         1.2824023060585676,
    #         -1.6960934354739192,
    #     ]
    # )
    ## From the pendant:
    RESET_JOINT_POSITION = np.array(
        [
            -0.01012291,
            1.209513,
            -0.65170594,
            -0.146608,
            0.1534144,
            1.2824679,
            -1.696111
        ]
    )
    REWARD_THRESHOLD: np.ndarray = np.array(
        [0.0035, 0.0035, 0.001, 0.174533, 0.174533, 0.174533]
    )
    APPLY_GRIPPER_PENALTY = False
    ACTION_SCALE = np.array([0.005, 0.01, 1]) # action_scale[1] defines the scaling in radians
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.008 # only deviates 8mm from target pose
    RANDOM_RZ_RANGE = np.pi / 6
    USE_GRIPPER: bool = False
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2],
            TARGET_POSE[3] - RANDOM_RZ_RANGE,
            # Least value of B is -10 degrees
            -0.174533,
            # Least value of C is -170 degrees
            -2.96706
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.012,
            TARGET_POSE[3] + RANDOM_RZ_RANGE,
            # Least value of B is +10 degrees
            0.174533,
            # Least value of C is +170 degrees
            2.96706,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 1000,
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
