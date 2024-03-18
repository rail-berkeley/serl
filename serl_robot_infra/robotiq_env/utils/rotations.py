import numpy as np
from scipy.spatial.transform import Rotation as R

"""
Robotiq UR5 represents the orientation in axis angle representation
"""


def rotvec_2_quat(rotvec):
    return R.from_rotvec(rotvec).as_quat()


def quat_2_rotvec(quat):
    return R.from_quat(quat).as_rotvec()


def quat_2_euler(quat):
    return R.from_quat(quat).as_euler('xyz')


def euler_2_quat(euler):
    return R.from_euler(euler).as_quat()


def pose2quat(rotvec_pose) -> np.ndarray:
    return np.concatenate((rotvec_pose[:3], rotvec_2_quat(rotvec_pose[3:])))


def pose2rotvec(quat_pose) -> np.ndarray:
    return np.concatenate((quat_pose[:3], quat_2_rotvec(quat_pose[3:])))
