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
    return R.from_quat(quat).as_euler()


def euler_2_quat(euler):
    return R.from_euler(euler).as_quat()
