import numpy as np
from scipy.spatial.transform import Rotation as R

"""
Robotiq UR5 represents the orientation in axis angle representation
"""


def rotvec_2_quat(rotvec):
    return R.from_rotvec(rotvec).as_quat()


def quat_2_rotvec(quat):
    return R.from_quat(quat).as_rotvec()
