from scipy.spatial.transform import Rotation as R
import numpy as np

def euler_to_quat(euler_angles_abc, degrees = True):
    q = R.from_euler('xyz', [euler_angles_abc[2], euler_angles_abc[1], euler_angles_abc[0]], degrees=degrees).as_quat()

    return q


def euler_to_rotmat(euler_angles_abc, degrees = True):
    rot_mat = R.from_euler('xyz', [euler_angles_abc[2], euler_angles_abc[1], euler_angles_abc[0]], degrees=degrees).as_matrix()

    return rot_mat


def rotmat_to_quat(rotmat):

    quat = R.from_matrix(rotmat).as_quat()

    return quat


def quat_to_rotmat(quat):

    rotmat = R.from_quat(quat).as_matrix()

    return rotmat

def quat_to_euler(quat):

    euler = R.from_quat(quat).as_euler("xyz")
    return euler


def convert_wrench_to_numpy(msg):
    force = msg.wrench.force
    torque = msg.wrench.torque
    return np.array([force.x, force.y, force.z, torque.x, torque.y, torque.z])

def xyzabc_to_se3(xyzabc, degrees = True):

    tranformation_mat = np.eye(4)
    tranformation_mat[0,3] = xyzabc[0]
    tranformation_mat[1,3] = xyzabc[1]
    if(len(xyzabc) == 6):
        tranformation_mat[2,3] = xyzabc[2]
        tranformation_mat[:3,:3] = euler_to_rotmat(xyzabc[3:], degrees)
    else:
        tranformation_mat[:3,:3] = euler_to_rotmat(xyzabc[2:], degrees)

    return tranformation_mat