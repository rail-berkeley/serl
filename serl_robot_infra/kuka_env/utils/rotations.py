from scipy.spatial.transform import Rotation as R


def quat_2_euler(quat, degrees=False):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("ZYX", degrees=degrees)


def euler_2_quat(xyz_as_abc):
    q = R.from_euler("ZYX", xyz_as_abc).as_quat()
    return q
