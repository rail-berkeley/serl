import numpy as np
import pyrealsense2 as rs


def convert_depth_frame_to_pointcloud(depth_image: np.ndarray, camera_intrinsics: rs.intrinsics):
    """
    Convert the depthmap to a 3D point cloud

    Parameters:
    -----------
    depth_frame (np.ndarray): the depth_frame containing the depth map
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

    Return:
    ----------
    x : array
        The x values of the pointcloud in meters
    y : array
        The y values of the pointcloud in meters
    z : array
        The z values of the pointcloud in meters
    """

    assert len(depth_image.shape) == 2 or depth_image.shape[2] == 1
    [height, width] = depth_image.shape[:2]

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    z = depth_image.flatten() / 1000
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return x, y, z
