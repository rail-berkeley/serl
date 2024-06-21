import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def pointcloud_2_o3d(pointcloud) -> o3d.geometry.PointCloud:
    if type(pointcloud) == o3d.geometry.PointCloud:
        return pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    return pcd


def finetune_pointcloud_fusion(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud):
    pcd1.estimate_normals()
    pcd2.estimate_normals()

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        transformation, info = pairwise_registration(pcd1, pcd2, max_correspondence_distance=1e-3)

    r = R.from_matrix(transformation[:3, :3].copy()).as_euler("xyz")
    t = transformation[:3, 3].copy().flatten()
    print(f"fusion result--> r: {r}   t: {t}")
    return transformation


def pairwise_registration(source, target, max_correspondence_distance):
    # see https://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html
    icp = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance,
        icp.transformation)
    return transformation_icp, information_icp


class PointCloudFusion:
    def __init__(self, angle=30., x_distance=0.195):
        self.pcd1 = o3d.geometry.PointCloud()
        self.pcd2 = o3d.geometry.PointCloud()
        self.fine_tuned_transformation = np.eye(4)
        self.coarse_transformed = False

        t1 = np.eye(4)
        t1[:3, :3] = R.from_euler("xyz", [angle, 0., 0.], degrees=True).as_matrix()
        t1[1, 3] = x_distance / 2.
        self.t1 = t1

        t2 = np.eye(4)
        t2[:3, :3] = R.from_euler("xyz", [-angle, 0., 0.], degrees=True).as_matrix()
        t2[1, 3] = -x_distance / 2.
        self.t2 = t2

    def append(self, pcd: np.ndarray):
        if self.pcd1.is_empty():
            self.pcd1 = pointcloud_2_o3d(pcd)
        elif self.pcd2.is_empty():
            self.pcd2 = pointcloud_2_o3d(pcd)
        else:
            raise NotImplementedError("3 pointclouds not supported")

    def calibrate_fusion(self):
        assert self.is_complete()
        # rough transform
        if not self.coarse_transformed:
            self._coarse_transform()

        # then calibrate
        t = finetune_pointcloud_fusion(pcd1=self.pcd1, pcd2=self.pcd2)
        return t

    def set_fine_tuned_transformation(self, transformation):
        self.fine_tuned_transformation[...] = transformation

    def clear(self):
        self.pcd1.clear()
        self.pcd2.clear()
        self.coarse_transformed = False

    def _coarse_transform(self):
        self.pcd1.transform(self.t1)
        self.pcd2.transform(self.t2)
        self.coarse_transformed = True

    def fuse_pointclouds(self):
        if not self.coarse_transformed:
            self._coarse_transform()
        self.pcd1.transform(self.fine_tuned_transformation)

        self.pcd1 += self.pcd2
        return self.pcd1

    def get_first(self):
        self.pcd1.transform(self.t1)
        return self.pcd1

    def get_both(self):
        return self.pcd1.__copy__(), self.pcd2.__copy__()

    def get_calibration_history(self):
        return np.array(self.calibration_history)

    def is_complete(self):
        return not self.pcd1.is_empty() and not self.pcd2.is_empty()

    def is_empty(self):
        return self.pcd1.is_empty() and self.pcd2.is_empty()