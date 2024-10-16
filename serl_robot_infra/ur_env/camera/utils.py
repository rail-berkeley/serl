import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import threading
from typing import Any


def finetune_pointcloud_fusion(pc1: np.ndarray, pc2: np.ndarray):
    pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd1.estimate_normals()
    pcd2.estimate_normals()

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

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        transformation, info = pairwise_registration(pcd1, pcd2, max_correspondence_distance=1e-3)

    r = R.from_matrix(transformation[:3, :3].copy()).as_euler("xyz")
    t = transformation[:3, 3].copy().flatten()
    print(f"fusion result--> r: {r}   t: {t}")
    return transformation


def pointcloud_to_voxel_grid(points: np.ndarray, voxel_size: float, min_bounds: np.ndarray, max_bounds: np.ndarray):
    points_filtered = crop_pointcloud(points, min_bounds=min_bounds, max_bounds=max_bounds)
    dimensions = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)
    voxel_indices = ((points_filtered - min_bounds) / voxel_size).astype(int)

    voxel_grid = np.zeros(dimensions, dtype=np.bool_)
    valid_indices = np.all((voxel_indices >= 0) & (voxel_indices < dimensions), axis=1)
    voxel_grid[voxel_indices[valid_indices, 0], voxel_indices[valid_indices, 1], voxel_indices[valid_indices, 2]] = True
    return voxel_grid, voxel_indices[valid_indices, :].astype(np.uint8)


def crop_pointcloud(points: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray):
    within_bounds = np.all((points >= min_bounds) & (points <= max_bounds), axis=1)
    return points[within_bounds]


def transform_point_cloud(points, transform_matrix):
    if points.shape[1] == 3:
        points = np.hstack([points, np.ones((points.shape[0], 1))])

    transformed_points = np.dot(points, transform_matrix.T)

    if transformed_points.shape[1] == 4:
        transformed_points = transformed_points[:, :3]

    return transformed_points


class PointCloudFusion:
    def __init__(self, angle=30., x_distance=0.195, y_distance=-0.0, voxel_grid_shape=(100, 100, 80)):
        self.pcd1, self.pcd2 = None, None

        # 10cm width and 8cm height for the box
        self.min_bounds = np.array([-0.05, -0.05, 0.075])
        self.max_bounds = np.array([0.05, 0.05, 0.155])

        vox_size = (self.max_bounds - self.min_bounds) / voxel_grid_shape
        assert np.all(np.isclose(vox_size, vox_size[0]))
        self.voxel_size: float = float(vox_size[0])

        self.original_pcds = []
        self._is_transformed = False
        self.fine_transformed = False

        t1 = np.eye(4)
        t1[:3, :3] = R.from_euler("xyz", [angle, 0., 0.], degrees=True).as_matrix()
        t1[1, 3] = x_distance / 2.
        t1[0, 3] = y_distance / 2.
        self.t1 = t1

        t2 = np.eye(4)
        t2[:3, :3] = R.from_euler("xyz", [-angle, 0., 0.], degrees=True).as_matrix()
        t2[1, 3] = -x_distance / 2.
        t2[0, 3] = -y_distance / 2.
        self.t2 = t2

    def save_finetuned(self):
        assert self.fine_transformed
        t_finetuned = np.zeros((2, *self.t1.shape))
        t_finetuned[0, ...] = self.t1
        t_finetuned[1, ...] = self.t2
        with open("PointCloudFusionFinetuned.npy", "wb") as f:
            np.save(f, t_finetuned)

    def get_voxelgrid_shape(self):
        return np.ceil((self.max_bounds - self.min_bounds) / self.voxel_size).astype(int)

    def load_finetuned(self):
        from os.path import exists
        if not exists("/home/nico/real-world-rl/spacemouse_tests/PointCloudFusionFinetuned.npy"):
            return False
        with open("/home/nico/real-world-rl/spacemouse_tests/PointCloudFusionFinetuned.npy", "rb") as f:
            t_finetuned = np.load(f)
            self.t1 = t_finetuned[0, ...]
            self.t2 = t_finetuned[1, ...]
        self.fine_transformed = True
        print(f"loaded finetuned Point Cloud fusion parameters!")
        return True

    def append(self, pcd: np.ndarray):
        if self.pcd1 is None:
            self.original_pcds.append(pcd)
            self.pcd1 = pcd
        elif self.pcd2 is None:
            self.original_pcds.append(pcd)
            self.pcd2 = pcd
        else:
            raise NotImplementedError("3 pointclouds not supported")

    def calibrate_fusion(self):
        assert self.is_complete()
        # rough transform
        if not self._is_transformed:
            self._transform()

        # then calibrate
        t = finetune_pointcloud_fusion(pc1=self.pcd1, pc2=self.pcd2)
        return t

    def set_fine_tuned_transformation(self, transformation):
        assert not self.fine_transformed

        t = transformation.copy()[:3, 3] / 2.  # half the translation
        rot = np.zeros((2, 3, 3))
        rot[0, ...] = transformation[:3, :3]
        rot[1, ...] = np.eye(3)
        r = R.from_matrix(rot).mean()  # half the rotation

        t1_fine = np.eye(4)
        t1_fine[:3, :3] = r.as_matrix()
        t1_fine[:3, 3] = t
        self.t1 = np.dot(self.t1, t1_fine)

        t2_fine = np.eye(4)
        t2_fine[:3, :3] = r.inv().as_matrix()
        t2_fine[:3, 3] = -t
        self.t2 = np.dot(self.t2, t2_fine)

        self.fine_transformed = True

    def clear(self):
        self.pcd1, self.pcd2 = None, None
        self._is_transformed = False
        self.original_pcds = []

    def _transform(self):
        assert not self.is_empty()
        self.pcd1 = transform_point_cloud(points=self.pcd1, transform_matrix=self.t1)
        if self.pcd2 is not None:
            self.pcd2 = transform_point_cloud(points=self.pcd2, transform_matrix=self.t2)
        self._is_transformed = True

    def voxelize(self, points: np.ndarray):
        grid, indices = pointcloud_to_voxel_grid(points, voxel_size=self.voxel_size, min_bounds=self.min_bounds,
                                                 max_bounds=self.max_bounds)
        return grid, indices

    def crop(self, points: np.ndarray):
        return crop_pointcloud(points=points, min_bounds=self.min_bounds, max_bounds=self.max_bounds)

    def get_pointcloud_representation(self, voxelize=True):
        if self.is_complete():
            return self.fuse_pointclouds(voxelize=voxelize)
        elif not self.is_empty():
            return self.get_first(voxelize=voxelize)

    def fuse_pointclouds(self, voxelize=True, cropped=True):
        if not self._is_transformed:
            self._transform()
        swap = lambda x: np.moveaxis(x, 0, 1)
        fused = swap(np.hstack([swap(self.pcd1), swap(self.pcd2)]))
        return self.voxelize(fused) if voxelize else (self.crop(fused) if cropped else fused)

    def get_first(self, voxelize=True):
        if not self._is_transformed:
            self.pcd1 = transform_point_cloud(self.pcd1, transform_matrix=self.t1)
        return self.voxelize(self.pcd1) if voxelize else self.crop(self.pcd1)

    def get_original_pcds(self):
        if len(self.original_pcds) == 1:
            return self.original_pcds[0]
        else:
            return self.original_pcds

    def is_complete(self):
        return self.pcd1 is not None and self.pcd2 is not None

    def is_empty(self):
        return self.pcd1 is None and self.pcd2 is None


class CalibrationTread(threading.Thread):
    def __init__(self, pc_fusion: PointCloudFusion, num_samples=20, verbose=False, *args, **kwargs):
        super(CalibrationTread, self).__init__(*args, **kwargs)
        self.pc_fusion = pc_fusion
        self.samples = np.zeros((num_samples, 4, 4))  # transformation matrix samples
        self.pc_backlog = []
        self.verbose = verbose

    def start(self):
        super().start()
        if self.verbose:
            print(f"Calibration Thread started at {self.native_id}")

    def append_backlog(self, pc1, pc2):
        self.pc_backlog.append([pc1, pc2])
        assert self.samples.shape[0] >= len(self.pc_backlog)

    def calibrate(self, visualize=False):
        print(f"calibrating for {len(self.pc_backlog)} samples...")
        for i, (pc1, pc2) in enumerate(self.pc_backlog):
            self.pc_fusion.clear()
            self.pc_fusion.append(pc1)
            self.pc_fusion.append(pc2)

            self.samples[i, ...] = self.pc_fusion.calibrate_fusion()

            if visualize:
                # visualize for testing
                pc = self.pc_fusion.pcd1.copy()
                pc2 = self.pc_fusion.pcd2.copy()
                pc = transform_point_cloud(points=pc, transform_matrix=self.samples[i])  # transform

                swap = lambda x: np.moveaxis(x, 0, 1)
                fused = swap(np.hstack([swap(pc), swap(pc2)]))

                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(fused)
                o3d.visualization.draw_geometries([pc])

        rotations = R.from_matrix(self.samples[:, :3, :3])
        mean_rot = rotations.mean().as_matrix()
        translation = np.mean(self.samples[:, :3, 3], axis=0)

        final = np.eye(4)
        final[:3, :3] = mean_rot
        final[:3, 3] = translation
        print(f"calibration result: {final}")
        self.pc_fusion.set_fine_tuned_transformation(final)
