import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import threading


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
        self._is_transformed = False
        self.fine_transformed = False
        # 14cm width and 12.5 height for the box
        self.crop_volume = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.07, -0.07, 0.075),
                                                               max_bound=(0.07, 0.07, 0.2))

        t1 = np.eye(4)
        t1[:3, :3] = R.from_euler("xyz", [angle, 0., 0.], degrees=True).as_matrix()
        t1[1, 3] = x_distance / 2.
        self.t1 = t1

        t2 = np.eye(4)
        t2[:3, :3] = R.from_euler("xyz", [-angle, 0., 0.], degrees=True).as_matrix()
        t2[1, 3] = -x_distance / 2.
        self.t2 = t2

    def save_finetuned(self):
        assert self.fine_transformed
        t_finetuned = np.zeros((2, *self.t1.shape))
        t_finetuned[0, ...] = self.t1
        t_finetuned[1, ...] = self.t2
        with open("PointCloudFusionFinetuned.npy", "wb") as f:
            np.save(f, t_finetuned)

    def load_finetuned(self):
        from os.path import exists
        if not exists("PointCloudFusionFinetuned.npy"):
            return False
        with open("PointCloudFusionFinetuned.npy", "rb") as f:
            t_finetuned = np.load(f)
            self.t1 = t_finetuned[0, ...]
            self.t2 = t_finetuned[1, ...]
        self.fine_transformed = True
        print(f"loaded finetuned Point Cloud fusion parameters!")
        return True

    def append(self, pcd: np.ndarray):
        # MASSIVE! speed up if float64 is used, see: https://github.com/isl-org/Open3D/issues/1045
        if self.pcd1.is_empty():
            self.pcd1.points = o3d.utility.Vector3dVector(pcd.astype(np.float64))
        elif self.pcd2.is_empty():
            self.pcd2.points = o3d.utility.Vector3dVector(pcd.astype(np.float64))
        else:
            raise NotImplementedError("3 pointclouds not supported")

    def calibrate_fusion(self):
        assert self.is_complete()
        # rough transform
        if not self._is_transformed:
            self._transform()

        # then calibrate
        t = finetune_pointcloud_fusion(pcd1=self.pcd1, pcd2=self.pcd2)
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
        self.pcd1.clear()
        self.pcd2.clear()
        self._is_transformed = False

    def _transform(self):
        self.pcd1.transform(self.t1)
        self.pcd2.transform(self.t2)
        self._is_transformed = True

    def fuse_pointclouds(self, voxelize=False):
        if not self._is_transformed:
            self._transform()

        self.pcd1 += self.pcd2
        mm = 1e-3
        if voxelize:
            return o3d.geometry.VoxelGrid.create_from_point_cloud(input=self.pcd1.crop(self.crop_volume), voxel_size=1 * mm)
        else:
            return self.pcd1.crop(self.crop_volume)

    def get_first(self, cropped=True):
        self.pcd1.transform(self.t1)
        if cropped:
            return self.pcd1.crop(self.crop_volume)
        else:
            return self.pcd1

    def get_both(self):
        return self.pcd1.__copy__(), self.pcd2.__copy__()

    def is_complete(self):
        return not self.pcd1.is_empty() and not self.pcd2.is_empty()

    def is_empty(self):
        return self.pcd1.is_empty() and self.pcd2.is_empty()


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

    def calibrate(self):
        print(f"calibrating for {len(self.pc_backlog)} samples...")
        for i, (pc1, pc2) in enumerate(self.pc_backlog):
            self.pc_fusion.clear()
            self.pc_fusion.append(pc1)
            self.pc_fusion.append(pc2)

            self.samples[i, ...] = self.pc_fusion.calibrate_fusion()

        rotations = R.from_matrix(self.samples[:, :3, :3])
        mean_rot = rotations.mean().as_matrix()
        translation = np.mean(self.samples[:, :3, 3], axis=0)

        final = np.eye(4)
        final[:3, :3] = mean_rot
        final[:3, 3] = translation
        print(f"calibration result: {final}")
        self.pc_fusion.set_fine_tuned_transformation(final)
