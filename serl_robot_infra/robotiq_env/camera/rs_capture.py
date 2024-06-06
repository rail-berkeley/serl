import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, rgb=True, depth=False):
        self.name = name
        # print(self.get_device_serial_numbers())
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.rgb = rgb
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.camera_intrinsics = None

        assert self.rgb or self.depth
        if self.rgb:
            self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)

        self.profile = self.pipe.start(self.cfg)

        if self.depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            self.max_clipping_distance = 1. / depth_scale        # 1m max clipping distance
            self.min_clipping_distance = 0.0 / depth_scale       # might mess things up

            # get the camera intrinsics
            profile = self.pipe.get_active_profile()
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            self.camera_intrinsics = depth_profile.get_intrinsics()

        # for some weird reason, these values have to be set in order for the image to appear with good lightning
        # for firmware >5.13, auto_exposure False & auto_white_balance True, below only auto_exposure True
        for sensor in self.profile.get_device().query_sensors():
            sensor.set_option(rs.option.enable_auto_exposure, True)
            # sensor.set_option(rs.option.enable_auto_white_balance, True)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        image, depth = None, None

        if self.rgb:
            color_frame = aligned_frames.get_color_frame()

            if color_frame.is_video_frame():
                image = np.asarray(color_frame.get_data())

        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

            if depth_frame.is_depth_frame():
                depth = np.asanyarray(depth_frame.get_data())
                # clip max
                depth = np.where((depth > self.max_clipping_distance) | (depth < self.min_clipping_distance), 0., self.max_clipping_distance - depth)
                # clip min
                # scale = self.max_clipping_distance / (self.max_clipping_distance - self.min_clipping_distance)
                # depth *= scale

                depth = (depth * (256. / self.max_clipping_distance)).astype(np.uint8)
                depth = depth[..., None]

        if isinstance(image, np.ndarray) and isinstance(depth, np.ndarray):
            return True, np.concatenate((image, depth), axis=-1)
        elif isinstance(image, np.ndarray):
            return True, image
        elif isinstance(depth, np.ndarray):
            return True, depth  # maybe invert depth and convert it to uint8 (maybe also filter for far away objects)
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
