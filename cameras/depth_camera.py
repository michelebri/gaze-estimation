import cv2
import pyrealsense2 as rs
import numpy as np

class RGDFrameIterator:
    def __init__(self, bag_file_path):
        self.bag_file_path = bag_file_path
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_image = None
        self.depth_frame = None

        try:
            rs.config.enable_device_from_file(self.config, self.bag_file_path, repeat_playback=False)
            self.config.enable_all_streams()

            # Start the pipeline
            self.pipeline.start(self.config)
            self.profile = self.pipeline.get_active_profile()
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            self.intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        except Exception as e:
            print(e)
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            frames = self.pipeline.wait_for_frames()
            align = rs.align(rs.stream.color)
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            self.depth_frame = frames.get_depth_frame()

            if not color_frame or not self.depth_frame:
                return None, None

            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(self.depth_frame.get_data())

            # Convert color image to RGB
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            self.color_image = color_image

            return color_image,  self.depth_frame 

        except Exception as e:
            print(e)
            raise StopIteration

    def get_3d_point(self, depth_frame, x, y):
        """
        Get the 3D point from depth frame using pixel coordinates
        Args:
            x: pixel x coordinate
            y: pixel y coordinate
        Returns:
            tuple: (x, y, z) coordinates in meters, or None if depth value is invalid
            Note: RealSense SDK automatically returns values in meters
        """
        if depth_frame is None:
            return None
        
        # get_distance() returns depth in meters
        depth_value = depth_frame.get_distance(int(x), int(y))
        if depth_value == 0:
            return None
            
        # rs2_deproject_pixel_to_point returns coordinates in meters
        point3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth_value)
        return point3d