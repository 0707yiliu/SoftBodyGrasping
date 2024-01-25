import pyrealsense2 as rs
import numpy as np
import cv2
import os

class RealSense:
    # mainly use get_aligned_images() function to get color and depth information of the camera
    def __init__(self, fps, bgrx, bgry, depx, depy):
        self.pipeline = rs.pipeline()  # define pipline
        self.bgrx = bgrx
        self.bgry = bgry
        self.fps = fps
        config = rs.config()  # define configuration
        config.enable_stream(rs.stream.depth, depx, depy, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, bgrx, bgry, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)
        align_to = rs.stream.color  # algin color stream
        self.align = rs.align(align_to)
        _, _, _, _, _= self.get_aligned_images() # init the first frame

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames() #waiting for getting the image frame
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                             'ppx': intr.ppx, 'ppy': intr.ppy,
                             'height': intr.height, 'width': intr.width,
                             'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
                             }

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))
        color_image = np.asanyarray(color_frame.get_data())
        return intr, self.depth_intrin, color_image, depth_image, aligned_depth_frame

    # def get_info(self, x=320, y=320):
    #     intr, depth_intrin, rgb, depth, aligned_depth_frame = self.get_aligned_images()
    #     # dis = aligned_depth_frame.get_distance(x, y)
    #     # camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)
    #     # return camera_coordinate # reture x,y,z
    def get_point_coodinate(self, aligned_depth_frame, dim):
        x = dim[0]
        y = dim[1]
        dis = aligned_depth_frame.get_distance(x, y)
        camera_coordinate = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x, y], dis)
        return camera_coordinate

    def release(self):
        self.pipeline.stop()





