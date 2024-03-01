# University Gent
# Auther: Yi Liu
# Description: support the detection result for grapsing
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
import torch
import cv2
from .realsense_func import RealSense
import argparse
from .utils import calculate_center_point

class Det_Common:
    def __init__(self, config, checkpoint, out_pth, score_thr=0.85):
        self.config = config # configuration file path
        self.checkpoint = checkpoint # checkpoint file path
        self.out_pth = out_pth # output recording video path
        self.score_thr = score_thr # the detection threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = init_detector(config, checkpoint, device=self.device)
        self.cam = RealSense(fps=30, bgrx=1280, bgry=720, depx=1280, depy=720) # TODO: Set parameters to be easily configurable
        self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
        self.visualizer.dataset_meta = self.model.dataset_meta
        self.wait_time = 1 # waiting time for mmcv.imshow
        self.all_lables = ['tofu', 'cans', 'mushroom', 'shrimp', 'sushi',
                      'banana', 'pork', 'papercup', 'bread', 'chickenbreast',
                      'salmon', 'strawberry', 'fishes', 'mango', 'tomato',
                      'orange', 'kiwis', 'egg', 'bakso', 'cashew'] # TODO: set the list to be easy changeable
        target = 'cans' # for testing
        self.target_id = self.all_lables.index(target)  # for testing in real world
        self.bbox_depth_coordinate = np.zeros(3)

    def record_config(self):
        # Record Video Function
        # get the RGB video, Depth video and their color video
        video_path = f'./video.mp4'
        video_depth_path = f'./video_depth.mp4'
        video_depthcolor_path = f'./video_depthcolor.mp4'
        video_depthcolor_camera_path = f'./video_depthcolor.mp4'
        fps, w, h = 30, 1280, 720
        mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        wr = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)  #
        wr_depth = cv2.VideoWriter(video_depth_path, mp4, fps, (w, h), isColor=False)
        wr_depthcolor = cv2.VideoWriter(video_depthcolor_path, mp4, fps, (w, h), isColor=True)
        wr_camera_colordepth = cv2.VideoWriter(video_depthcolor_camera_path, mp4, fps, (w, h), isColor=True)

    def det_info(self):
        _, _, color_image, _, aligned_depth_frame = self.cam.get_aligned_images()
        # get color image and depth aligned depth frame, other params can be seen in realsense functions
        result = inference_detector(self.model, color_image)
        # use mmdet pkg get the result
        self.visualizer.add_datasample(
            'result',
            color_image,
            data_sample=result,
            draw_gt=False,
            # wait_time=0,
            show=False,
            pred_score_thr=self.score_thr,
        )
        # filter results for precise results
        frame = self.visualizer.get_image()

        print(result.to_dict()['pred_instances']['labels'])
        print(result.to_dict()['pred_instances']['scores'])
        print(result.to_dict()['pred_instances']['bboxes'])
        # print(result.pred_instances)
        print("----------------")
        # show image part
        mmcv.imshow(frame, 'bbox video', wait_time=self.wait_time)
        _labels = result.to_dict()['pred_instances']['labels']
        _scores = result.to_dict()['pred_instances']['scores']
        _bboxes = result.to_dict()['pred_instances']['bboxes']
        _index = torch.where(_scores > self.score_thr)
        # ---------------
        for i in range(len(_index[0])):
            if _labels[_index[0][i]] == self.target_id:
                target_bbox = _bboxes[_index[0][i]]
                target_mask = _bboxes[_index[0][i]]
                bbox_center, mask_center = calculate_center_point(target_bbox, target_mask)
                self.bbox_depth_coordinate = self.cam.get_point_coodinate(aligned_depth_frame, bbox_center)
                # mask_depth_coordinate = cam.get_point_coodinate(aligned_depth_frame, mask_center)
                print(self.bbox_depth_coordinate)
            # else:
            #     pass
        return self.bbox_depth_coordinate # TODO: init the value at the first line of this function

