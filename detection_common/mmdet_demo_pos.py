import os

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv
import torch

import time
import pyrealsense2 as rs
import numpy as np
import cv2
from realsense_func import RealSense
import argparse
from utils import calculate_center_point

parser = argparse.ArgumentParser(description='mmdetection realsense video detection')
parser.add_argument('config', help='config file path')
parser.add_argument('checkpoint', help='checkpoint file')
parser.add_argument('--out', type=str, help='output video path')
parser.add_argument('--score_thr', type=float, default=0.9, help='bbox score threshold')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(args.config, args.checkpoint, device=device)

cam = RealSense(fps=30, bgrx=1280, bgry=720, depx=1280, depy=720)
video_path = f'./video.mp4'
video_depth_path = f'./video_depth.mp4'
video_depthcolor_path = f'./video_depthcolor.mp4'
video_depthcolor_camera_path = f'./video_depthcolor.mp4'
# init parameters
fps, w, h = 30, 1280, 720
mp4 = cv2.VideoWriter_fourcc(*'mp4v')
wr = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True) #
wr_depth = cv2.VideoWriter(video_depth_path, mp4, fps, (w, h), isColor=False)
wr_depthcolor = cv2.VideoWriter(video_depthcolor_path, mp4, fps, (w, h), isColor=True)
wr_camera_colordepth = cv2.VideoWriter(video_depthcolor_camera_path, mp4, fps, (w, h), isColor=True)

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
wait_time = 1
all_lables = ['tofu', 'cans', 'mushroom', 'shrimp', 'sushi',
          'banana', 'pork', 'papercup', 'bread', 'chickenbreast',
          'salmon', 'strawberry', 'fishes', 'mango', 'tomato',
          'orange', 'kiwis', 'egg', 'bakso', 'cashew']
target = 'cans'
target_id = all_lables.index(target) # for testing in real world





while True:
    _, _, color_image, _, aligned_depth_frame = cam.get_aligned_images()
    result = inference_detector(model, color_image)
    visualizer.add_datasample(
        'result',
        color_image,
        data_sample=result,
        draw_gt=False,
        # wait_time=0,
        show=False,
        pred_score_thr=args.score_thr,
    )
    frame = visualizer.get_image()
    # print(result.to_dict()['pred_instances'])
    print(result.to_dict()['pred_instances']['labels'])
    print(result.to_dict()['pred_instances']['scores'])
    print(result.to_dict()['pred_instances']['bboxes'])

    # print(result.pred_instances)
    print("----------------")
    mmcv.imshow(frame, 'bbox video', wait_time=wait_time)
    _labels = result.to_dict()['pred_instances']['labels']
    _scores = result.to_dict()['pred_instances']['scores']
    _bboxes = result.to_dict()['pred_instances']['bboxes']
    _index = torch.where(_scores > args.score_thr)
    # print(_index)
    # !the scores has one dim, so the _index has one dim
    for i in range(len(_index[0])):
        if _labels[_index[0][i]] == target_id:
            target_bbox = _bboxes[_index[0][i]]
            target_mask = _bboxes[_index[0][i]]
            bbox_center, mask_center = calculate_center_point(target_bbox, target_mask)
            bbox_depth_coordinate = cam.get_point_coodinate(aligned_depth_frame, bbox_center)
            # mask_depth_coordinate = cam.get_point_coodinate(aligned_depth_frame, mask_center)
            print(bbox_depth_coordinate)
        else:
            pass
    #--------------


cv2.destroyAllWindows()
wr.release()
# wr_camera_colordepth.release()
cam.release()

