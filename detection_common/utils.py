import torch

def calculate_center_point(bbox, mask):
    xmin_box = bbox[0]
    ymin_box = bbox[1]
    xmax_box = bbox[2]
    ymax_box = bbox[3]
    bbox_center_x = (xmax_box + xmin_box) / 2
    bbox_center_y = (ymax_box + ymin_box) / 2
    bbox_center = torch.tensor([bbox_center_x, bbox_center_y])
    _int_bbox_centor = torch.floor(bbox_center)
    return _int_bbox_centor, 0