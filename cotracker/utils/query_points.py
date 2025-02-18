import torch

from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.models.extractors.sift import SIFT


def get_query_points(superpoint, sift, query_image, max_query_num=500):
    # Run superpoint and sift on the target frame
    # Feel free to modify for your own

    pred_sp = superpoint({"image": query_image})["keypoints"]
    pred_sift = sift({"image": query_image})["keypoints"]

    query_points = torch.cat([pred_sp, pred_sift], dim=1)
    
    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[:max_query_num]
        query_points = query_points[:, random_point_indices, :]

    return query_points