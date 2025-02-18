# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import glob
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
from lightglue import SuperPoint, SIFT 
from dependency.mast3r.dust3r.load_att_feature import load_dust3r_model

def get_query_points(superpoint, sift, query_image, max_query_num=100):
    # Run superpoint and sift on the target frame
    # Feel free to modify for your own
    
    pred_sp = superpoint({"image": query_image})["keypoints"]
    pred_sift = sift({"image": query_image})["keypoints"]
    # breakpoint()
    # num_pixel = query_image.shape[-1] * query_image.shape[-2]
    query_points = torch.cat([pred_sp, pred_sift], dim=1)
    
    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[:max_query_num]
        query_points = query_points[:, random_point_indices, :]

    return query_points


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        # default="/data1/local_userdata/wangzhiwei/datasets/KITTI/sequence/11/image_2/",
        default="/data1/local_userdata/wangzhiwei/datasets/Cambridge/GreatCourt/seq2/",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default="/home/wangzhiwei/depth_estimation/co-tracker/cotracker/checkpoints/scaled_offline.pth",
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        # action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )

    args = parser.parse_args()

    # load the input video frame by frame
    video_list = glob.glob(args.video_path + "/*.png")
    video_list.sort()
    img_lists = []
    for i, frame in enumerate(video_list):
        if i % 1 == 0:
            img_lists.append(np.array(Image.open(frame)))
        if i == 15:
            break
    frames = np.stack(img_lists)
    # video = read_video_from_path(np.array(video_list))
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    segm_mask = torch.from_numpy(segm_mask)[None, None]
    superpoint = SuperPoint().cuda().eval()
    sift = SIFT().cuda().eval()
    features_loader = load_dust3r_model(image_list=video_list[:16], model_name='/home/wangzhiwei/depth_estimation/dust3r_1228/dust3r/checkpoint/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    # features_loader = load_dust3r_model(model_name='/home/wangzhiwei/depth_estimation/dust3r_1228/dust3r/checkpoint/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:

            window_len = 60

            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                # offline=args.offline,
                offline=True,
                window_len=window_len,
                feature_loader=features_loader,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)
    query_points = get_query_points(superpoint, sift, video[:, 0])
    
    pred_tracks, pred_visibility = model(
        video,
        queries=query_points,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        backward_tracking=args.backward_tracking,
        # segm_mask=segm_mask
    )
    print("computed")
    # save a video with predicted tracks
    # breakpoint()
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0 if args.backward_tracking else args.grid_query_frame,
    )

