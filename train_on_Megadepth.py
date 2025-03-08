# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import signal
import argparse
import logging
from pathlib import Path
from cotracker.utils.train_utils import (
    sig_handler,
    term_handler,
)
from trainer import Lite

if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="cotracker_three", help="model name")
    parser.add_argument("--restore_ckpt", help="path to restore a checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="batch size used during training."
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=6, help="number of dataloader workers"
    )
    parser.add_argument("--window_length", type=int, default=6)
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="max learning rate.")
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200000, help="length of training schedule."
    )
    parser.add_argument("--scene_info_dir",
                        default="/data1/local_userdata/wangzhiwei/datasets/MegaDepth_v1/multiview_matching_indices/scenes/",
                        help="path to save checkpoints")
    parser.add_argument("--dataset_root",
                        default="/data1/local_userdata/wangzhiwei/datasets/MegaDepth_v1/",
                        help="path to save checkpoints")
    parser.add_argument("--train_val_list_path",
                        default="/data1/local_userdata/wangzhiwei/datasets/MegaDepth_v1/multiview_matching_indices/train_val_list/",
                        help="path to save checkpoints")
    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--save_loss_every_n_step",
        type=int,
        default=500,
        help="evaluate during training after every n epochs, after every epoch by default",
    )

    parser.add_argument(
        "--save_every_n_epoch",
        type=int,
        default=1,
        help="save checkpoints during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--validate_at_start",
        action="store_true",
        help="whether to run evaluation before training starts",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="frequency of trajectory visualization during training",
    )
    parser.add_argument(
        "--traj_per_sample",
        type=int,
        default=768,
        help="the number of trajectories to sample for training",
    )


    parser.add_argument(
        "--train_iters",
        type=int,
        default=4,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--sequence_len", type=int, default=8, help="train sequence length"
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["tapvid_davis_first"],
        help="what datasets to use for evaluation",
    )
    parser.add_argument(
        "--num_virtual_tracks",
        type=int,
        default=None,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--dont_use_augs",
        action="store_true",
        help="don't apply augmentations during training",
    )
    parser.add_argument(
        "--sample_vis_1st_frame",
        action="store_true",
        help="only sample trajectories with points visible on the first frame",
    )
    parser.add_argument(
        "--sliding_window_len",
        type=int,
        default=8,
        help="length of the CoTracker sliding window",
    )
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--img_resize",
        type=int,
        nargs="+",
        default=[384, 512],
        help="crop videos to this resolution during training",
    )
    parser.add_argument(
        "--eval_max_seq_len",
        type=int,
        default=1000,
        help="maximum length of evaluation videos",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="saves launch time for faster debug",
    )
    parser.add_argument(
        "--random_frame_rate",
        action="store_true",
        help="random_frame_rate",
    )
    parser.add_argument(
        "--real_data_splits",
        type=int,
        nargs="+",
        default=[0],
        help="real data folders",
    )
    parser.add_argument(
        "--loss_only_for_visible_pts",
        action="store_true",
        help="compute sequence loss only for visible points",
    )
    parser.add_argument(
        "--real_data_filter_sift",
        action="store_true",
        help="select point to track based on SIFT features",
    )
    parser.add_argument(
        "--train_grid_size",
        type=int,
        default=5,
        help="number of extra regular grid points that we sample at training. This number will be squared",
    )
    parser.add_argument(
        "--train_sift_size",
        type=int,
        default=0,
        help="number of extra SIFT points that we sample at training.",
    )
    parser.add_argument(
        "--assist_model_path",
        type=str,
        default='/home/wangzhiwei/depth_estimation/dust3r_1228/dust3r/checkpoint/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
        help="path to assist model ckpt",
    )
    parser.add_argument(
        "--real_data_filter_superpoint",
        action="store_true",
        help="select point to track based on SuperPoint features",
    )
    parser.add_argument(
        "--train_only_visible_points",
        action="store_true",
        help="Loss only for visible points",
    )
    parser.add_argument(
        "--offline_model",
        action="store_true",
        help="training the offline model",
    )
    parser.add_argument(
        "--clean_kubric",
        action="store_true",
        help="filtering out bad tracks in Kubric",
    )
    parser.add_argument(
        "--random_number_traj",
        action="store_true",
        help="when training on Kubric, sampling a random number \
            of tracks between 1 and args.traj_per_sample",
    )
    parser.add_argument(
        "--random_seq_len",
        action="store_true",
        help="when training on Kubric, cropping the sequence \
            to have a length between 10 and args.sequence_len frames",
    )
    parser.add_argument(
        "--uniform_query_sampling_method",
        action="store_true",
        help="Whether to sample points uniformly across time. Kubric training only",
    )
    parser.add_argument(
        "--limit_samples", type=int, default=10000, help="limit samples on real data"
    )
    parser.add_argument(
        "--max_queries", type=int, default=1000, help="max query points for training"
    )
    parser.add_argument(
        "--introduce_teacher", action="store_true", 
        help="whether introduce teacher network to training",
    )


    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=False),
        devices="auto",
        accelerator="gpu",
        precision="bf16" if args.mixed_precision else 32,
        num_nodes=args.num_nodes,
        # precision=32,
    ).run(args)
