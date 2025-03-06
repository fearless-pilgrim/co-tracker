# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import signal
import socket
import sys
import json

import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from pytorch_lightning.lite import LightningLite

from cotracker.models.core.cotracker.cotracker import CoTracker2
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline

from cotracker.utils.visualizer import Visualizer

from megadepth_build import MultiviewMatcherDataModule
from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.datasets.geometry import dense_grid_spv
from cotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from cotracker.models.core.model_utils import (
    get_query_ponts,
    get_uniformly_sampled_pts,
    get_points_on_a_grid,
    get_sift_sampled_pts,
    get_superpoint_sampled_pts,
)
from cotracker.models.core.cotracker.losses import sequence_loss
from cotracker.models.build_cotracker import build_cotracker
from dependency.mast3r.dust3r.load_att_feature import load_dust3r_model
from megadepth_build import MultiviewMatcherDataModule
from cotracker.utils.train_utils import (
    Logger,
    get_eval_dataloader,
    sig_handler,
    term_handler,
    run_test_eval,
)


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    for name, param in model.named_parameters():
        if "vis_conf_head" in name:
            param.requires_grad = False

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.0,
        cycle_momentum=False,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def forward_batch(batch, model, args, teacher_models, visualizer=None):
    video = batch.video
    trajs_g = batch.track
    vis_g = batch.visibility
    valids = batch.valid
    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device
    failed_sample = False
    image_list = batch.image_list
    assist_model = load_dust3r_model(model_name=args.assist_model_path,
                                     image_list=image_list, batch_size=B,
                                     img_size=args.img_resize, device=device, cat_model="to_origin")
    if args.real_data_filter_sift:
        queries = get_sift_sampled_pts(video, N, T, [H, W], num_sampled_frames=args.window_length, device=device)
        if queries.shape[1] < N:
            logging.warning(
                f"SIFT wasn't able to extract enough features: {queries.shape[1]}"
            )
            failed_sample = True
            queries = get_uniformly_sampled_pts(N, T, [H, W], device=device)
    elif args.real_data_filter_superpoint:
        queries = get_superpoint_sampled_pts(video, N, T, [H, W], device=device)

        if queries.shape[1] < N:
            logging.warning("SuperPoint wasn't able to extract enough features")
            failed_sample = True
            queries = get_uniformly_sampled_pts(N, T, [H, W], device=device)
    else:
        # queries = get_uniformly_sampled_pts(N, T, [H, W], device=device)
        queries, sample_mask = get_query_ponts(video[:,0], max_query_num=args.max_query_num)
    # Inference with additional points sampled on a regular grid usually makes predictions better.
    # So we sample these points and discard them thereafter
    if args.introduce_teacher:
        teacher_model_ind = random.choice(range(len(teacher_models)))

        teacher_model_type, teacher_model = teacher_models[teacher_model_ind]
        uniform_size = grid_size = sift_size = 0
        queries_cat = queries.clone()
        if "online" in teacher_model_type:
            grid_size = args.train_grid_size
            sift_size = args.train_sift_size
            if grid_size > 0:
                xy = get_points_on_a_grid(grid_size, [H, W], device=device)
                xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2)  #
                queries_cat = torch.cat([queries_cat, xy], dim=1)  #

            if sift_size > 0:
                xy = get_sift_sampled_pts(video, sift_size, T, [H, W], device=device)
                if xy.shape[1] == sift_size:
                    queries_cat = torch.cat([queries_cat, xy], dim=1)  #
                else:
                    sift_size = 0
        elif "offline" in teacher_model_type:
            uniform_size = 100
            if uniform_size > 0:
                xy = get_uniformly_sampled_pts(uniform_size, T, [H, W], device=device)
                queries_cat = torch.cat([queries_cat, xy], dim=1)  #
        elif teacher_model_type == "tapir":
            pass
        else:
            raise ValueError(f"Model type {teacher_model_type} doesn't exist")

        if "cotracker_three" in teacher_model_type:
            with torch.no_grad():
                (
                    trajs_g,
                    vis_g,
                    confidence,
                    __,
                ) = teacher_model(video, queries_cat)
        else:
            with torch.no_grad():
                trajs_g, vis_g, *_ = teacher_model(video, queries_cat)
                confidence = torch.ones_like(vis_g)

        # discarding additional points
        if sift_size > 0 or grid_size > 0 or uniform_size > 0:
            trajs_g = trajs_g[:, :, : -(grid_size**2) - sift_size - uniform_size]
            vis_g = vis_g[:, :, : -(grid_size**2) - sift_size - uniform_size]
            confidence = confidence[:, :, : -(grid_size**2) - sift_size - uniform_size]

        vis_g = vis_g > 0.9

    batch.trajectory = trajs_g
    batch.visibility = vis_g
    # visualizer.visualize(
    #     video=batch.video.clone(),
    #     tracks=trajs_g[..., sample_mask, :].clone(),
    #     visibility=batch.visibility[..., sample_mask].clone().unsqueeze(-1),
    #     filename="sample_gt_traj",
    # )

    if args.model_name == "cotracker_three":
        if (
            torch.isnan(queries).any()
            or torch.isnan(trajs_g).any()
            or queries.abs().max() > 1500
        ):
            logging.warning("failed_sample")
            queries = torch.ones_like(queries).to(queries.device).float()
            valids = torch.zeros_like(valids).to(valids.device).float()

        tracks, visibility, confidence, train_data = model(feature_loader=assist_model,
            video=video, queries=queries, iters=args.train_iters, is_train=True
        )
        coord_predictions, vis_predictions, confidence_predicitons, valid_mask = (
            train_data
        )

        if failed_sample:
            valid_mask = torch.zeros_like(vis_g)
            logging.warning("Making mask zero for failed sample")

        vis_gts = []
        invis_gts = []
        traj_gts = []
        valids_gts = []
        if args.offline_model:
            S = T
            seq_len = (S // 2) + 1
        else:
            S = args.sliding_window_len
            seq_len = T
        for ind in range(0, seq_len - S // 2, S // 2):
            vis_gts.append(vis_g[:, ind : ind + S].float())
            invis_gts.append(1 - vis_g[:, ind : ind + S].float())
            traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
            valids_gts.append(valids[:, ind : ind + S] * valid_mask[:, ind : ind + S])

        seq_loss = sequence_loss(
            coord_predictions,
            traj_gts,
            valids_gts,
            vis=vis_gts,
            gamma=0.8,
            add_huber_loss=True,
            loss_only_for_visible=True,
        )

        output = {
            "flow": {"predictions": (tracks[0].detach() * valid_mask[..., None])[0]}
        }
        output["flow"]["loss"] = seq_loss.mean() * 0.05
        output["flow"]["queries"] = queries.clone()
        output["flow"]["query_frame"] = queries[0, :, 0].cpu().int()

        output["visibility"] = {
            "predictions": visibility[0].detach(),
        }
        if not (teacher_model_type == "tapir" or args.train_only_visible_points):
            seq_loss_invisible = sequence_loss(
                coord_predictions,
                traj_gts,
                valids_gts,
                vis=invis_gts,
                gamma=0.8,
                add_huber_loss=False,
                loss_only_for_visible=True,
            )
            output["flow_invisible"] = {"loss": seq_loss_invisible.mean() * 0.01}

        return output
    else:
        predictions, visibility, train_data = model(
            video=video, queries=queries, iters=args.train_iters, is_train=True
        )
        coord_predictions, vis_predictions, valid_mask = train_data

        if failed_sample:
            valid_mask = torch.zeros_like(valid_mask)
            logging.warning("Making mask zero for failed sample")

        vis_gts = []
        traj_gts = []
        valids_gts = []
        delta = 6
        S = args.sliding_window_len
        pred_ind = 0

        for ind in range(0, args.sequence_len - S // 2, S // 2):
            vis_gts.append(vis_g[:, ind : ind + S])
            traj_gts.append(trajs_g[:, ind : ind + S])
            if (
                teacher_model_type == "tapir"
                or teacher_model_type == "online_cotracker_three"
                or args.train_only_visible_points
            ):
                valids_gts.append(
                    valids[:, ind : ind + S]
                    * valid_mask[:, ind : ind + S]
                    * vis_g[:, ind : ind + S]
                    > 0.9
                )
            else:
                valids_gts.append(
                    valids[:, ind : ind + S] * valid_mask[:, ind : ind + S]
                )
            pred_ind += 1

        seq_loss = sequence_loss(
            coord_predictions,
            traj_gts,
            vis_gts,
            valids_gts,
            gamma=0.8,
            loss_only_for_visible_pts=False,
        )

        batch.trajectory = batch.trajectory * valid_mask[..., None]
        output = {"flow": {}}

        output["flow"]["predictions"] = (predictions.detach() * valid_mask[..., None])[
            0
        ]
        output["flow"]["loss"] = seq_loss.mean()
        output["flow"]["query_frame"] = queries[0, :, 0].cpu().int()
        output["visibility"] = {
            "predictions": visibility[0].detach(),
        }
        return output


class Lite(LightningLite):
    def run(self, args):
        def seed_everything(seed: int):
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        
        train_dataset = MultiviewMatcherDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_resize=args.img_resize,
            random_seed=0,max_queries=args.max_queries,
            img_pad=True, pin_memory=True,
            scene_info_dir=args.scene_info_dir,
            dataset_path=args.dataset_root,
            train_val_list_path = args.train_val_list_path,
        )
        train_dataset.setup()

        if self.global_rank == 0:
            eval_dataloaders = train_dataset.val_dataloader()
            final_dataloaders = train_dataset.test_dataloader()

            evaluator = Evaluator(args.ckpt_path)

            visualizer = Visualizer(
                save_dir=args.ckpt_path,
                pad_value=180,
                fps=1,
                show_first_frame=0,
                tracks_leave_trace=0,
            )

        if args.model_name == "cotracker":
            model = CoTracker2(
                stride=args.model_stride,
                window_len=args.sliding_window_len,
                num_virtual_tracks=args.num_virtual_tracks,
                model_resolution=args.img_resize,
            )
        elif args.model_name == "cotracker_three":
            if args.offline_model:
                model = CoTrackerThreeOffline(
                    stride=4,
                    corr_radius=3,
                    window_len=args.window_length,
                    model_resolution=args.img_resize,
                    linear_layer_for_vis_conf=True,
                )
            else:
                model = CoTrackerThreeOnline(
                    stride=4,
                    corr_radius=3,
                    window_len=args.window_length,
                    model_resolution=args.img_resize,
                    linear_layer_for_vis_conf=True,
                )
        else:
            raise ValueError(f"Model {args.model_name} doesn't exist")

        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        model.cuda()
        teacher_models = []
        # from cotracker.datasets import real_dataset

        train_loader = train_dataset.train_dataloader()
        
        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)
        print("LEN TRAIN LOADER", len(train_loader))


        if args.model_name == "cotracker":
            teacher_model_online = (
                build_cotracker(
                    window_len=args.sliding_window_len, checkpoint=args.restore_ckpt
                )
                .cuda()
                .eval()
            )
            teacher_models.append(("online", teacher_model_online))
        elif args.model_name == "cotracker_three":
            # teacher_model_online = (
            #     build_cotracker(
            #         window_len=args.window_length,
            #         offline=False,
            #         checkpoint="/home/wangzhiwei/depth_estimation/co-tracker/cotracker/checkpoints/scaled_offline.pth",
            #         v2=True,
            #     )
            #     .cuda()
            #     .eval()
            # )
            # teacher_models.append(("online", teacher_model_online))
            teacher_models.append(("online", None))
        else:
            raise ValueError(f"Model {args.model_name} doesn't exist")

        online_checkpoint = "./checkpoints/baseline_online.pth"
        if args.model_name == "cotracker_three" and not args.offline_model:
            online_checkpoint = args.restore_ckpt
        # print("online_checkpoint", online_checkpoint)
        # teacher_model_online_cot_three = (
        #     build_cotracker(checkpoint=online_checkpoint, offline=False, window_len=16)
        #     .cuda()
        #     .eval()
        # )
        teacher_models.append(
            ("online_cotracker_three", None)
        )

        # offline_checkpoint = "./checkpoints/baseline_offline.pth"
        # if args.model_name == "cotracker_three" and args.offline_model:
        #     offline_checkpoint = args.restore_ckpt

        # teacher_model_offline_cot_three = (
        #     build_cotracker(checkpoint=offline_checkpoint, offline=True, window_len=60)
        #     .cuda()
        #     .eval()
        # )
        teacher_models.append(
            ("offline_cotracker_three", None)
        )

        teacher_model_tapir = None
        teacher_models.append(("tapir", teacher_model_tapir))


        optimizer, scheduler = fetch_optimizer(args, model)

        total_steps = 0
        if self.global_rank == 0:
            logger = Logger(model, scheduler, args.ckpt_path)

        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")

        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(
                ".pt"
            )
            logging.info("Loading checkpoint...")

            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            model_state_dict = model.state_dict()  # 获取新模型的state_dict
            # 过滤掉不匹配的键
            
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }

            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
            model_state_dict.update(filtered_state_dict)

            model.load_state_dict(model_state_dict)

            logging.info(f"Done loading checkpoint")
        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        # model.cuda()
        model.train()

        save_freq = args.save_freq
        scaler = GradScaler(enabled=False)

        should_keep_training = True
        global_batch_num = 0
        epoch = -1

        # if self.global_rank == 0 and args.validate_at_start:
        #     run_test_eval(
        #         evaluator,
        #         model,
        #         eval_dataloaders,
        #         logger.writer,
        #         total_steps,
        #     )
        #     model.train()
        #     torch.cuda.empty_cache()

        while should_keep_training:
            epoch += 1
            for i_batch, batch in enumerate(tqdm(train_loader)):
                
                # batch, gotit = dense_grid_spv(batch)
                batch, gotit = batch
                if not all(gotit):
                    print("batch is None")
                    continue
                visualizer.visualize(
                    video=batch.video.clone(),
                    tracks=batch.trajectory.clone(),
                    visibility=batch.visibility.clone().unsqueeze(-1),
                    filename="sampler_warp_gt_traj",
                    # writer=logger.writer,
                    step=total_steps,
                )
                breakpoint()
                dataclass_to_cuda_(batch)

                optimizer.zero_grad()

                assert model.training

                output = forward_batch(
                    batch, model, args, teacher_models=teacher_models, visualizer=visualizer
                )

                loss = 0
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]

                if self.global_rank == 0:
                    for k, v in output.items():
                        if "loss" in v:
                            logger.writer.add_scalar(
                                f"live_{k}_loss", v["loss"].item(), total_steps
                            )
                        if "metrics" in v:
                            logger.push(v["metrics"], k)
                    if total_steps % save_freq == save_freq - 1:
                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=batch.trajectory.clone(),
                            visibility=batch.visibility.clone(),
                            filename="train_gt_traj",
                            query_frame=output["flow"]["query_frame"],
                            writer=logger.writer,
                            step=total_steps,
                        )

                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=output["flow"]["predictions"][None],
                            visibility=output["visibility"]["predictions"][None] > 0.6,
                            filename="train_pred_traj",
                            query_frame=output["flow"]["query_frame"],
                            writer=logger.writer,
                            step=total_steps,
                        )

                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    global_batch_num += 1

                self.barrier()

                self.backward(scaler.scale(loss))

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                total_steps += 1
                if self.global_rank == 0:
                    if i_batch >= len(train_loader) - 1:
                        if (epoch + 1) % args.save_every_n_epoch == 0:
                            ckpt_iter = "0" * (6 - len(str(total_steps))) + str(
                                total_steps
                            )
                            save_path = Path(
                                f"{args.ckpt_path}/model_{args.model_name}_{ckpt_iter}.pth"
                            )

                            save_dict = {
                                "model": model.module.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "total_steps": total_steps,
                            }

                            logging.info(f"Saving file {save_path}")
                            self.save(save_dict, save_path)

                        if (epoch + 1) % args.evaluate_every_n_epoch == 0:
                            run_test_eval(
                                evaluator,
                                model,
                                eval_dataloaders,
                                logger.writer,
                                total_steps,
                            )
                            model.train()
                            torch.cuda.empty_cache()

                self.barrier()
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break
        if self.global_rank == 0:
            print("FINISHED TRAINING")

            PATH = f"{args.ckpt_path}/{args.model_name}_final.pth"
            torch.save(model.module.module.state_dict(), PATH)
            run_test_eval(
                evaluator, model, final_dataloaders, logger.writer, total_steps
            )
            logger.close()
