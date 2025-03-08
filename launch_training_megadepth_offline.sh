#!/bin/bash

EXP_DIR=/home/wangzhiwei/depth_estimation/co-tracker/trainer_save
EXP_NAME=Megadepth_offline
DATE=3-3
RESTORE=/home/wangzhiwei/depth_estimation/co-tracker/cotracker/checkpoints/scaled_offline.pth

echo conda environment `which python`
# --restore_ckpt ${RESTORE}
# mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/;
# mkdir ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3;
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node = 4 --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port=29500 \
    train_on_Megadepth.py --batch_size 12 \
--ckpt_path ${EXP_DIR}/${DATE}_${EXP_NAME} --model_name cotracker_three  \
--save_freq 200 --sequence_len 6 --eval_datasets tapvid_davis_first tapvid_stacking \
--traj_per_sample 512 --sliding_window_len 60 --window_length 6 --img_resize 384 512 \
--save_every_n_epoch 5 --evaluate_every_n_epoch 10 --model_stride 4 --num_nodes 2 \
--num_virtual_tracks 64 --mixed_precision --offline_model --random_frame_rate \
--wdecay 0.0005 --random_seq_len --validate_at_start --restore_ckpt ${RESTORE} \
--max_queries 1000