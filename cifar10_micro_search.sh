#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
  --data_path="data/cifar10" \
  --output_dir="outputs" \
  --batch_size=16 \
  --num_epochs=310 \
  --log_every=50 \
  --child_num_epochs=3 \
  --child_num_layers=5 \
  --child_out_filters=256 \
  --child_num_cells=5 \
  --child_keep_prob=0.90 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr=0.001 \
  --child_lr_dec_rate=0.999995 \
  --child_num_gpu=8 \
  --child_key_size=16 \
  --child_value_size=128 \
  --child_num_logistics=10 \
  --child_polyak_decay=0.9995 \
  --child_random_seed=1 \
  --child_num_conv=8 \
  --child_num_act=6 \
  --child_num_comb=3 \
  --child_log_interval=1 \
  --controller_entropy_weight=0.0001 \
  --controller_train_steps=2000 \
  --controller_lr=0.00035 \
  --controller_tanh_constant=1.10 \
  --controller_op_tanh_reduce=2.5 \
  "$@"

