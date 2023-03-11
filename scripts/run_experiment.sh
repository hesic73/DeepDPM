#!/bin/bash

PROJECT=DeepDPM_CNG
EXP_NAME=deep_init_k_5
DATASET=CNG

MINIMAL_ARGS="--dataset "$DATASET" \
--seed 12345 \
--project "$PROJECT" \
--exp_name "$EXP_NAME" \
--use_labels_for_eval \
--save_checkpoints \
--max_epochs 500 \
--batch-size 512 \
--gpus 1,2,3"


EXPERIMENT_ARGS=" --init_k 5 \
--clusternet_hidden_layer_list 128 256 512 512 256 128 \
--start_computing_params 25 \
--how_to_compute_mu kmeans \
--start_sub_clustering 45 \
--start_splitting 55 \
--start_merging 55 \
--split_merge_every_n_epochs 30"

ARGS=$MINIMAL_ARGS$EXPERIMENT_ARGS

echo $ARGS
export CUDA_VISIBLE_DEVICES=0,1,2,3
python DeepDPM.py $ARGS
