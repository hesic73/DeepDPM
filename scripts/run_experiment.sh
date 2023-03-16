#!/bin/bash

PROJECT=DeepDPM_CNG_test
EXP_NAME=init_k_1
DATASET=CNG

MINIMAL_ARGS="--dataset "$DATASET" \
--seed 12345 \
--project "$PROJECT" \
--exp_name "$EXP_NAME" \
--use_labels_for_eval \
--save_checkpoints \
--max_epochs 500 \
--batch-size 128 \
--gpus 0"


EXPERIMENT_ARGS=" --init_k 1 \
--clusternet_hidden_layer_list 50 50 \
--start_computing_params 25 \
--how_to_compute_mu soft_assign \
--how_to_init_mu_sub kmeans_1d \
--start_sub_clustering 45 \
--start_splitting 55 \
--start_merging 55 \
--split_merge_every_n_epochs 30 \
--prior_sigma_scale .005 \
--NIW_prior_nu 300"

ARGS=$MINIMAL_ARGS$EXPERIMENT_ARGS

echo $ARGS
export CUDA_VISIBLE_DEVICES=0
python DeepDPM.py $ARGS
