#!/bin/bash

PROJECT=DeepDPM_CNG
EXP_NAME=test1
DATASET=CNG

MINIMAL_ARGS="--dataset "$DATASET" \
--dir ./pretrained_embeddings/umap_embedded_datasets/MNIST \
--seed 12345 \
--project "$PROJECT" \
--exp_name "$EXP_NAME" \
--use_labels_for_eval \
--save_checkpoints \
--max_epochs 500 \
--batch-size 128 \
--gpus 0,1,2,3"


EXPERIMENT_ARGS=" --init_k 20"

ARGS=$MINIMAL_ARGS$EXPERIMENT_ARGS

echo $ARGS
export CUDA_VISIBLE_DEVICES=4,5,6,7
python DeepDPM.py $ARGS
