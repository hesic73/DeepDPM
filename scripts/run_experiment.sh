#!/bin/bash

MINIMAL_ARGS="--dataset CNG \
--dir /Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/cng_features \
--seed 12345 \
--project DeepDPM_CNG \
--exp_name test \
--use_labels_for_eval \
--save_checkpoints \
--max_epochs 500 \
--gpus 0"


EXPERIMENT_ARGS=

ARGS=$MINIMAL_ARGS $EXPERIMENT_ARGS

echo $ARGS

python DeepDPM.py $ARGS
