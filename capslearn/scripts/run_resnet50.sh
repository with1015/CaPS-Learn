#!/bin/bash

LOG_DIR=${HOME}/CaPS-Learn/capslearn/logs
DATA_DIR=/scratch/with1015/datasets/cifar10
RUN_DIR=${HOME}/CaPS-Learn/capslearn/run

APP="resnet50_run.py"


source ${RUN_DIR}/set-env.sh
python3 ${RUN_DIR}/${APP} \
  --epoch 1 \
  --batch-size 32 \
  --lr 0.001 \
  --world-size 2 \
  --rank 0 \
  --log-mode False \
  $DATA_DIR

