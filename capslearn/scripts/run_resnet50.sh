#!/bin/bash

LOG_DIR=${HOME}/CaPS-Learn/capslearn/logs
DATA_DIR=/scratch/with1015/datasets/cifar10
RUN_DIR=${HOME}/CaPS-Learn/capslearn/run

APP="resnet50_run.py"
MASTER_ADDR="dumbo049"
MASTER_PORT="28000"

source ${RUN_DIR}/set-env.sh
python3 ${RUN_DIR}/${APP} \
  --epoch 1 \
  --batch-size 32 \
  --lr 0.001 \
  --world-size 2 \
  --rank 0 \
  --master-addr $MASTER_ADDR \
  --master-port $MASTER_PORT \
  --log-mode \
  $DATA_DIR

