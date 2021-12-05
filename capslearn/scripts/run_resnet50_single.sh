#!/bin/bash

LOG_DIR=${HOME}/CaPS-Learn/capslearn/logs
DATA_DIR=/scratch/with1015/datasets/cifar10
RUN_DIR=${HOME}/CaPS-Learn/capslearn/run

APP="resnet50_run.py"
MASTER_ADDR="localhost"
MASTER_PORT="28000"

source ${RUN_DIR}/set-env.sh
python3 ${RUN_DIR}/${APP} \
  --epoch 100 \
  --batch-size 32 \
  --workers 16 \
  --lr 0.001 \
  --unchange-rate 10.0 \
  --lower-bound 0.0 \
  --scheduling-freq 10 \
  --history-length 5 \
  --round-factor -1 \
  --world-size 1 \
  --rank 0 \
  --master-addr $MASTER_ADDR \
  --master-port $MASTER_PORT \
  $DATA_DIR

