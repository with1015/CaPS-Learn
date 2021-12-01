#!/bin/bash

LOG_DIR=${HOME}/CaPS-Learn/capslearn/logs
DATA_DIR=${HOME}/CaPS-Learn/capslearn/data
RUN_DIR=${HOME}/CaPS-Learn/capslearn/run

APP="alexnet_dist.py"


source ${RUN_DIR}/set-env.sh
python3 ${RUN_DIR}/${APP}

