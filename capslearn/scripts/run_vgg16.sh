#!/bin/bash

LOG_DIR=${HOME}/capslearn/capslearn/logs
DATA_DIR=${HOME}/capslearn/capslearn/data
RUN_DIR=${HOME}/capslearn/capslearn/run

APP="vgg16_run.py"


source ${RUN_DIR}/set-env.sh
python3 ${RUN_DIR}/${APP}

