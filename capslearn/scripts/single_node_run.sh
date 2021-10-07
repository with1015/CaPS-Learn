#!/bin/bash


data_dir='/scratch/with1015/datasets/imagenet_2012'
if [ ! -d $data_dir ]; then
  echo "No such data dir:"$data_dir
  exit 1
fi

source ../run/set-env.sh

model='resnet50'
visible_gpus='0'
batch_size=128  # total batch size on current node
epochs=1

log_file=""
hostname=`hostname`
echo $hostname
if [ -z $log_file ]; then
  log_file="${model}_${batch_size}_${hostname}.txt"
fi
echo "log file:"${log_file}

#gpu_list=0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3
gpu_list=0
echo -e "run main.py!"
CUDA_VISIBLE_DEVICES=$visible_gpus python3 ../run/resnet50_run_v2.py \
  -b $batch_size \
  --epochs $epochs \
  --gpu 0 \
  $data_dir
  #$data_dir 2>&1 | tee -a -i $log_file 

echo -e "finish main.py!"
