# CaPS-Learn: Convergence-aware Parameter Skipped Learning System in Distributed Environment
### 2021 CSE61401 AI framework Project
Hyunjoon Jeong 20215350, UNIST

## Requirement
~~~
torch == 1.8.0 (recommand)
torchvision == 0.8.0 (recommand)
tqdm == 4.60.0 (recommand)
~~~

## How to Use
When you train your model, you just wrap your optimizer by CapsOptimizer.\
Refer below usage example and see capslearn/run/resnet50_run.py.\
You can test CapsOptimizer using script files in capslearn/scripts.
~~~
import torch
import capslearn.torch.optimizer as opt
from capslearn.torch.distributed import DistributedDataParallel

...

optimizer = torch.optim.Adam(model.paramters(), lr=learning_rate)
optimizer = opt.CapsOptimizer(optimizer, **CapsParameter)
~~~

## Run example
See capslearn/script examples
~~~
python3 resnet50_run.py \
  --epoch 100 \
  --batch-size 32 \
  --workers 16 \
  --lr 0.001 \
  --unchange-rate 90.0 \
  --lower-bound 0.0 \
  --scheduling-freq 10 \
  --history-length 5 \
  --round-factor -1 \
  --random-select 0.001 \
  --world-size 4 \
  --rank 0 \
  --master-addr $MASTER_ADDR \
  --master-port $MASTER_PORT \
  $DATA_DIR
~~~

## Input Parameter List
~~~
# Training parameter
data $DIR : dataset directory
--workers $INT : number of workers to use in data load
--epochs $INT : number of epochs to train
--batch-size $INT : batch size to train
--lr $FLOAT : Learning rate
--momentum $FLOAT : Momentum using in optimizer
--weight-decay $FLOAT : Weight decay using in optimizer

# Distributed training parameter
--gpu $INT : GPU ID to train in current node
--rank $INT : Rank to determine each worker
--world-size $INT : World-size for DistributedDataParallel (must same with total number of workers)
--master-addr $STR : Master node address to load parameter server
--master-port $STR : Port to use for communication

# CapsOptimizer options
--unchange-rate $FLOAT : Start rate of unchange paramter ratio in one layer.
--scheduling-freq $INT : Scheduling frequency to adjust unchage rate
--lower-bound $FLOAT : Set the lowest value of unchange rate scheduling
--max-bound $FLOAT : Set the maximum value of unchange rate scheduling
--history-lenght $INT : Determine size of history queue for unchage rate scheduling
--random-select $INT : Use random index selection in CapsOptimizer instead of naive solution
--hbs-init $INT : Use history-based selection with initial step instead of naive solution
--round-factor $INT : Rounding digit in comparison between layers (not recommend to use)
~~~
