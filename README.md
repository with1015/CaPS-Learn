# CaPS-Learn: Convergence-aware Parameter Skipped Learning System in Distributed Environment
2021 CSE61401 AI framework Project

## requirement
~~~
torch == 1.8.0 (recommand)
torchvision == 0.8.0 (recommand)
tqdm == 4.60.0 (recommand)
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
