import os
import argparse

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import torchvision
import torchvision.transforms as transforms

import capslearn.torch.distributed as dist
import capslearn.torch.utils.imagenet_load as ld
import capslearn.models.alexnet as alexnet

from threading import Lock

param_server = None
global_lock = Lock()


def get_parameter_server(num_gpus=0, model=None):

    global param_server
    with global_lock:
        if not param_server:
            param_server = dist.ParameterServer(model=model, num_gpus=num_gpus)
        return param_server


def run_parameter_server(rank, world_size):
    rpc.init_rpc(name="paramter_server", rank=rank, world_size=world_size)
    print("Init RPC PS done!")
    rpc.shutdown()


def run_training_loop(rank, num_gpus, train_loader, test_loader):
    model = alexnet.AlexNet()
    trainer = TrainerNet(model=model, num_gpus=num_gpus)
    param_rrefs = trainer.get_global_param_rref()
    optimizer = DistributedOptimizer(torch.optim.SGD, param_rrefs, lr=0.03)
    criterion = torch.nn.CrossEntropyLoss()

    for idx, data in enumerate(train_loader):
        with dist_autograd.context() as cid:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = trainer(inputs)
            loss = criterion(outputs, labels)
            dist_autograd.backward(cid, [loss])
            assert(dist.remote_method(
                    dist.ParameterServer.get_dist_gradients,
                    trainer.param_server_rref, cid
                    ) != {})
            optimizer.step(cid)

            if idx % 5 == 0:
                print(f"Rank {rank} training batch {idx} loss {loss.item()}")


def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=world_size)
    run_training_loop(rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()


class TrainerNet(nn.Module):

    def __init__(self, model=None, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.model = model
        self.param_server_rref = rpc.remote(0, get_parameter_server, args=(num_gpus, model))


    def get_global_param_rref(self):
        remote_params = dist.remote_method(dist.ParameterServer.get_param_rrefs, self.param_server_rref)
        return remote_params


    def forward(self, x):
        output = dist.remote_method(dist.ParameterServer.forward, self.param_server_rref, x.to("cpu"))
        return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--rank", "-r", type=int, default=None)

    args = parser.parse_args()

    num_gpus = 1
    batch_size = 32
    world_size = args.world_size

    os.environ['MASTER_ADDR'] = "ib041"
    os.environ['MASTER_PORT'] = "50051"

    processes = []

    if args.rank == 0:
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        transforms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        ])

        train_loader, test_loader = ld.get_loader(batch_size=batch_size,
                                                data_dir="/scratch/with1015/datasets/imagenet_2012",
                                                transforms=transforms)

        p = mp.Process(target=run_worker,
                        args=(args.rank, world_size, num_gpus, train_loader, test_loader)
                        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
