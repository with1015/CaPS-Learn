import torch
import torch.distributed as dist
import torchvision
import torchvision.models as models

from capslearn.torch.distributed_test2 import DistributedDataParallel


cuda = torch.device("cuda:0")

dist.init_process_group(backend='gloo', rank=0, world_size=2)

model = torch.tensor()
model = DistributedDataParallel(model, device_ids=[0])
