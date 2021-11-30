import os
import torch
import torch.distributed as dist
import torchvision
import torchvision.models as models

from capslearn.torch.distributed_test2 import DistributedDataParallel

cuda = torch.device("cuda:0")

os.environ['MASTER_ADDR'] = 'dumbo049'
os.environ['MASTER_PORT'] = '38000'

rank = 0
dist.init_process_group(backend='gloo', rank=rank, world_size=2)

inputs = torch.rand(4).to(cuda)

model = torch.nn.Linear(4, 1).to(cuda)
print(rank, list(model.parameters()))
model = DistributedDataParallel(model, device_ids=[0])

output = model(inputs)
print(rank, list(model.parameters()))
