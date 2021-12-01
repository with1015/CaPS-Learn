import os
import time
import socket

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import capslearn.models.alexnet as alexnet
import capslearn.torch.optimizer as opt
from capslearn.torch.distributed_test2 import DistributedDataParallel

#import capslearn.torch.utils.cifar10_load as ld
import capslearn.torch.utils.imagenet_load as ld

from tqdm import tqdm

cuda = torch.device("cuda:0")

batch_size = 128
epochs = 10
learning_rate = 0.1

os.environ['MASTER_ADDR'] = 'dumbo049'
os.environ['MASTER_PORT'] = '28000'

# For CIFAR-10
#train_loader, test_loader, classes = ld.get_loader(batch_size=batch_size, resize=70,
#                                                    data_dir="/home/with1015/capslearn/capslearn/data")
transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])

train_loader, test_loader = ld.get_loader(batch_size=batch_size,
                                        data_dir="/scratch/with1015/dataset",
                                        transforms=transforms)

num_classes = 1000
model = alexnet.AlexNet(num_classes=num_classes).to(device=cuda)

rank = 0
dist.init_process_group(backend='gloo', rank=rank, world_size=2)
model = DistributedDataParallel(model, device_ids=[0])

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = opt.CapsOptimizer(optimizer)

for epoch in range(epochs):
    print("Epoch: ", epoch)
    with tqdm(train_loader, unit="iter") as iteration:
        for idx, data in enumerate(iteration):
            inputs, labels = data
            inputs, labels = inputs.to(device=cuda), labels.to(device=cuda)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iteration.set_postfix(loss=loss.item())

#with open("/home/with1015/capslearn/capslearn/logs/log.txt", 'a') as f:
#    for percent in optimizer.percent_set:
#        f.write(str(percent) + "\n")
