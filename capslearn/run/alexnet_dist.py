import time
import socket

import torch
import torchvision
import torchvision.transforms as transforms

import capslearn.models.alexnet as alexnet
import capslearn.torch.optimizer as opt
import capslearn.torch.distributed as test_dist

#import capslearn.torch.utils.cifar10_load as ld
import capslearn.torch.utils.imagenet_load as ld

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

cuda = torch.device("cuda:0")

batch_size = 256
epochs = 10
learning_rate = 0.1

# For CIFAR-10
#train_loader, test_loader, classes = ld.get_loader(batch_size=batch_size, resize=70,
#                                                    data_dir="/home/with1015/capslearn/capslearn/data")

transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])

train_loader, test_loader = ld.get_loader(batch_size=batch_size,
                                        data_dir="/scratch/with1015/datasets/imagenet_2012",
                                        transforms=transforms)

num_classes = 1000
model = alexnet.AlexNet(num_classes=num_classes).to(device=cuda)

if socket.gethostname() == "dumbo041":
    test_dist.setup(0, 2)
    model = DDP(model, device_ids=[0])
else:
    test_dist.setup(1, 2)
    model = DDP(model, device_ids=[0])

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = opt.TestOptimizer(optimizer)

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

test_dist.cleanup()
#with open("/home/with1015/capslearn/capslearn/logs/log.txt", 'a') as f:
#    for percent in optimizer.percent_set:
#        f.write(str(percent) + "\n")
