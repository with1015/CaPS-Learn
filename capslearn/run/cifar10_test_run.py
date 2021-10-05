import time

import torch
import torchvision
import torchvision.transforms as transforms

import capslearn.models.cifar10_net as TestNet
import capslearn.torch.optimizer as opt

import capslearn.torch.utils.cifar10_load as ld

from tqdm import tqdm

device = "cuda:0"

batch_size = 4
epochs = 30
learning_rate = 0.001
momentum = 0.9

train_loader, test_loader, classes = ld.get_loader(batch_size=batch_size, resize=32,
                                                    data_dir="/home/with1015/capslearn/capslearn/data")
num_classes = len(classes)

model = TestNet.Net().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = opt.TestOptimizer(optimizer)

for epoch in range(epochs):
    print("Epoch: ", epoch)
    with tqdm(train_loader, unit="iter") as iteration:
        for idx, data in enumerate(iteration):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iteration.set_postfix(loss=loss.item())


#with open("/home/with1015/capslearn/capslearn/logs/log_cifar10Net.txt", 'a') as f:
#    for percent in optimizer.percent_set:
#        f.write(str(percent) + "\n")
