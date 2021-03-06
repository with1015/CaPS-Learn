import time

import torch
import torchvision
import torchvision.transforms as transforms

import capslearn.models.alexnet as alexnet
import capslearn.torch.optimizer as opt

import capslearn.torch.utils.cifar10_load as ld
#import capslearn.torch.utils.imagenet_load as ld

from tqdm import tqdm

cuda = torch.device("cuda:0")

batch_size = 128
epochs = 300
learning_rate = 0.0001

train_loader, test_loader, num_classes = ld.get_loader(batch_size=batch_size, resize=70,
                                                    data_dir="/home/with1015/capslearn/capslearn/data")

#transforms = transforms.Compose([
#                transforms.Resize((224, 224)),
#                transforms.ToTensor(),
#                ])

#train_loader, test_loader = ld.get_loader(batch_size=batch_size,
#                                        data_dir="/scratch/with1015/datasets/imagenet_2012",
#                                        transforms=transforms)

#num_classes = 1000

model = alexnet.AlexNet(num_classes=num_classes).to(device=cuda)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = opt.CapsOptimizer(optimizer, unchange_rate=99.0)

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
            #with open("/home/with1015/capslearn/capslearn/logs/loss.txt", 'a') as f:
            #    f.write(str(loss.item()) + "\n")

#with open("/home/with1015/capslearn/capslearn/logs/log.txt", 'a') as f:
#    for percent in optimizer.percent_set:
#        f.write(str(percent) + "\n")

print("Test accuracy...")
correct = 0
total = 0

with tqdm(test_loader, unit="iter") as iteration:
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

print("Accuracy: %d %%" % (100 * correct / total))
