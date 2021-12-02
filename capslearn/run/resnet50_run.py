import os
import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import capslearn.torch.optimizer as opt
import capslearn.torch.utils.cifar10_load as ld
from capslearn.torch.distributed import DistributedDataParallel
import capslearn.torch.utils.others as utility

from tqdm import tqdm

args = utility.argument_parsing()
device = torch.device("cuda:0")

batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr

rank = args.rank
world_size = args.world_size

os.environ['MASTER_ADDR'] = args.master_addr
os.environ['MASTER_PORT'] = args.master_port

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])

#transforms = transforms.Compose([
#                transforms.Resize((70, 70)),
#                transforms.ToTensor(),
#                ])

train_loader, test_loader, num_classes = ld.get_loader(batch_size=batch_size,
                                                    data_dir=args.data,
                                                    resize=70,
                                                    num_workers=args.num_workers)


model = models.resnet50(num_classes=num_classes)
model = model.to(device)

dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
model = DistributedDataParallel(model, device_ids=[0])

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Apply CapsOptimizer
scheduling_freq = args.scheduling_freq * len(train_loader)
optimizer = opt.CapsOptimizer(optimizer,
                              unchange_rate=args.unchange_rate,
                              lower_bound=args.lower_bound,
                              scheduling_freq=scheduling_freq,
                              history_length=args.history_length)

for epoch in range(epochs):
    print("Epoch: ", epoch)
    with tqdm(train_loader, unit="iter") as iteration:
        for idx, data in enumerate(iteration):
            inputs, labels = data
            inputs = inputs.cuda(device, non_blocking=True)
            labels = labels.cuda(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iteration.set_postfix(loss=loss.item())

    print("Validate epoch:", epoch)
    accuracy_cnt = 0
    for idx, data in test_loader:
        inputs, labels = data
        inputs = inputs.cuda(device)
        labels = labels.cuda(device)
        outputs = model(inputs)

        if outputs == labels:
            accuracy_cnt += 1

    accuracy = 100 * accuracy_cnt / len(test_loader)
    print("Accuracy:", accuracy)

    optimizer.get_validation(accuracy)
