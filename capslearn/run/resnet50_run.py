import os
import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import capslearn.torch.optimizer as opt
import capslearn.torch.utils.cifar10_load as ld
import capslearn.torch.utils.others as utility

from capslearn.torch.distributed import DistributedDataParallel

from tqdm import tqdm


# Argument Parsing
args = utility.argument_parsing()

device = torch.device("cuda:0")
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr
rank = args.rank
world_size = args.world_size


os.environ['MASTER_ADDR'] = args.master_addr
os.environ['MASTER_PORT'] = args.master_port
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])

#transforms = transforms.Compose([
#                transforms.Resize((70, 70)),
#                transforms.ToTensor(),
#                ])

train_loader, test_loader, num_classes = ld.get_loader(batch_size=batch_size,
                                                    data_dir=args.data,
                                                    resize=70,
                                                    num_workers=args.workers,
                                                    sampler=True)


# Define model with CaPS-DDP
model = models.resnet50(num_classes=num_classes)
model = model.to(device)
model = DistributedDataParallel(model, device_ids=[0], find_unused_parameters=True)


with open("/home/with1015/CaPS-Learn/capslearn/run/log/layer_name.txt", 'a') as f:
    for n, _ in model.named_modules():
        f.write(str(n) + "\n")

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Apply CapsOptimizer
scheduling_freq = args.scheduling_freq * len(train_loader)
optimizer = opt.CapsOptimizer(optimizer,
                              unchange_rate=args.unchange_rate,
                              lower_bound=args.lower_bound,
                              scheduling_freq=scheduling_freq,
                              history_length=args.history_length,
                              round_factor=args.round_factor,
                              random_select=args.random_select,
                              log_mode=args.log_mode,
                              log_dir=args.log_dir)

for epoch in range(epochs):
    print("Epoch: ", epoch)
    with tqdm(train_loader, unit="iter") as iteration:
        for idx, data in enumerate(iteration):
            inputs, labels = data
            inputs = inputs.cuda(device, non_blocking=True)
            labels = labels.cuda(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            loss = criterion(outputs, labels)
            correctness = (predictions == labels).sum().item()
            accuracy = correctness / batch_size
            loss.backward()
            optimizer.step()

            iteration.set_postfix(loss=loss.item(), accuracy=100.*accuracy)

        optimizer.get_validation(loss.item())
