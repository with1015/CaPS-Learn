import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import capslearn.torch.utils.others as utility
import capslearn.torch.utils.cifar10_load as ld

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
model = DistributedDataParallel(model, device_ids=[0])

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    print("Epoch: ", epoch)
    with tqdm(train_loader, unit="iter") as iteration:
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=10, warmup=10, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/with1015/CaPS-Learn/capslearn/scripts/log/baseline'),
                record_shapes=True,
                with_stack=True
                ) as prof:
            for idx, data in enumerate(iteration):
                inputs, labels = data
                inputs = inputs.cuda(device, non_blocking=True)
                labels = labels.cuda(device, non_blocking=True)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                prof.step()

            iteration.set_postfix(loss=loss.item())

