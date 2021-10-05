import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import capslearn.torch.optimizer as opt
import capslearn.torch.utils.imagenet_load as ld

from tqdm import tqdm

device = torch.device("cuda:0")

batch_size = 128
epochs = 1
learning_rate = 0.1

model = models.resnet50()

transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])

train_loader, test_loader = ld.get_loader(batch_size=batch_size,
                                        data_dir="/scratch/with1015/datasets/imagenet_2012",
                                        transforms=transforms)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = opt.CapsOptimizer(optimizer, unchange_rate=90.0)

for epoch in range(epochs):
    print("Epoch: ", epoch)
    with tqdm(train_loader, unit="iter") as iteration:
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('../logs/resnet50'),
                record_shapes=True,
                with_stack=True
                ) as prof:
            for idx, data in enumerate(iteration):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                prof.step()

                iteration.set_postfix(loss=loss.item())
