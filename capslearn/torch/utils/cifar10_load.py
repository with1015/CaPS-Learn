import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


def get_loader(batch_size=32, resize=224, data_dir="./data", num_workers=4):

    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, len(classes)

