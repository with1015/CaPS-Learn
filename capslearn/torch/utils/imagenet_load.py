import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


def get_loader(batch_size=32, data_dir="./data", transforms=None):

    imagenet_train = torchvision.datasets.ImageFolder(root=data_dir + "/train", transform=transforms)
    imagenet_test = torchvision.datasets.ImageFolder(root=data_dir + "/val", transform=transforms)

    train_loader = DataLoader(imagenet_train,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=16,
                            prefetch_factor=2,
                            pin_memory=True)

    test_loader = DataLoader(imagenet_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16,
                            prefetch_factor=2,
                            pin_memory=True)

    return train_loader, test_loader


