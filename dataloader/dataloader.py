import os

import torch
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def imagenet_loader(
        is_train: bool,
        is_distributed: bool,
        dataset_dir: str,
        batch_size: int,
        num_workers: int):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    sampler = None

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        shuffle = True

    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        shuffle = False

    dataset = datasets.ImageFolder(dataset_dir, transform)
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler)

    return loader, sampler
