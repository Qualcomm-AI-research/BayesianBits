# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch.utils.data as torch_data
import torchvision
import torchvision.transforms as transforms


class MnistDataLoader(object):
    def __init__(self, images_dir: str, batch_size: int):
        train_set = torchvision.datasets.MNIST(
            root=images_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self._train_loader = torch_data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )

        test_set = torchvision.datasets.MNIST(
            root=images_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        self._val_loader = torch_data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=2
        )

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._val_loader
