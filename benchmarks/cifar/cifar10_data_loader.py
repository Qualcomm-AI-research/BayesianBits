# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch.utils.data as torch_data
import torchvision
import torchvision.transforms as transforms


class Cifar10DataLoader(object):
    def __init__(self, images_dir: str, batch_size: int, num_workers=1):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self):
        if not self._train_loader:
            train_set = torchvision.datasets.CIFAR10(
                root=self.images_dir,
                train=True,
                download=True,
                transform=self.transform_train,
            )
            self._train_loader = torch_data.DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
        return self._train_loader

    @property
    def val_loader(self):
        if not self._val_loader:
            val_set = torchvision.datasets.CIFAR10(
                root=self.images_dir,
                train=False,
                download=True,
                transform=self.transform_test,
            )
            self._val_loader = torch_data.DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return self._val_loader
