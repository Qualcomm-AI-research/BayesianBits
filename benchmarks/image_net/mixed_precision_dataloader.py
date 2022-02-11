# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import os
import sys

from torchvision.datasets.folder import default_loader, has_file_allowed_extension
import torch.utils.data as torch_data
from torchvision import transforms


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif")


class ImageNetDataLoaders(object):
    """
    Data loader provider for ImageNet images, providing a train and a validation loader.
    It assumes that the structure of the images is
        images_dir
            - train
                - label1
                - label2
                - ...
            - val
                - label1
                - label2
                - ...
    """

    def __init__(
        self,
        images_dir: str,
        size: int,
        batch_size: int,
        num_workers: int,
        num_val_samples_per_class: int = None,
    ):
        """

        Parameters
        ----------
        images_dir: str
            Root image directory
        size : int
            Number of pixels the image will be re-sized to (square)
        batch_size : int
            Batch size of both the training and validation loaders
        num_workers
            Number of parallel workers loading the images
        num_val_samples_per_class : int
            Maximum number of images per class to load

        """

        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_samples_per_class = num_val_samples_per_class

        # For normalization, mean and std dev values are calculated per channel
        # and can be found on the web.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(size + 24),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self) -> torch_data.DataLoader:
        if not self._train_loader:
            train_set = CachedImageFolder(
                root=os.path.join(self.images_dir, "train"),
                transform=self.train_transforms,
            )
            self._train_loader = torch_data.DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._train_loader

    @property
    def val_loader(self) -> torch_data.DataLoader:
        if not self._val_loader:
            val_set = CachedImageFolder(
                root=os.path.join(self.images_dir, "val"),
                transform=self.val_transforms,
                num_samples_per_class=self.num_val_samples_per_class,
            )
            self._val_loader = torch_data.DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._val_loader


def make_dataset(
    directory: str,
    class_to_idx: dict,
    extensions: list,
    num_samples_per_class: int,
    cache_image_list: bool,
):
    images = []
    directory = os.path.expanduser(directory)
    from time import time

    t0 = time()

    def do_convert(i, s):
        if i == 0:
            return s
        elif i == 1:
            return int(s)
        else:
            raise ValueError("Should never get i not in 0, 1 here but got {}".format(i))

    cached_img_name = os.path.join(directory, "__CACHED_IMAGE_LIST.txt")
    if cache_image_list:
        if os.path.exists(cached_img_name):
            print("Reading image list from", cached_img_name)
            with open(cached_img_name, "r") as f:
                img_str = f.read().strip().split("\n")
                images = [
                    [do_convert(i, s) for i, s in enumerate(l.split("::"))]
                    for l in img_str
                ]
                print("Took: {:.4f}s".format(time() - t0))
                return images

    for class_name in sorted(class_to_idx.keys()):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            class_idx = class_to_idx[class_name]
            add_images_for_class(
                class_path, extensions, num_samples_per_class, class_idx, images
            )

    if cache_image_list:
        print("Caching image list to", cached_img_name)
        print("Reading up until now took: {:.4f}s".format(time() - t0))
        with open(cached_img_name, "w") as f:
            img_str = "\n".join(["::".join([str(s) for s in l]) for l in images])
            f.write(img_str)
    return images


def add_images_for_class(
    class_path, extensions, num_samples_per_class, class_idx, images
):
    count = 0
    for dir, _, file_names in sorted(os.walk(class_path)):
        for file_name in sorted(file_names):
            if num_samples_per_class and count >= num_samples_per_class:
                break
            if has_file_allowed_extension(file_name, extensions):
                image_path = os.path.join(dir, file_name)
                item = (image_path, class_idx)
                images.append(item)
                count += 1


class CachedImageFolder(torch_data.Dataset):
    """
        Copy of torchvision.datasets.folder.DatasetFolder, specific for images, with the
        possibility to limit the number of images per class.

        Parameters
        ----------
        root : str
            Root directory path
        transform :
            A function/transform that takes in a sample and returns a transformed version
            (optional)
        target_transform :
            A function/transform that takes in the target and transforms it (optional)
        num_samples_per_class : int
            Number of images per class to put in the data set. If None, all images will be
            taken into account.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        num_samples_per_class: int = None,
    ):
        classes, class_to_idx = self._find_classes(root)
        self.samples = make_dataset(
            root, class_to_idx, IMG_EXTENSIONS, num_samples_per_class, False
        )
        if not self.samples:
            raise (
                RuntimeError(
                    "Found 0 files in sub folders of: {}\nSupported extensions are: {}".format(
                        root, ",".join(IMG_EXTENSIONS)
                    )
                )
            )

        self.root = root
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in self.samples]

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples

    @staticmethod
    def _find_classes(directory: str):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        else:
            classes = [
                d
                for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
