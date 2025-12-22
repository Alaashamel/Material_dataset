from pathlib import Path
from typing import Tuple, List
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

from .utils import IMAGENET_MEAN, IMAGENET_STD


class SimpleDS(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.tf(img)
        return img, label


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, val_tf


def get_dataloaders(data_dir: str, img_size: int, batch_size: int, num_workers: int = 0):
    root = Path(data_dir)
    train_tf, val_tf = get_transforms(img_size)

    # الحالة الأولى: train / val / test folders
    if (root / "train").exists():
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
        val_ds   = datasets.ImageFolder(root / "val",   transform=val_tf)
        test_ds  = datasets.ImageFolder(root / "test",  transform=val_tf)

        return (
            DataLoader(train_ds, batch_size, shuffle=True,  num_workers=num_workers),
            DataLoader(val_ds,   batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test_ds,  batch_size, shuffle=False, num_workers=num_workers),
            train_ds.classes
        )

    # الحالة الثانية: صور في فولدر واحد
    files = list(root.glob("*.jpg"))
    token_map = {}

    for p in files:
        m = re.search(r"_([A-Za-z]+)\.", p.name)
        if m:
            cls = m.group(1).lower()
            token_map.setdefault(cls, []).append(p)

    classes = sorted(token_map.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    random.seed(42)
    train_items, val_items, test_items = [], [], []

    for cls, paths in token_map.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(0.7 * n)
        n_val   = int(0.15 * n)

        train_items += [(p, class_to_idx[cls]) for p in paths[:n_train]]
        val_items   += [(p, class_to_idx[cls]) for p in paths[n_train:n_train+n_val]]
        test_items  += [(p, class_to_idx[cls]) for p in paths[n_train+n_val:]]

    return (
        DataLoader(SimpleDS(train_items, train_tf), batch_size, shuffle=True),
        DataLoader(SimpleDS(val_items,   val_tf),   batch_size, shuffle=False),
        DataLoader(SimpleDS(test_items,  val_tf),   batch_size, shuffle=False),
        classes
    )
