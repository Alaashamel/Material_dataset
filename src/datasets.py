from pathlib import Path
from typing import Tuple, List
from torchvision import transforms, datasets
import torch
from PIL import Image
import re
import random
from .utils import IMAGENET_MEAN, IMAGENET_STD

class SimpleDS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert('RGB')
        x = self.tf(img)
        return x, y

def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf

def get_dataloaders(data_dir: str, img_size: int, batch_size: int, num_workers: int = 0):
    root = Path(data_dir)
    if (root / "train").exists():
        train_tf, val_tf = get_transforms(img_size)
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
        val_ds = datasets.ImageFolder(root / "val", transform=val_tf)
        test_ds = datasets.ImageFolder(root / "test", transform=val_tf)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dl, val_dl, test_dl, train_ds.classes
    else:
        img_dir = root if root.exists() else Path('JPEGImages')
        files: List[Path] = [p for p in img_dir.glob('*.jpg')]
        token_map = {}
        for p in files:
            m = re.search(r"_([A-Za-z]+)\.[^.]+$", p.name)
            if m:
                token = m.group(1).lower()
                token_map.setdefault(token, []).append(p)
        classes = sorted(token_map.keys())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        random.seed(42)
        train_files = []
        val_files = []
        test_files = []
        for c, arr in token_map.items():
            random.shuffle(arr)
            n = len(arr)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)
            train_files += [(p, class_to_idx[c]) for p in arr[:n_train]]
            val_files += [(p, class_to_idx[c]) for p in arr[n_train:n_train+n_val]]
            test_files += [(p, class_to_idx[c]) for p in arr[n_train+n_val:]]
        train_tf, val_tf = get_transforms(img_size)

        train_ds = SimpleDS(train_files, train_tf)
        val_ds = SimpleDS(val_files, val_tf)
        test_ds = SimpleDS(test_files, val_tf)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dl, val_dl, test_dl, classes
