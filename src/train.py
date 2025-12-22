import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd

from .datasets import get_dataloaders
from .models import build_model
from .utils import seed_all, get_device


def run_epoch(model, dl, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in tqdm(dl, leave=False):
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            out = model(x)
            if isinstance(out, tuple):
                out = out[0]

            loss = criterion(out, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data_split')
    p.add_argument('--model', default='resnet50')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_backbone', type=float, default=1e-4)
    p.add_argument('--save_dir', default='models')
    args = p.parse_args()

    seed_all(42)
    device = get_device()

    train_dl, val_dl, test_dl, classes = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size
    )

    model = build_model(args.model, len(classes), pretrained=True)
    model.to(device)

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': args.lr_head}
    ])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    best_val = 1e9
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_dl, device, criterion)
        scheduler.step(val_loss)

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'classes': classes
            }, save_dir / f'{args.model}_best.pt')

    pd.DataFrame(history).to_csv(save_dir / f'{args.model}_training.csv', index=False)


if __name__ == '__main__':
    main()