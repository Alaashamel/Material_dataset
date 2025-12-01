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

def train_one_epoch(model, dl, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def eval_one_epoch(model, dl, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data_split')
    p.add_argument('--model', type=str, default='resnet50')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_backbone', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_dir', type=str, default='models')
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device()
    train_dl, val_dl, test_dl, classes = get_dataloaders(args.data_dir, args.img_size, args.batch_size)
    num_classes = len(classes)
    model = build_model(args.model, num_classes, pretrained=True)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda t: t.requires_grad, model.parameters()), lr=args.lr_head)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val = 1e9
    records = []

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f'{args.model}_best.pt'

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, device, criterion, optimizer)
        val_loss, val_acc = eval_one_epoch(model, val_dl, device, criterion)
        scheduler.step(val_loss)
        records.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': args.model, 'state_dict': model.state_dict(), 'classes': classes}, ckpt_path)

    df = pd.DataFrame(records)
    df.to_csv(save_dir / f'{args.model}_training.csv', index=False)

if __name__ == '__main__':
    main()

