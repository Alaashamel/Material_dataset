import argparse
from pathlib import Path

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from .datasets import get_dataloaders
from .models import build_model
from .utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data_split')
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--out_dir', type=str, default='docs')
    args = p.parse_args()

    # device
    device = get_device()

    # data
    _, _, test_dl, classes = get_dataloaders(
        args.data_dir,
        args.img_size,
        args.batch_size
    )

    # model
    model = build_model(
        args.model,
        num_classes=len(classes),
        pretrained=False
    )

    state = torch.load(args.checkpoint, map_location=device)

    # أمان مع أي شكل checkpoint
    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    # inference
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            out = model(x)

            if isinstance(out, tuple):
                out = out[0]

            preds = out.argmax(1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save classes
    pd.DataFrame({'class': classes}).to_csv(
        out_dir / 'classes.csv',
        index=False
    )

    # save confusion matrix csv
    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    ).to_csv(out_dir / f'{args.model}_confusion_matrix.csv')

    # plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_dir / f'{args.model}_confusion_matrix.png')
    plt.close()

    # save metrics
    pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1'],
        'value': [acc, prec, rec, f1]
    }).to_csv(
        out_dir / f'{args.model}_metrics.csv',
        index=False
    )


if __name__ == '__main__':
    main()
