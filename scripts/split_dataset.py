import argparse
import os
import random
import shutil
import json
from pathlib import Path
import re

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def find_classes(src: Path):
    classes = []
    for item in sorted(src.iterdir()):
        if item.is_dir():
            files = [p for p in item.rglob("*") if is_image(p)]
            if files:
                classes.append(item.name)
    return classes

def collect_class_files(src: Path, cls: str):
    root = src / cls
    return [p for p in root.rglob("*") if is_image(p)]

def extract_token_from_name(p: Path):
    m = re.search(r"_([A-Za-z]+)\.[^.]+$", p.name)
    return m.group(1).lower() if m else None

def split_indices(n, train_ratio, val_ratio, seed):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return train_idx, val_idx, test_idx

def copy_files(files, out_dir: Path, cls: str):
    dest = out_dir / cls
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dest / f.name
        i = 1
        while target.exists():
            target = dest / (f.stem + f"_{i}" + f.suffix)
            i += 1
        shutil.copy2(str(f), str(target))

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_from_names", action="store_true")
    args = parser.parse_args()

    src = Path(args.source_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio if args.test_ratio is not None else max(0.0, 1.0 - train_ratio - val_ratio)

    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise SystemExit("Invalid ratios")
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise SystemExit("Ratios must sum to 1.0")

    classes = find_classes(src)
    class_to_idx = {}
    splits = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)
        for cls in classes:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    if classes and not args.infer_from_names:
        class_to_idx = {cls: i for i, cls in enumerate(sorted(classes))}
        for cls in classes:
            files = collect_class_files(src, cls)
            if not files:
                continue
            train_idx, val_idx, test_idx = split_indices(len(files), train_ratio, val_ratio, args.seed)
            train_files = [files[i] for i in train_idx]
            val_files = [files[i] for i in val_idx]
            test_files = [files[i] for i in test_idx]
            copy_files(train_files, out_root / "train", cls)
            copy_files(val_files, out_root / "val", cls)
            copy_files(test_files, out_root / "test", cls)
            splits["train"][cls] = len(train_files)
            splits["val"][cls] = len(val_files)
            splits["test"][cls] = len(test_files)
    else:
        images = [p for p in src.rglob("*") if is_image(p)]
        if not images:
            raise SystemExit("No images found")
        tokens = {}
        for p in images:
            tok = extract_token_from_name(p)
            if tok:
                tokens.setdefault(tok, []).append(p)
        if not tokens:
            raise SystemExit("Could not infer classes from filenames")
        classes = sorted(tokens.keys())
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for split in ["train", "val", "test"]:
            for cls in classes:
                (out_root / split / cls).mkdir(parents=True, exist_ok=True)
        for cls, files in tokens.items():
            train_idx, val_idx, test_idx = split_indices(len(files), train_ratio, val_ratio, args.seed)
            train_files = [files[i] for i in train_idx]
            val_files = [files[i] for i in val_idx]
            test_files = [files[i] for i in test_idx]
            copy_files(train_files, out_root / "train", cls)
            copy_files(val_files, out_root / "val", cls)
            copy_files(test_files, out_root / "test", cls)
            splits["train"][cls] = len(train_files)
            splits["val"][cls] = len(val_files)
            splits["test"][cls] = len(test_files)

    meta = {
        "source_dir": str(src),
        "output_dir": str(out_root),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "classes": sorted(classes),
        "class_to_idx": class_to_idx,
        "counts": splits
    }
    save_json(class_to_idx, out_root / "class_to_idx.json")
    save_json(meta, out_root / "split_summary.json")

if __name__ == "__main__":
    main()

