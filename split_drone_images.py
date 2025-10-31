#!/usr/bin/env python3
"""
Split and copy images from class subfolders into train/test directories.

Source tree expectation (example):
real_drone_photos_85_compression/
  car/
    images/
      img001.jpg
      img002.jpg
  truck/
    images/
      ...

This script will create the following structure under `drone_images`:

- drone_images/train/<class_name>/...
- drone_images/test/<class_name>/...

By default, 80% of images go to train and 20% to test, per class.

Usage:
  python split_drone_images.py \
    --source real_drone_photos_85_compression \
    --dest drone_images \
    --train-ratio 0.8 \
    [--seed 42] [--dry-run]

Notes:
- Files are COPIED (not moved). Existing identical filenames will be overwritten unless --skip-existing is provided.
- Only regular files inside each `<class>/images` folder are considered. Non-image files are not filtered by extension to stay generic,
  but you can restrict via --ext if needed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence
import random


def iter_files(folder: Path, allow_exts: Sequence[str] | None = None) -> Iterable[Path]:
    if not folder.exists():
        return []
    for p in folder.iterdir():
        if p.is_file():
            if allow_exts:
                if p.suffix.lower().lstrip(".") in allow_exts:
                    yield p
            else:
                yield p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_files(files: Sequence[Path], dest_dir: Path, skip_existing: bool = False, dry_run: bool = False) -> int:
    ensure_dir(dest_dir)
    count = 0
    for src in files:
        dst = dest_dir / src.name
        if skip_existing and dst.exists():
            continue
        if dry_run:
            print(f"DRY-RUN copy {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        count += 1
    return count


def split_indices(n: int, train_ratio: float, rng: random.Random) -> tuple[List[int], List[int]]:
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(round(n * train_ratio))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return train_idx, test_idx


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy images into train/test splits by class.")
    parser.add_argument("--source", type=Path, default=Path("real_drone_photos_85_compression"), help="Source root directory")
    parser.add_argument("--dest", type=Path, default=Path("drone_images"), help="Destination root directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proportion of images to put in train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--ext", nargs="*", default=None, help="Optional list of allowed file extensions (without dot), e.g. jpg png jpeg")
    parser.add_argument("--skip-existing", action="store_true", help="Skip copying files that already exist at destination")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying files")
    args = parser.parse_args(argv)

    source_root: Path = args.source
    dest_root: Path = args.dest
    train_ratio: float = args.train_ratio

    if not source_root.exists():
        print(f"Source directory not found: {source_root}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)

    total_train = 0
    total_test = 0
    classes_processed = 0

    # Iterate subdirectories of source_root; each subdir is a class name
    for class_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        images_dir = class_dir / "images"
        files = list(iter_files(images_dir, allow_exts=args.ext))
        if not files:
            # No files found; skip this class
            continue

        n = len(files)
        train_idx, test_idx = split_indices(n, train_ratio, rng)
        train_files = [files[i] for i in train_idx]
        test_files = [files[i] for i in test_idx]

        train_dest = dest_root / "train" / class_name
        test_dest = dest_root / "test" / class_name

        n_train_copied = copy_files(train_files, train_dest, skip_existing=args.skip_existing, dry_run=args.dry_run)
        n_test_copied = copy_files(test_files, test_dest, skip_existing=args.skip_existing, dry_run=args.dry_run)

        total_train += n_train_copied
        total_test += n_test_copied
        classes_processed += 1

        print(f"Class '{class_name}': {n} found -> train {n_train_copied}, test {n_test_copied}")

    print("\nSummary:")
    print(f"Classes processed: {classes_processed}")
    print(f"Total copied to train: {total_train}")
    print(f"Total copied to test:  {total_test}")

    if classes_processed == 0:
        print("Warning: no classes with images were found. Check the source directory structure.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
