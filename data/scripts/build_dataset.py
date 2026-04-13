"""
Combine Zero123++ outputs + Blender renders into unified dataset format.
Usage: python data/scripts/build_dataset.py
"""
import os
import json
import shutil
import random
from pathlib import Path

ZERO123_DIR = "data/zero123_out"
BLENDER_DIR = "data/train/laukry"
TRAIN_DIR   = "data/train"
TEST_DIR    = "data/test"
TEST_RATIO  = 0.2   # 20% for test (10 out of 50)
SEED        = 42

random.seed(SEED)

def build():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Collect all Zero123++ samples
    samples = sorted(Path(ZERO123_DIR).iterdir())
    random.shuffle(samples)

    n_test  = max(1, int(len(samples) * TEST_RATIO))
    n_train = len(samples) - n_test

    print(f"Total: {len(samples)} | Train: {n_train} | Test: {n_test}")

    for i, sample in enumerate(samples):
        split = "test" if i < n_test else "train"
        dest = Path(f"data/{split}/{sample.name}")
        if not dest.exists():
            shutil.copytree(sample, dest)
        print(f"  [{split}] {sample.name}")

    # Blender render goes to train only (it's one fixed character)
    print(f"\nBlender renders already in: {BLENDER_DIR}")
    print("Dataset ready.")

    # Print summary
    train_samples = list(Path(TRAIN_DIR).iterdir())
    test_samples  = list(Path(TEST_DIR).iterdir())
    print(f"\nFinal dataset:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

if __name__ == "__main__":
    build()
