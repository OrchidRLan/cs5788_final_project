"""Checkpoint, logging, and visualization utilities."""
import os
import json
import torch
import numpy as np
from pathlib import Path


def save_checkpoint(model, optimizer, step, loss, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "step":           step,
        "loss":           loss,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    path = os.path.join(output_dir, f"checkpoint_{step:05d}.pt")
    torch.save(ckpt, path)
    if is_best:
        torch.save(ckpt, os.path.join(output_dir, "best.pt"))
    print(f"Saved checkpoint: {path}")
    return path


def load_checkpoint(model, optimizer, path, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Loaded checkpoint: {path} (step={ckpt['step']}, loss={ckpt['loss']:.4f})")
    return ckpt["step"]


def log_metrics(metrics: dict, step: int, log_file: str = "logs/train.jsonl"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    entry = {"step": step, **metrics}
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total | {trainable:,} trainable ({100*trainable/total:.1f}%)")
    return trainable
