#!/bin/bash
# ============================================================
# CS5788 Final Project - One-click setup
# Run in WSL: cd ~/projects && bash setup_project.sh
# ============================================================

set -e
PROJECT="cs5788_final_project"
echo "Creating project: $PROJECT"
mkdir -p $PROJECT && cd $PROJECT

# ── Top-level files ──────────────────────────────────────────
cat > .gitignore << 'EOF'
pretrained_weights/
data/train/
data/test/
data/cache/
*.npy
*.glb
*.obj
*.ply
__pycache__/
*.pyc
*.pyo
.env
.venv
logs/
checkpoints/
wandb/
*.egg-info/
dist/
build/
.DS_Store
Thumbs.db
EOF

cat > requirements.txt << 'EOF'
torch>=2.1.0
torchvision>=0.16.0
diffusers>=0.27.0
transformers>=4.38.0
accelerate>=0.27.0
peft>=0.9.0
einops>=0.7.0
trimesh>=4.0.0
pysdf>=0.1.1
open3d>=0.18.0
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0
tqdm>=4.66.0
omegaconf>=2.3.0
gradio>=4.20.0
lpips>=0.1.4
clip @ git+https://github.com/openai/CLIP.git
EOF

cat > README.md << 'EOF'
# CS5788: CircleMic Style 3D Generation via TripoSG Finetuning

**Jian-Peng Li · Jiangxiang Ling · Ruolan Chen** | Cornell CS5788 Spring 2026

## Overview
We finetune TripoSG on a CircleMic-style chibi character dataset to improve 
3D reconstruction fidelity for stylized cartoon inputs.

**Pipeline:**
```
MidJourney 2D image → Zero123++ multiview → TripoSG (finetuned) → 3D mesh
```

## Setup
```bash
conda create -n triposg_chibi python=3.10
conda activate triposg_chibi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -r requirements.txt
git clone https://github.com/VAST-AI-Research/TripoSG.git
pip install -e TripoSG/
```

## Data Preparation
```bash
# 1. Blender render (run inside Blender scripting panel)
python data/scripts/blender_render.py

# 2. Zero123++ multiview generation
python data/scripts/zero123pp_infer.py --input_dir data/mj_images/ --output_dir data/zero123_out/

# 3. Compute GT SDF
python data/scripts/compute_sdf.py --mesh_path data/laukry.glb --output data/cache/laukry_sdf.npy

# 4. Build dataset
python data/scripts/build_dataset.py
```

## Training
```bash
python scripts/train.py --config configs/default.yaml
```

## Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --test_dir data/test/
```

## Demo
```bash
python demo/app.py
```

## Results
*To be filled after experiments*

## Citation
```bibtex
@article{li2025triposg,
  title={TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models},
  author={Li, Yangguang and others},
  journal={arXiv preprint arXiv:2502.06608},
  year={2025}
}
```
EOF

# ── Directory structure ──────────────────────────────────────
mkdir -p data/scripts
mkdir -p data/mj_images
mkdir -p data/cache
mkdir -p triposg_finetune
mkdir -p configs/ablation
mkdir -p scripts
mkdir -p demo
mkdir -p notebooks
mkdir -p results/loss_curves
mkdir -p results/qualitative
touch data/mj_images/.gitkeep
touch data/cache/.gitkeep
touch results/loss_curves/.gitkeep
touch results/qualitative/.gitkeep

# ── configs/default.yaml ─────────────────────────────────────
cat > configs/default.yaml << 'EOF'
model:
  pretrained: "VAST-AI/TripoSG"
  freeze_vae: true          # freeze VAE encoder/decoder
  lora_rank: 16             # LoRA rank for DiT cross-attention
  lora_alpha: 32

training:
  output_dir: "checkpoints"
  num_steps: 3000
  batch_size: 1
  gradient_accumulation_steps: 4   # effective batch = 4
  learning_rate: 1.0e-5
  weight_decay: 1.0e-4
  lr_scheduler: "cosine"
  warmup_steps: 100
  save_every: 500
  log_every: 50
  mixed_precision: "fp16"          # saves VRAM on RTX 5070
  gradient_clip: 1.0
  seed: 42

loss:
  lambda_normal: 0.1
  lambda_eikonal: 0.05
  n_sample_points: 512             # reduce if OOM, increase for quality

data:
  train_dir: "data/train"
  test_dir: "data/test"
  image_size: 512
  num_workers: 2

eval:
  clip_model: "ViT-L/14"
EOF

cat > configs/ablation/no_normal.yaml << 'EOF'
defaults:
  - ../default

loss:
  lambda_normal: 0.0
  lambda_eikonal: 0.05
EOF

cat > configs/ablation/no_eikonal.yaml << 'EOF'
defaults:
  - ../default

loss:
  lambda_normal: 0.1
  lambda_eikonal: 0.0
EOF

# ── data/scripts/blender_render.py ───────────────────────────
cat > data/scripts/blender_render.py << 'EOF'
"""
Blender batch render script.
Run inside Blender: Scripting panel -> paste and run.
Or headless: blender --background model.blend --python blender_render.py
"""
import bpy
import math
import json
import os

OUTPUT_DIR = "/path/to/data/train/laukry"   # CHANGE THIS
N_VIEWS = 100
RESOLUTION = 512
ELEVATION_DEG = 20.0

os.makedirs(f"{OUTPUT_DIR}/multiview", exist_ok=True)

# Scene setup
scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = True   # RGBA, transparent background

# Place camera on sphere
cam = bpy.data.objects.get("Camera") or bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
bpy.context.scene.collection.objects.link(cam)
scene.camera = cam

radius = 2.5
cameras = []

for i in range(N_VIEWS):
    azimuth = (360.0 / N_VIEWS) * i
    elevation = ELEVATION_DEG

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    x = radius * math.cos(el_rad) * math.cos(az_rad)
    y = radius * math.cos(el_rad) * math.sin(az_rad)
    z = radius * math.sin(el_rad)

    cam.location = (x, y, z)

    # Point camera at origin
    direction = cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    # Render
    filepath = f"{OUTPUT_DIR}/multiview/{i:03d}.png"
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

    cameras.append({
        "id": i,
        "azimuth": azimuth,
        "elevation": elevation,
        "image": f"multiview/{i:03d}.png",
        "camera_matrix": [list(row) for row in cam.matrix_world]
    })
    print(f"Rendered {i+1}/{N_VIEWS}: azimuth={azimuth:.1f}")

# Save cameras.json
with open(f"{OUTPUT_DIR}/cameras.json", "w") as f:
    json.dump({"views": cameras}, f, indent=2)

print(f"Done. Saved {N_VIEWS} views to {OUTPUT_DIR}")
EOF

# ── data/scripts/zero123pp_infer.py ──────────────────────────
cat > data/scripts/zero123pp_infer.py << 'EOF'
"""
Zero123++ batch multiview generation.
Usage: python data/scripts/zero123pp_infer.py \
           --input_dir data/mj_images/ \
           --output_dir data/zero123_out/
"""
import argparse
import os
import json
from pathlib import Path
from PIL import Image
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_views", type=int, default=12)
    return p.parse_args()

# Zero123++ fixed output azimuths (6-view tiled layout)
AZIMUTHS  = [  30,  90, 150, 210, 270, 330,
               30,  90, 150, 210, 270, 330]
ELEVATIONS = [  20,  20,  20,  20,  20,  20,
               -10, -10, -10, -10, -10, -10]

def load_pipeline(device):
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing='trailing'
    )
    pipe.to(device)
    return pipe

def remove_background(image: Image.Image) -> Image.Image:
    """Simple rembg-based background removal."""
    try:
        from rembg import remove
        return remove(image)
    except ImportError:
        print("rembg not installed, skipping background removal")
        return image

def run(args):
    pipe = load_pipeline(args.device)
    input_paths = sorted(Path(args.input_dir).glob("*.png")) + \
                  sorted(Path(args.input_dir).glob("*.jpg"))

    print(f"Found {len(input_paths)} input images")

    for idx, img_path in enumerate(input_paths):
        sample_id = f"{idx:03d}"
        out_dir = Path(args.output_dir) / sample_id
        views_dir = out_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx+1}/{len(input_paths)}] Processing {img_path.name}")

        # Load + preprocess
        img = Image.open(img_path).convert("RGBA")
        img = remove_background(img)
        img_rgb = Image.new("RGB", img.size, (255, 255, 255))
        img_rgb.paste(img, mask=img.split()[3])

        # Copy input
        img_rgb.save(out_dir / "input.png")

        # Generate 6-view tiled output
        result = pipe(img_rgb, num_inference_steps=75).images[0]

        # Split tiled image (3x2 grid, each 320x320)
        w, h = result.size
        tile_w, tile_h = w // 3, h // 2
        cameras = []

        view_idx = 0
        for row in range(2):
            for col in range(3):
                box = (col*tile_w, row*tile_h, (col+1)*tile_w, (row+1)*tile_h)
                view = result.crop(box).resize((512, 512))
                fname = f"views/{view_idx:03d}.png"
                view.save(out_dir / fname)
                cameras.append({
                    "id": view_idx,
                    "azimuth":   AZIMUTHS[view_idx],
                    "elevation": ELEVATIONS[view_idx],
                    "image": fname
                })
                view_idx += 1

        with open(out_dir / "cameras.json", "w") as f:
            json.dump({"views": cameras}, f, indent=2)

        print(f"  Saved 6 views to {out_dir}")

    print("Done.")

if __name__ == "__main__":
    run(parse_args())
EOF

# ── data/scripts/compute_sdf.py ──────────────────────────────
cat > data/scripts/compute_sdf.py << 'EOF'
"""
Compute GT SDF values from a mesh file.
Usage: python data/scripts/compute_sdf.py \
           --mesh_path data/laukry.glb \
           --output data/cache/laukry_sdf.npy \
           --n_points 100000
"""
import argparse
import numpy as np
import trimesh
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_points", type=int, default=100000)
    p.add_argument("--bbox_scale", type=float, default=1.2)
    return p.parse_args()

def load_and_normalize_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    # Normalize to unit sphere
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.apply_scale(1.0 / scale)
    print(f"Loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh

def compute_sdf(mesh, n_points, bbox_scale):
    try:
        from pysdf import SDF
        sdf_fn = SDF(mesh.vertices, mesh.faces)

        # Sample points: 70% near surface, 30% random in bbox
        n_surface = int(n_points * 0.7)
        n_random  = n_points - n_surface

        # Near-surface points (perturbed)
        surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
        noise = np.random.randn(*surface_pts.shape) * 0.02
        surface_pts = surface_pts + noise

        # Random bbox points
        random_pts = np.random.uniform(-bbox_scale, bbox_scale, (n_random, 3))

        points = np.vstack([surface_pts, random_pts]).astype(np.float32)
        sdf_vals = sdf_fn(points).astype(np.float32)

        return points, sdf_vals

    except ImportError:
        raise ImportError("pysdf not installed. Run: pip install pysdf")

def run(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mesh = load_and_normalize_mesh(args.mesh_path)
    points, sdf_vals = compute_sdf(mesh, args.n_points, args.bbox_scale)

    # Compute normals at surface points
    n_surface = int(args.n_points * 0.7)
    _, _, normals = trimesh.proximity.closest_point(mesh, points[:n_surface])

    np.save(args.output, {
        "points": points,
        "sdf":    sdf_vals,
        "normals_surface_idx": n_surface,
        "normals": normals.astype(np.float32)
    })
    print(f"Saved SDF cache: {args.output}")
    print(f"  Points: {len(points)}, SDF range: [{sdf_vals.min():.4f}, {sdf_vals.max():.4f}]")

if __name__ == "__main__":
    run(parse_args())
EOF

# ── data/scripts/build_dataset.py ────────────────────────────
cat > data/scripts/build_dataset.py << 'EOF'
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
EOF

# ── triposg_finetune/__init__.py ─────────────────────────────
touch triposg_finetune/__init__.py

# ── triposg_finetune/dataset.py ──────────────────────────────
cat > triposg_finetune/dataset.py << 'EOF'
"""
PyTorch Dataset for TripoSG finetuning.
Each sample: (input_image, sample_points, gt_sdf, gt_normals)
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class ChibiDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 512,
                 n_sample_points: int = 512, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.n_sample_points = n_sample_points
        self.augment = augment

        # Collect all samples that have views/
        self.samples = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "input.png").exists()
        ])
        print(f"Dataset: {len(self.samples)} samples from {data_dir}")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.augment:
            img = T.RandomHorizontalFlip(p=0.5)(img)
            img = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(img)
        return self.transform(img)

    def _sample_sdf(self, sample_dir: Path):
        """Load precomputed SDF cache or return zeros if not available."""
        sdf_path = sample_dir / "sdf_cache.npy"
        if sdf_path.exists():
            data = np.load(sdf_path, allow_pickle=True).item()
            points  = data["points"]
            sdf     = data["sdf"]
            n_surf  = data["normals_surface_idx"]
            normals_surf = data["normals"]

            # Random subsample
            idx = np.random.choice(len(points), self.n_sample_points, replace=False)
            pts = torch.tensor(points[idx], dtype=torch.float32)
            sdf_vals = torch.tensor(sdf[idx], dtype=torch.float32)

            # Normals: only valid for surface points
            surf_mask = idx < n_surf
            normals = torch.zeros(self.n_sample_points, 3)
            surf_idx = np.where(surf_mask)[0]
            if len(surf_idx) > 0:
                normals[surf_idx] = torch.tensor(
                    normals_surf[idx[surf_mask]], dtype=torch.float32)

            return pts, sdf_vals, normals, torch.tensor(surf_mask)
        else:
            # No SDF cache: return dummy (Zero123++ samples don't have mesh GT)
            pts = torch.rand(self.n_sample_points, 3) * 2 - 1
            return pts, torch.zeros(self.n_sample_points), \
                   torch.zeros(self.n_sample_points, 3), \
                   torch.zeros(self.n_sample_points, dtype=torch.bool)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Input image (use input.png or random view as input)
        views_dir = sample_dir / "views"
        if views_dir.exists() and random.random() < 0.5:
            view_imgs = list(views_dir.glob("*.png"))
            img_path = random.choice(view_imgs) if view_imgs else sample_dir / "input.png"
        else:
            img_path = sample_dir / "input.png"

        image = self._load_image(img_path)
        points, sdf_vals, normals, surf_mask = self._sample_sdf(sample_dir)

        return {
            "image":     image,          # [3, H, W]
            "points":    points,         # [N, 3]
            "sdf":       sdf_vals,       # [N]
            "normals":   normals,        # [N, 3]
            "surf_mask": surf_mask,      # [N] bool
            "sample_id": sample_dir.name
        }
EOF

# ── triposg_finetune/loss.py ─────────────────────────────────
cat > triposg_finetune/loss.py << 'EOF'
"""
Training losses for TripoSG finetuning.

L_total = L_SDF + lambda_normal * L_normal + lambda_eikonal * L_eikonal

References:
  - TripoSG paper (arXiv 2502.06608) Section 3.2
  - Eikonal loss: Gropp et al. "Implicit Geometric Regularization" (2020)
"""
import torch
import torch.nn.functional as F


def loss_sdf(pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> torch.Tensor:
    """
    L1 loss between predicted and ground-truth SDF values.

    Args:
        pred_sdf: [B, N] predicted SDF values
        gt_sdf:   [B, N] ground-truth SDF values
    Returns:
        scalar loss
    """
    return F.l1_loss(pred_sdf, gt_sdf)


def loss_normal(pred_sdf: torch.Tensor, points: torch.Tensor,
                gt_normals: torch.Tensor, surf_mask: torch.Tensor) -> torch.Tensor:
    """
    Surface normal consistency loss.
    Computes predicted normals via autograd (gradient of SDF w.r.t. points).
    Only computed at surface points (surf_mask == True).

    Args:
        pred_sdf:   [B, N] predicted SDF (must be computed with points.requires_grad=True)
        points:     [B, N, 3] input points (requires_grad=True)
        gt_normals: [B, N, 3] ground-truth normals (zero for non-surface points)
        surf_mask:  [B, N] bool, True for surface points
    Returns:
        scalar loss (0 if no surface points in batch)
    """
    if not surf_mask.any():
        return torch.tensor(0.0, device=pred_sdf.device, requires_grad=True)

    # Compute gradient of SDF w.r.t. points
    grad = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True
    )[0]  # [B, N, 3]

    pred_normals = F.normalize(grad, dim=-1)  # [B, N, 3]

    # Only compute loss at surface points
    mask = surf_mask.unsqueeze(-1).expand_as(pred_normals)  # [B, N, 3]
    pred_n = pred_normals[mask].view(-1, 3)
    gt_n   = gt_normals[mask].view(-1, 3)

    # Cosine distance
    cos_sim = (pred_n * gt_n).sum(dim=-1)  # [-1, 1]
    return (1.0 - cos_sim).mean()


def loss_eikonal(pred_sdf: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Eikonal regularization: ||∇SDF|| = 1 everywhere.

    Args:
        pred_sdf: [B, N] predicted SDF
        points:   [B, N, 3] input points (requires_grad=True)
    Returns:
        scalar loss
    """
    grad = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True
    )[0]  # [B, N, 3]

    grad_norm = grad.norm(dim=-1)  # [B, N]
    return ((grad_norm - 1.0) ** 2).mean()


class TotalLoss(torch.nn.Module):
    def __init__(self, lambda_normal: float = 0.1, lambda_eikonal: float = 0.05):
        super().__init__()
        self.lambda_normal   = lambda_normal
        self.lambda_eikonal  = lambda_eikonal

    def forward(self, pred_sdf, points, gt_sdf, gt_normals, surf_mask):
        l_sdf  = loss_sdf(pred_sdf, gt_sdf)
        l_norm = loss_normal(pred_sdf, points, gt_normals, surf_mask)
        l_eik  = loss_eikonal(pred_sdf, points)

        total = l_sdf + self.lambda_normal * l_norm + self.lambda_eikonal * l_eik

        return total, {
            "loss/total":    total.item(),
            "loss/sdf":      l_sdf.item(),
            "loss/normal":   l_norm.item(),
            "loss/eikonal":  l_eik.item()
        }
EOF

# ── triposg_finetune/utils.py ────────────────────────────────
cat > triposg_finetune/utils.py << 'EOF'
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
EOF

# ── triposg_finetune/trainer.py ──────────────────────────────
cat > triposg_finetune/trainer.py << 'EOF'
"""
Main training loop for TripoSG finetuning.
Freezes VAE, only trains DiT cross-attention layers.
"""
import os
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from .dataset import ChibiDataset
from .loss import TotalLoss
from .utils import save_checkpoint, log_metrics, get_trainable_params


def freeze_except_attention(model):
    """Freeze all params except cross-attention in DiT transformer."""
    for name, param in model.named_parameters():
        is_attn = any(k in name for k in ["attn", "cross_attention", "to_q", "to_k", "to_v", "to_out"])
        param.requires_grad = is_attn
    get_trainable_params(model)


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.training.learning_rate,
                              weight_decay=cfg.training.weight_decay)


def build_scheduler(optimizer, cfg):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    return CosineAnnealingLR(optimizer, T_max=cfg.training.num_steps,
                              eta_min=cfg.training.learning_rate * 0.1)


def train(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────
    print("Loading TripoSG...")
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    pipe = TripoSGPipeline.from_pretrained(
        cfg.model.pretrained,
        torch_dtype=torch.float16 if cfg.training.mixed_precision == "fp16" else torch.float32
    ).to(device)

    # Freeze VAE, only train DiT attention
    freeze_except_attention(pipe.transformer)

    # ── Dataset & DataLoader ───────────────────────────────
    train_ds = ChibiDataset(cfg.data.train_dir, cfg.data.image_size,
                             cfg.loss.n_sample_points, augment=True)
    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                          shuffle=True, num_workers=cfg.data.num_workers,
                          pin_memory=True)

    # ── Loss, optimizer, scheduler ─────────────────────────
    criterion = TotalLoss(cfg.loss.lambda_normal, cfg.loss.lambda_eikonal)
    optimizer = build_optimizer(pipe.transformer, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.training.mixed_precision == "fp16"))

    # ── Training loop ──────────────────────────────────────
    step = 0
    best_loss = float("inf")
    accumulation_steps = cfg.training.gradient_accumulation_steps
    optimizer.zero_grad()

    pbar = tqdm(total=cfg.training.num_steps, desc="Training")

    while step < cfg.training.num_steps:
        for batch in train_dl:
            if step >= cfg.training.num_steps:
                break

            image     = batch["image"].to(device)       # [B, 3, H, W]
            points    = batch["points"].to(device)       # [B, N, 3]
            gt_sdf    = batch["sdf"].to(device)          # [B, N]
            gt_norm   = batch["normals"].to(device)      # [B, N, 3]
            surf_mask = batch["surf_mask"].to(device)    # [B, N]

            points.requires_grad_(True)

            with torch.cuda.amp.autocast(enabled=(cfg.training.mixed_precision == "fp16")):
                # Encode image → latent tokens
                image_embeds = pipe.encode_image(image)

                # Decode latent → SDF at sample points
                # NOTE: This interface needs to be verified against
                #       triposg/models/autoencoders/autoencoder_kl_triposg.py
                pred_sdf = pipe.vae.decode_sdf(image_embeds, points)

                loss, metrics = criterion(pred_sdf, points, gt_sdf, gt_norm, surf_mask)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    pipe.transformer.parameters(), cfg.training.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if step % cfg.training.log_every == 0:
                log_metrics(metrics, step)
                pbar.set_postfix({k.split("/")[1]: f"{v:.4f}" for k, v in metrics.items()})

            # Checkpoint
            if step % cfg.training.save_every == 0 and step > 0:
                is_best = metrics["loss/total"] < best_loss
                if is_best:
                    best_loss = metrics["loss/total"]
                save_checkpoint(pipe.transformer, optimizer, step,
                                metrics["loss/total"], cfg.training.output_dir, is_best)

            step += 1
            pbar.update(1)

    pbar.close()
    save_checkpoint(pipe.transformer, optimizer, step, best_loss,
                    cfg.training.output_dir, is_best=True)
    print("Training complete.")
EOF

# ── scripts/train.py ─────────────────────────────────────────
cat > scripts/train.py << 'EOF'
"""Training entry point."""
import argparse
import sys
sys.path.insert(0, ".")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    from triposg_finetune.trainer import train
    train(args.config)

if __name__ == "__main__":
    main()
EOF

# ── scripts/evaluate.py ──────────────────────────────────────
cat > scripts/evaluate.py << 'EOF'
"""
Evaluation: CLIP score + LPIPS between input image and rendered 3D output.
Usage: python scripts/evaluate.py --checkpoint checkpoints/best.pt --test_dir data/test/
"""
import argparse
import os
import torch
import clip
import lpips
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_dir",   required=True)
    p.add_argument("--output",     default="results/eval.json")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()

def compute_clip_score(model, preprocess, img1, img2, device):
    i1 = preprocess(img1).unsqueeze(0).to(device)
    i2 = preprocess(img2).unsqueeze(0).to(device)
    with torch.no_grad():
        f1 = model.encode_image(i1)
        f2 = model.encode_image(i2)
    f1 = f1 / f1.norm(dim=-1, keepdim=True)
    f2 = f2 / f2.norm(dim=-1, keepdim=True)
    return (f1 * f2).sum().item()

def run(args):
    device = args.device
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    transform = T.Compose([T.Resize((512,512)), T.ToTensor(),
                            T.Normalize([0.5]*3, [0.5]*3)])

    test_samples = sorted(Path(args.test_dir).iterdir())
    results = []

    for sample in test_samples:
        input_img = Image.open(sample / "input.png").convert("RGB")

        # TODO: run finetuned TripoSG inference → render → compare
        # For now, compare input vs first view (baseline)
        views = sorted((sample / "views").glob("*.png"))
        if not views:
            continue
        recon_img = Image.open(views[0]).convert("RGB")

        clip_score = compute_clip_score(clip_model, preprocess, input_img, recon_img, device)

        t1 = transform(input_img).unsqueeze(0).to(device)
        t2 = transform(recon_img).unsqueeze(0).to(device)
        lpips_score = lpips_fn(t1, t2).item()

        results.append({
            "sample": sample.name,
            "clip_score":  clip_score,
            "lpips_score": lpips_score
        })
        print(f"{sample.name}: CLIP={clip_score:.4f}  LPIPS={lpips_score:.4f}")

    avg_clip  = sum(r["clip_score"]  for r in results) / len(results)
    avg_lpips = sum(r["lpips_score"] for r in results) / len(results)
    print(f"\nAverage CLIP:  {avg_clip:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "avg_clip": avg_clip, "avg_lpips": avg_lpips}, f, indent=2)

if __name__ == "__main__":
    run(parse_args())
EOF

# ── scripts/inference.py ─────────────────────────────────────
cat > scripts/inference.py << 'EOF'
"""
Single image inference with finetuned TripoSG.
Usage: python scripts/inference.py --image input.png --checkpoint checkpoints/best.pt
"""
import argparse
import torch
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True)
    p.add_argument("--checkpoint", default=None, help="Finetuned checkpoint (optional)")
    p.add_argument("--output",     default="output.glb")
    p.add_argument("--faces",      type=int, default=10000)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()

def run(args):
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    print("Loading TripoSG pipeline...")
    pipe = TripoSGPipeline.from_pretrained(
        "VAST-AI/TripoSG", torch_dtype=torch.float16
    ).to(args.device)

    # Load finetuned weights if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        pipe.transformer.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Loaded finetuned checkpoint: {args.checkpoint}")

    # Background removal
    from rembg import remove
    image = Image.open(args.image).convert("RGBA")
    image = remove(image)
    image_rgb = Image.new("RGB", image.size, (255, 255, 255))
    image_rgb.paste(image, mask=image.split()[3])

    # Inference
    print("Running inference...")
    mesh = pipe(image_rgb, mc_resolution=256, num_inference_steps=50).meshes[0]

    # Export
    mesh.export(args.output, include_normals=True)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    run(parse_args())
EOF

# ── demo/app.py ──────────────────────────────────────────────
cat > demo/app.py << 'EOF'
"""
Gradio demo: Upload image → CircleMic style 3D mesh.
Usage: python demo/app.py
"""
import sys
sys.path.insert(0, ".")

import gradio as gr
import torch
import tempfile
import os
from PIL import Image

CHECKPOINT = "checkpoints/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline():
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    pipe = TripoSGPipeline.from_pretrained(
        "VAST-AI/TripoSG", torch_dtype=torch.float16
    ).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        pipe.transformer.load_state_dict(ckpt["model_state"], strict=False)
        print("Loaded finetuned weights.")
    return pipe

pipe = None

def generate_3d(image):
    global pipe
    if pipe is None:
        pipe = load_pipeline()

    from rembg import remove
    img_rgba = remove(image.convert("RGBA"))
    img_rgb  = Image.new("RGB", img_rgba.size, (255, 255, 255))
    img_rgb.paste(img_rgba, mask=img_rgba.split()[3])

    mesh = pipe(img_rgb, mc_resolution=256, num_inference_steps=50).meshes[0]

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        mesh.export(f.name)
        return f.name

with gr.Blocks(title="CircleMic 3D Generator") as demo:
    gr.Markdown("# CircleMic Style 3D Character Generator\nCS5788 · Cornell Tech")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input Image (MidJourney style)")
            btn = gr.Button("Generate 3D", variant="primary")
        with gr.Column():
            out = gr.Model3D(label="Output 3D Mesh")
    btn.click(generate_3d, inputs=[inp], outputs=[out])

if __name__ == "__main__":
    demo.launch(share=False)
EOF

# ── notebooks ────────────────────────────────────────────────
cat > notebooks/01_data_exploration.ipynb << 'EOF'
{
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":["# Data Exploration\nVisualize dataset samples and SDF distributions."]},
  {"cell_type":"code","metadata":{},"source":["import sys; sys.path.insert(0,'..') \nimport matplotlib.pyplot as plt\nfrom triposg_finetune.dataset import ChibiDataset\n\nds = ChibiDataset('../data/train', augment=False)\nprint(f'Train samples: {len(ds)}')\nsample = ds[0]\nprint('Keys:', list(sample.keys()))\nprint('Image shape:', sample['image'].shape)\nprint('SDF range:', sample['sdf'].min().item(), sample['sdf'].max().item())"],"outputs":[]}
 ],
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10.0"}},
 "nbformat":4,"nbformat_minor":5
}
EOF

cat > notebooks/02_loss_sanity_check.ipynb << 'EOF'
{
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":["# Loss Sanity Check\nVerify L_SDF, L_normal, L_eikonal compute correctly on dummy data."]},
  {"cell_type":"code","metadata":{},"source":["import sys; sys.path.insert(0,'..')\nimport torch\nfrom triposg_finetune.loss import TotalLoss\n\nB, N = 2, 512\npoints   = torch.rand(B, N, 3, requires_grad=True)\ngt_sdf   = torch.rand(B, N) * 2 - 1\ngt_norm  = torch.randn(B, N, 3)\ngt_norm  = gt_norm / gt_norm.norm(dim=-1, keepdim=True)\nsurf_mask= torch.rand(B, N) > 0.5\npred_sdf = (points ** 2).sum(dim=-1).sqrt() - 0.5  # sphere SDF\n\ncriterion = TotalLoss(lambda_normal=0.1, lambda_eikonal=0.05)\nloss, metrics = criterion(pred_sdf, points, gt_sdf, gt_norm, surf_mask)\nprint('Loss computed successfully:')\nfor k, v in metrics.items(): print(f'  {k}: {v:.6f}')"],"outputs":[]}
 ],
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10.0"}},
 "nbformat":4,"nbformat_minor":5
}
EOF

cat > notebooks/03_results_visualization.ipynb << 'EOF'
{
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":["# Results Visualization\nPlot loss curves and qualitative comparisons."]},
  {"cell_type":"code","metadata":{},"source":["import json\nimport matplotlib.pyplot as plt\n\nwith open('../logs/train.jsonl') as f:\n    logs = [json.loads(l) for l in f]\n\nsteps = [l['step'] for l in logs]\nloss_total  = [l['loss/total']   for l in logs]\nloss_sdf    = [l['loss/sdf']     for l in logs]\nloss_normal = [l['loss/normal']  for l in logs]\nloss_eik    = [l['loss/eikonal'] for l in logs]\n\nfig, axes = plt.subplots(2, 2, figsize=(12, 8))\nfor ax, vals, title in zip(axes.flat,\n    [loss_total, loss_sdf, loss_normal, loss_eik],\n    ['Total Loss', 'L_SDF', 'L_normal', 'L_eikonal']):\n    ax.plot(steps, vals)\n    ax.set_title(title); ax.set_xlabel('Step'); ax.grid(True)\nplt.tight_layout()\nplt.savefig('../results/loss_curves/training_curves.png', dpi=150)\nplt.show()"],"outputs":[]}
 ],
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10.0"}},
 "nbformat":4,"nbformat_minor":5
}
EOF

echo ""
echo "============================================"
echo "Project structure created successfully!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  cd $PROJECT"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial project structure'"
echo "  git remote add origin https://github.com/YOUR_USERNAME/cs5788_final_project.git"
echo "  git push -u origin main"
echo ""
echo "Then in WSL:"
echo "  conda create -n triposg_chibi python=3.10"
echo "  conda activate triposg_chibi"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
echo "  pip install -r requirements.txt"
echo "  git clone https://github.com/VAST-AI-Research/TripoSG.git"
echo "  pip install -e TripoSG/"
echo ""
echo "Sanity check loss functions (no GPU needed):"
echo "  conda activate triposg_chibi"
echo "  cd $PROJECT"
echo "  jupyter notebook notebooks/02_loss_sanity_check.ipynb"
