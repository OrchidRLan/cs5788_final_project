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
