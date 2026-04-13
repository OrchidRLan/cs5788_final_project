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
