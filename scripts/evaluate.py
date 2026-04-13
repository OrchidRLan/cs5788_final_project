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
