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
