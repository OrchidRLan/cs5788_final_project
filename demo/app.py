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
