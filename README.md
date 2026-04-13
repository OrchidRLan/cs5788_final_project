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
