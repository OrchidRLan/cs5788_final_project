Description: Given a single portrait photo of any person, generate a 
chibi-style avatar image that preserves the person's facial identity, with 
user-selectable body templates.

Tech stack:
- Base model: Stable Diffusion 1.5
- Training: Textual Inversion, DreamBooth + LoRA, Identity Loss (ArcFace)
- Inference: IP-Adapter FaceID, ControlNet (pose)
- Backend: FastAPI
- Frontend: HTML/CSS/JS (single page)

Repository structure:
cs5788_final_project/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── class_images/
│   └── template/
│       ├── previews/
│       └── poses/
├── training/
│   ├── textual_inversion.py
│   ├── dreambooth_lora.py
│   ├── identity_loss.py
│   └── train_config.yaml
├── checkpoints/
├── inference/
│   ├── pipeline.py
│   ├── face_extractor.py
│   └── controlnet_utils.py
├── postprocess/
│   ├── reconstruct_3d.py
│   └── rigging.py
├── evaluation/
│   ├── eval_identity.py
│   ├── eval_style.py
│   └── eval_structure.py
└── frontend/
    ├── app.py
    ├── static/
    └── templates/
        └── index.html

data/ — 原始图、处理后图、class images、模板预览和姿态
training/ — 三个训练脚本 + 配置文件
checkpoints/ — 模型权重存放目录
inference/ — 推理 pipeline 及工具脚本
postprocess/ — 3D 重建与绑骨脚本
evaluation/ — 身份/风格/结构评估脚本
frontend/ — FastAPI 后端、静态资源、HTML 模板

README should include these sections:
1. Project overview (2-3 sentences, what it does and why)
2. Method overview (briefly explain the 3 training modules and inference pipeline)
3. Repository structure (the folder tree above, with one-line explanation per folder)
4. Setup instructions (conda env, pip install requirements.txt, download base models)
5. Usage
   - Training (run order: dreambooth_lora.py first, then identity_loss.py, 
     textual_inversion.py can run in parallel)
   - Inference (python inference/pipeline.py --photo your_photo.jpg --template female_casual)
   - Evaluation (python evaluation/eval_identity.py)
   - Frontend (uvicorn frontend/app.py)
6. Training data (describe that we use 80 chibi-style images of one character, 
   plus MidJourney-generated diverse chibi images)
7. Model weights (note that checkpoints/ is gitignored, provide HuggingFace 
   links placeholder)
8. Team
   - A: Identity Loss module + Evaluation pipeline
   - B: Textual Inversion training
   - C: DreamBooth + LoRA training
9. References (4 papers: SD1.5, Textual Inversion, DreamBooth, ControlNet, 
   IP-Adapter, ArcFace)
