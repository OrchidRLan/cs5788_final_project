
Tech stack:
- Base model: Stable Diffusion 1.5
- Training: Textual Inversion, DreamBooth + LoRA, Identity Loss (ArcFace)
- Inference: IP-Adapter FaceID, ControlNet (pose)
- Backend: FastAPI
- Frontend: HTML/CSS/JS (single page)


README should include these sections:
## 1 Project overview 
    We present a style-agnostic avatar generation framework designed for 
    brand-consistent character creation. Given a company-defined illustration 
    style (provided as a curated image dataset) and a user-uploaded portrait 
    photo, our system generates a 2D avatar image that simultaneously preserves 
    the user's facial identity and conforms to the brand's visual style and 
    canonical body structure.

    The framework is designed with two distinct roles in mind: a designer role, 
    which configures the target style by supplying training images and body 
    template poses, and an end-user role, which simply uploads a photo and 
    selects a template. Once trained, the model generalizes to any new user 
    without retraining.

    The generated 2D avatar is designed as the front-end of a downstream 3D 
    pipeline, with output structured to support VRM avatar reconstruction 
    and auto-rigging.
 
类似toB产品：
```bash
公司（一次性配置）                    用户（每次使用）
─────────────────                   ─────────────────
提供：                               提供：
- 品牌画风训练数据                    - 一张自己的照片
- 身体 template 图（多套）            - 选择 template 款式

         ↓ 训练（你们的三块代码）
         
         模型学会"这家公司的画风"
         
                  ↓ 推理
                  
                  生成：这个用户脸的
                        公司画风 avatar 2D图
                        
                            ↓ 后接 pipeline
                            
                            VRM 3D avatar
```

## 2 Method overview (briefly explain the 3 training modules and inference pipeline)
Our pipeline is designed for two roles: a **designer** who configures the 
target style once, and an **end-user** who uploads a portrait photo at 
inference time. The framework consists of three training modules and a 
multi-conditioned inference stage.

**Module 1 — Textual Inversion** (A)  
We optimize a single learnable token embedding `<style>` within SD1.5's 
text encoder, keeping all model weights frozen. Training runs on the 
designer-provided style image dataset using standard denoising loss. 
The result is a new "word" that encodes the brand's visual style and 
can be used in any prompt.

**Module 2 — DreamBooth + LoRA** (B)  
We fine-tune lightweight LoRA adapters (rank 4, applied to UNet attention 
layers) using a subject loss on the style dataset and a prior preservation 
loss on generic anime/illustration images to prevent catastrophic forgetting. 
This produces a stronger style representation than Module 1 by directly 
updating model weights, and serves as the base checkpoint for Module 3.

**Module 3 — Identity-Guided Fine-tuning** (C)  
Building on Module 2's LoRA checkpoint, we introduce an additional 
identity loss:

$$L_{id} = 1 - \cos(f(x_{gen}),\ f(x_{input}))$$

where $f$ is a frozen ArcFace model. This encourages the model to generate 
faces that structurally resemble a given input portrait, bridging brand-style 
transfer and per-user identity preservation.

**Inference Pipeline**  
At inference time, the end-user uploads a single portrait photo. Three 
conditioning signals are applied simultaneously to SD1.5:
- **Style**: Module 2/3 LoRA weights encoding the designer-defined visual style
- **Identity**: IP-Adapter FaceID injects the user's ArcFace face embedding 
  into cross-attention layers
- **Structure**: ControlNet conditioned on an OpenPose skeleton extracted 
  from the designer-provided body template (selectable by the end-user 
  at runtime)

The three conditioning scales (α, β, γ) are tunable at inference time.
The output is a fully generated 2D avatar image — not a composite — 
intended as input to a downstream VRM reconstruction and auto-rigging 
pipeline.

## 3 Repository structure 
cs5788_final_project/
│
├── README.md
├── requirements.txt
├── .gitignore                         # checkpoints/, data/raw/
│
├── data/
│   ├── raw/                           # Designer-provided style images (gitignored)
│   ├── processed/                     # Preprocessed images (512x512, captioned)
│   ├── class_images/                  # Prior preservation images (generic illustrations)
│   └── template/
│       ├── previews/                  # Template thumbnails shown to end-user in UI
│       └── poses/                     # Pre-extracted OpenPose skeletons for ControlNet
│
├── training/
│   ├── textual_inversion.py           # Module 1: learns <style> token embedding
│   ├── dreambooth_lora.py             # Module 2: LoRA fine-tuning with prior preservation
│   ├── identity_loss.py               # Module 3: identity-guided fine-tuning (ArcFace)
│   └── train_config.yaml             # Shared hyperparameters for all training modules
│
├── checkpoints/                       # Training outputs (gitignored)
│   ├── style_token.pt                 # Module 1 output
│   ├── style_lora.safetensors         # Module 2 output
│   └── style_id_lora.safetensors      # Module 3 output
│
├── inference/
│   ├── pipeline.py                    # Joint inference: LoRA + FaceID + ControlNet
│   ├── face_extractor.py              # ArcFace embedding extraction from user photo
│   └── controlnet_utils.py           # Template loading and OpenPose preprocessing
│
├── postprocess/                       # Downstream pipeline (optional for course demo)
│   ├── reconstruct_3d.py              # 2D avatar → 3D mesh (CharacterGen)
│   └── rigging.py                     # 3D mesh → VRM (UniRig)
│
├── evaluation/
│   ├── eval_identity.py               # ArcFace cosine similarity
│   ├── eval_style.py                  # CLIP score vs style prompt
│   └── eval_structure.py             # OpenPose keypoint error vs template
│
└── frontend/
    ├── app.py                         # FastAPI backend
    ├── static/
    └── templates/
        └── index.html                 # Single-page UI: upload → select → generate → download

data/ — 原始图、处理后图、class images、模板预览和姿态
training/ — 三个训练脚本 + 配置文件
checkpoints/ — 模型权重存放目录
inference/ — 推理 pipeline 及工具脚本
postprocess/ — 3D 重建与绑骨脚本
evaluation/ — 身份/风格/结构评估脚本
frontend/ — FastAPI 后端、静态资源、HTML 模板


## 4 Setup Instructions
### Prerequisites
- WSL2 (Ubuntu 20.04+) or Linux
- CUDA 12.4+
- conda

---

### 1. Clone the Repository

```bash
git clone https://github.com/OrchidRLan/cs5788_final_project.git
cd cs5788_final_project
```

### 2. Create Conda Environment

```bash
conda create -n avatar python=3.10 -y
conda activate avatar
pip install -r requirements.txt
```

### 3. Download Base Models

All base models are loaded from HuggingFace automatically on first run.  
To pre-download manually:

```bash
# Stable Diffusion 1.5 (base model)
huggingface-cli download runwayml/stable-diffusion-v1-5 \
    --local-dir checkpoints/sd15

# IP-Adapter FaceID
huggingface-cli download h94/IP-Adapter-FaceID \
    ip-adapter-faceid_sd15.bin \
    --local-dir checkpoints/ip_adapter

# ControlNet OpenPose
huggingface-cli download lllyasviel/control_v11p_sd15_openpose \
    --local-dir checkpoints/controlnet

# ArcFace (for identity loss and evaluation)
huggingface-cli download deepinsight/insightface \
    --local-dir checkpoints/arcface
```

### 4. Prepare Training Data
Place designer-provided style images in `data/raw/`, then run preprocessing:

```bash
python data/preprocess.py \
    --input_dir data/raw/ \
    --output_dir data/processed/ \
    --size 512
```

Place generic illustration images for prior preservation in `data/class_images/`.

### 5. Prepare Body Templates

Place designer-provided template images in `data/template/previews/`,  
then pre-extract OpenPose skeletons:

```bash
python inference/controlnet_utils.py \
    --input_dir data/template/previews/ \
    --output_dir data/template/poses/
```

### 6. Verify Setup

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0))
"
```

Expected output:
```
PyTorch: 2.6.0
CUDA available: True
CUDA device: NVIDIA GeForce RTX XXXX
```


5. Usage
   - Training (run order: dreambooth_lora.py first, then identity_loss.py, 
     textual_inversion.py can run in parallel)
   - Inference (python inference/pipeline.py --photo your_photo.jpg --template female_casual)
   - Evaluation (python evaluation/eval_identity.py)
   - Frontend (uvicorn frontend/app.py)

## 6 Training Data

测试可以拿
A类：Stylebreeder https://huggingface.co/datasets/stylebreeder/stylebreeder
Danbooru SFW 512px Character Filter  https://huggingface.co/datasets/hayden-donnelly/db-sfw-512px-character-filter
B类：Stable Diffusion Regularization Images	  https://github.com/Distraict/Stable-Diffusion-Regularization-Images

The training pipeline requires two categories of images:
### Category A — Style Images (`data/raw/`)

Used to teach the model the target illustration style.  
These images are provided by the designer and define the brand's visual identity.

**Requirements:**
- 50–100 images
- All in the target illustration style
- Diverse characters (different faces, hair, gender) — do not use the same 
  character repeatedly
- Front-facing or three-quarter view preferred
- Clean or simple background
- Preprocessed to 512×512

**Content:** Character illustrations are recommended, as the inference target 
is a human avatar. However, any image that clearly exhibits the target style 
(scenes, props) is acceptable.

```
data/raw/
├── char_001.png
├── char_002.png
├── char_003.png
├── ...
└── char_080.png         # 50–100 images total, diverse characters, unified style
```

---

### Category B — Class Images (`data/class_images/`)

Used as prior preservation data during DreamBooth training to prevent the 
model from forgetting general illustration ability (catastrophic forgetting).  
These images do not need to match the target style — diversity is preferred.

**Requirements:**
- 100–200 images
- Any illustration or photographic content is acceptable
- Must NOT be in the target style
- No quality or composition constraints

**Sources:** Publicly available datasets (e.g., Danbooru, Pinterest) or 
any generic anime/illustration collection.

```
data/class_images/
├── generic_001.png
├── generic_002.png
├── generic_003.png
├── ...
└── generic_150.png      # 100–200 images, diverse styles and content
```
## 7 Model weights (note that checkpoints/ is gitignored, provide HuggingFace 
   links placeholder)

   ```markdown

### Base Models (Auto-downloaded from HuggingFace)

The following pretrained models are required and will be downloaded 
automatically on first run. To pre-download manually, see Setup Instructions.

| Model | HuggingFace Link | Used In |
|-------|-----------------|---------|
| Stable Diffusion 1.5 | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | All training modules + inference |
| IP-Adapter FaceID | [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) | Inference: identity conditioning |
| ControlNet OpenPose | [lllyasviel/control_v11p_sd15_openpose](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose) | Inference: structure conditioning |
| ArcFace (InsightFace) | [deepinsight/insightface](https://huggingface.co/deepinsight/insightface) | Module 3 training + evaluation |

---

### Trained Checkpoints (gitignored)

Checkpoints produced by our three training modules are stored in 
`checkpoints/` and are not tracked by git.

```
checkpoints/
├── sd15/                      # Base model cache
├── ip_adapter/                # IP-Adapter FaceID weights
├── controlnet/                # ControlNet OpenPose weights
├── arcface/                   # ArcFace weights
│
├── style_token.pt             # Output of Module 1 (Textual Inversion)
├── style_lora.safetensors     # Output of Module 2 (DreamBooth + LoRA)
└── style_id_lora.safetensors  # Output of Module 3 (Identity-Guided Fine-tuning)
```

To reproduce our results, run the three training modules in order:

```bash
# Module 1 and 2 can run in parallel
python training/textual_inversion.py    # → style_token.pt
python training/dreambooth_lora.py      # → style_lora.safetensors

# Module 3 depends on Module 2 output
python training/identity_loss.py        # → style_id_lora.safetensors
```

> **Note:** Pre-trained checkpoints for our Laukry-style demo will be 
> released on HuggingFace after the course concludes.  
> 🔗 Link: `[to be released]`
```

## 8 Team
   - A: Identity Loss module + Evaluation pipeline
   - B: Textual Inversion training
   - C: DreamBooth + LoRA training
## 9 References (4 papers: SD1.5, Textual Inversion, DreamBooth, ControlNet, 
   IP-Adapter, ArcFace)
