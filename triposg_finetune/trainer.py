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
