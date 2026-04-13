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
