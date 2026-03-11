"""
Occlusion Layer Utilities for 3D Gaussian Splatting

This module leverages the Top-K contributor tracking to extract "Layered Depth" information.
It allows "semantic peeling" of the scene (e.g. rendering depth of just "Fruit" ignoring "Leaves")
to apply geometric priors to occluded regions.

Concept based on OccNeRF/OccGS.
"""

import torch
import torch.nn.functional as F


def extract_layered_depths(viewspace_points, contrib_indices, contrib_opacities):
    """
    Extract depth values for each of the Top-K layers.

    Args:
        viewspace_points: [N, 3] Tensor of gaussian points in view space (X, Y, Depth)
        contrib_indices: [H, W, K] Indices of top-K gaussians per pixel (-1 = empty)
        contrib_opacities: [H, W, K] Computed alpha of each layer

    Returns:
        layer_depths: [H, W, K] Depth of each layer (0 for empty)
        layer_alphas: [H, W, K] Opacity of each layer
    """
    H, W, K = contrib_indices.shape
    _ = (H, W, K)  # silence unused lint in minimal contexts

    # Handle invalid indices (-1)
    valid_mask = (contrib_indices >= 0)
    safe_indices = torch.clamp(contrib_indices, min=0).long()

    # Gather depth (Z coordinate from viewspace points)
    pixel_depths = viewspace_points[safe_indices, 2]

    # Mask invalid layers
    layer_depths = pixel_depths * valid_mask.float()

    # Filter opacities
    layer_alphas = contrib_opacities * valid_mask.float()

    return layer_depths, layer_alphas


def render_semantic_peeled_depth(
    target_class_id,
    semantic_mask,
    viewspace_points,
    contrib_indices,
    contrib_opacities,
    geometric_prior_mode="nearest"
):
    """
    Render a "peeled" depth map for a specific semantic class, ignoring occluders.

    Args:
        target_class_id: Int ID of the class to render (e.g. 1 for Fruit)
        semantic_mask: [N] labels or [N, C] logits per Gaussian
        viewspace_points, contrib_indices, contrib_opacities: renderer outputs
        geometric_prior_mode: 'nearest' or 'weighted'

    Returns:
        peeled_depth: [H, W] Depth of the nearest surface of target_class_id
        peeled_mask: [H, W] Binary mask indicating where this class exists
    """
    H, W, K = contrib_indices.shape
    _ = (H, W, K)

    layer_depths, layer_alphas = extract_layered_depths(viewspace_points, contrib_indices, contrib_opacities)

    valid_mask = (contrib_indices >= 0)
    safe_indices = torch.clamp(contrib_indices, min=0).long()

    if semantic_mask.dim() > 1:
        gaussian_classes = torch.argmax(semantic_mask, dim=1)
    else:
        gaussian_classes = semantic_mask

    layer_classes = gaussian_classes[safe_indices]  # [H, W, K]
    is_target = (layer_classes == target_class_id) & valid_mask

    if geometric_prior_mode == "nearest":
        max_dist = 1000.0
        target_depths = torch.where(
            is_target,
            layer_depths,
            torch.tensor(max_dist, device=layer_depths.device)
        )
        peeled_depth, _ = torch.min(target_depths, dim=2)
        peeled_mask = (peeled_depth < max_dist).float()
        peeled_depth = peeled_depth * peeled_mask
    elif geometric_prior_mode == "weighted":
        weights = layer_alphas * is_target.float()
        total_weight = weights.sum(dim=2, keepdim=True) + 1e-6
        peeled_depth = (layer_depths * weights).sum(dim=2) / total_weight.squeeze(2)
        peeled_mask = (weights.sum(dim=2) > 0.1).float()
    else:
        raise ValueError(f"Unsupported geometric_prior_mode: {geometric_prior_mode}")

    return peeled_depth, peeled_mask


def occlusion_regularization_loss(peeled_depth, peeled_mask, target_prior_depth=None):
    """
    Penalize hidden geometry.

    1) Smoothness loss on peeled depth (TV) within peeled_mask
    2) Optional prior consistency to a target depth map
    """
    loss = torch.tensor(0.0, device=peeled_depth.device)

    if peeled_mask.sum() > 0:
        d_dx = torch.abs(peeled_depth[:, 1:] - peeled_depth[:, :-1])
        d_dy = torch.abs(peeled_depth[1:, :] - peeled_depth[:-1, :])

        mask_dx = peeled_mask[:, 1:] * peeled_mask[:, :-1]
        mask_dy = peeled_mask[1:, :] * peeled_mask[:-1, :]

        smooth_loss = (d_dx * mask_dx).mean() + (d_dy * mask_dy).mean()
        loss += smooth_loss * 0.1

    if target_prior_depth is not None:
        valid = (target_prior_depth > 0) & (peeled_mask > 0)
        if valid.sum() > 0:
            diff = torch.abs(peeled_depth - target_prior_depth)
            loss += (diff * valid.float()).mean()

    return loss


def occlusion_order_loss(front_depth, front_mask, back_depth, back_mask, epsilon=0.01):
    """
    Penalize background objects appearing in front of foreground objects.

    If both exist along a ray and depth(back) < depth(front) - epsilon, apply penalty.
    """
    overlap = (front_mask > 0.5) & (back_mask > 0.5)
    if overlap.sum() == 0:
        return torch.tensor(0.0, device=front_depth.device)

    violation = F.relu((front_depth - epsilon) - back_depth)
    loss = (violation * overlap.float()).mean()
    return loss
