import torch
from typing import Optional


def compute_depth_order_loss(
    depth: torch.Tensor,
    prior_depth: torch.Tensor,
    scene_extent: float = 1.0,
    max_pixel_shift_ratio: float = 0.05,
    normalize_loss: bool = True,
    log_space: bool = False,
    log_scale: float = 20.0,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None,
):
    """Compute a loss encouraging pixels to preserve relative depth ordering.

    This loss does not require accurate absolute depth scale. It only enforces
    that the sign of depth differences matches between the prediction and prior.
    """
    depth_map = depth.squeeze()
    prior_map = prior_depth.squeeze()

    if depth_map.shape != prior_map.shape:
        raise ValueError(
            f"Depth shapes do not match: {depth_map.shape} vs {prior_map.shape}"
        )

    height, width = depth_map.shape
    if height == 0 or width == 0:
        return depth_map.sum() * 0.0

    extent = float(scene_extent) if scene_extent and scene_extent > 0 else 1.0
    max_pixel_shift = max(round(max_pixel_shift_ratio * max(height, width)), 1)

    if mask is None:
        ys = torch.arange(height, device=depth_map.device)
        xs = torch.arange(width, device=depth_map.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pixel_coords = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)
    else:
        mask_map = mask.squeeze().to(depth_map.device)
        valid = mask_map > 0
        if valid.sum() == 0:
            return depth_map.sum() * 0.0
        pixel_coords = valid.nonzero(as_tuple=False)

    pixel_shifts = torch.randint(
        -max_pixel_shift,
        max_pixel_shift + 1,
        pixel_coords.shape,
        device=depth_map.device,
    )
    shifted_pixel_coords = (pixel_coords + pixel_shifts).clamp(
        min=torch.tensor([0, 0], device=depth_map.device),
        max=torch.tensor([height - 1, width - 1], device=depth_map.device),
    )

    depth_vals = depth_map[pixel_coords[:, 0], pixel_coords[:, 1]]
    shifted_depth_vals = depth_map[
        shifted_pixel_coords[:, 0], shifted_pixel_coords[:, 1]
    ]
    prior_vals = prior_map[pixel_coords[:, 0], pixel_coords[:, 1]]
    shifted_prior_vals = prior_map[
        shifted_pixel_coords[:, 0], shifted_pixel_coords[:, 1]
    ]

    if mask is not None:
        shifted_valid = mask_map[
            shifted_pixel_coords[:, 0], shifted_pixel_coords[:, 1]
        ] > 0
        if shifted_valid.sum() == 0:
            return depth_map.sum() * 0.0
        depth_vals = depth_vals[shifted_valid]
        shifted_depth_vals = shifted_depth_vals[shifted_valid]
        prior_vals = prior_vals[shifted_valid]
        shifted_prior_vals = shifted_prior_vals[shifted_valid]

    diff = (depth_vals - shifted_depth_vals) / extent
    prior_diff = (prior_vals - shifted_prior_vals) / extent

    if normalize_loss:
        prior_diff = prior_diff / prior_diff.detach().abs().clamp(min=1e-8)

    depth_order_loss = - (diff * prior_diff).clamp(max=0)
    if log_space:
        depth_order_loss = torch.log1p(log_scale * depth_order_loss)

    if reduction == "mean":
        return depth_order_loss.mean()
    if reduction == "sum":
        return depth_order_loss.sum()
    if reduction == "none":
        return depth_order_loss

    raise ValueError(f"Invalid reduction: {reduction}")
