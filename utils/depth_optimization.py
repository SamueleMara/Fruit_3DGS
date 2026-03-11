import torch
import torch.nn.functional as F

from utils.depth_order_loss import compute_depth_order_loss

def _to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def _resize_tensor(tensor: torch.Tensor, size, mode: str) -> torch.Tensor:
    mode = str(mode)
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        return F.interpolate(_to_nchw(tensor), size=size, mode=mode, align_corners=True).squeeze(0)
    return F.interpolate(_to_nchw(tensor), size=size, mode=mode).squeeze(0)


def compute_multiscale_depth_order_loss(*, depth, prior_depth, scene_extent, max_pixel_shift_ratio=0.05, scales=[1], normalize_loss=True, log_space=True, log_scale=20.0, mask=None):
    if not isinstance(depth, torch.Tensor):
        return 0.0

    depth_tensor = depth
    prior_tensor = prior_depth

    if depth_tensor.dim() < 2 or prior_tensor.dim() < 2:
        return depth_tensor.sum() * 0.0

    height, width = depth_tensor.shape[-2:]
    losses = []
    for scale in scales:
        scale = int(scale)
        if scale <= 1:
            depth_s = depth_tensor
            prior_s = prior_tensor
            mask_s = mask
        else:
            size = (max(1, height // scale), max(1, width // scale))
            depth_s = _resize_tensor(depth_tensor, size=size, mode="bilinear")
            prior_s = _resize_tensor(prior_tensor, size=size, mode="bilinear")
            if mask is not None:
                mask_s = _resize_tensor(mask, size=size, mode="nearest")
                if mask_s.dim() == 3 and mask_s.shape[0] == 1:
                    mask_s = mask_s.squeeze(0)
            else:
                mask_s = None

        loss = compute_depth_order_loss(
            depth=depth_s,
            prior_depth=prior_s,
            scene_extent=scene_extent,
            max_pixel_shift_ratio=max_pixel_shift_ratio,
            normalize_loss=normalize_loss,
            log_space=log_space,
            log_scale=log_scale,
            reduction="mean",
            mask=mask_s,
        )
        losses.append(loss)

    if not losses:
        return depth_tensor.sum() * 0.0

    if isinstance(losses[0], torch.Tensor):
        return torch.stack(losses).mean()
    return float(sum(losses) / len(losses))


def compute_depth_gradient_smoothness(depth, mask=None, edge_aware_weight=None, lambda_depth_grad=0.0):
    if not isinstance(depth, torch.Tensor):
        return 0.0

    depth_map = depth.squeeze()
    if depth_map.dim() != 2:
        return depth_map.sum() * 0.0

    dx = torch.abs(depth_map[:, 1:] - depth_map[:, :-1])
    dy = torch.abs(depth_map[1:, :] - depth_map[:-1, :])

    if mask is not None:
        mask_map = mask.squeeze().to(depth_map.device)
        mask_x = mask_map[:, 1:] * mask_map[:, :-1]
        mask_y = mask_map[1:, :] * mask_map[:-1, :]
        dx = dx * mask_x
        dy = dy * mask_y

    if edge_aware_weight is not None:
        edge_map = edge_aware_weight.squeeze().to(depth_map.device).clamp(0.0, 1.0)
        edge_x = (1.0 - edge_map[:, 1:]) * (1.0 - edge_map[:, :-1])
        edge_y = (1.0 - edge_map[1:, :]) * (1.0 - edge_map[:-1, :])
        dx = dx * edge_x
        dy = dy * edge_y

    loss = 0.5 * (dx.mean() + dy.mean())
    return loss * float(lambda_depth_grad)


def compute_depth_magnitude_consistency(rendered_depth, prior_depth, scene_extent, lambda_magnitude=0.0, mask=None):
    if not isinstance(rendered_depth, torch.Tensor):
        return 0.0

    depth_map = rendered_depth.squeeze()
    prior_map = prior_depth.squeeze()
    if depth_map.shape != prior_map.shape:
        return depth_map.sum() * 0.0

    extent = float(scene_extent) if scene_extent and scene_extent > 0 else 1.0
    diff = torch.abs(depth_map - prior_map) / extent

    if mask is not None:
        mask_map = mask.squeeze().to(depth_map.device)
        valid = mask_map > 0
        if valid.sum() == 0:
            return depth_map.sum() * 0.0
        diff = diff[valid]

    return diff.mean() * float(lambda_magnitude)


def compute_depth_range_loss(pred_depth, prior_depth, lambda_range=0.0, mask=None):
    if not isinstance(pred_depth, torch.Tensor):
        return 0.0

    pred_map = pred_depth.squeeze()
    prior_map = prior_depth.squeeze()
    if pred_map.shape != prior_map.shape:
        return pred_map.sum() * 0.0

    if mask is not None:
        mask_map = mask.squeeze().to(pred_map.device)
        valid = mask_map > 0
        if valid.sum() == 0:
            return pred_map.sum() * 0.0
        prior_vals = prior_map[valid]
        pred_vals = pred_map[valid]
    else:
        prior_vals = prior_map
        pred_vals = pred_map

    min_d = prior_vals.min()
    max_d = prior_vals.max()
    range_penalty = torch.relu(pred_vals - max_d) + torch.relu(min_d - pred_vals)
    return range_penalty.mean() * float(lambda_range)


def compute_adaptive_depth_weighting(*args, **kwargs):
    return 1.0


class DepthTrainingScheduler:
    def __init__(self, weight_init=1.0, weight_final=0.01, max_steps=10000, warmup_steps=0, plateau_patience=15, plateau_threshold=0.002):
        self.weight_init = weight_init
        self.weight_final = weight_final
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.best = float('inf')

    def get_weight(self, step):
        # simple linear interpolation
        t = min(max(step / max(1, self.max_steps), 0.0), 1.0)
        return self.weight_init * (1.0 - t) + self.weight_final * t

    def update_plateau_detection(self, value):
        # dummy: never boost
        return False


def estimate_depth_quality_score(*args, **kwargs):
    return 1.0
