import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2


def _import_depth_anything(repo_path: Optional[str]):
    try:
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose
        return DepthAnything, Resize, NormalizeImage, PrepareForNet, Compose
    except Exception:
        if repo_path is None:
            repo_path = "Depth-Anything"
        repo_path = Path(repo_path)
        if repo_path.exists():
            sys.path.insert(0, str(repo_path))
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose
        return DepthAnything, Resize, NormalizeImage, PrepareForNet, Compose


def load_depth_anything_model(
    variant: str = "vitl14",
    input_size: int = 518,
    device: Optional[str] = None,
    repo_path: Optional[str] = "Depth-Anything",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    DepthAnything, Resize, NormalizeImage, PrepareForNet, Compose = _import_depth_anything(repo_path)

    transform = Compose([
        Resize(
            input_size,
            input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    hf_id = f"LiheYoung/depth_anything_{variant}"
    cwd = os.getcwd()
    try:
        repo_path = Path(repo_path) if repo_path is not None else None
        if repo_path is not None and repo_path.exists():
            os.chdir(repo_path)
        model = DepthAnything.from_pretrained(hf_id)
    finally:
        os.chdir(cwd)

    model.to(device)
    model.eval()
    return model, transform, device


@torch.no_grad()
def infer_depth_anything(
    model,
    transform,
    image: torch.Tensor,
    device: str,
    debug: bool = False,
    debug_name: str = "",
):
    """Infer depth from a CHW image tensor in [0,1]. Returns HxW tensor."""
    image = image[:3].detach().float().cpu().permute(1, 2, 0).numpy()
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    # Depth-Anything preprocessing expects RGB float in [0,1].
    if image.max() > 1.0 or image.min() < 0.0:
        image = np.clip(image, 0.0, 255.0) / 255.0
    else:
        image = np.clip(image, 0.0, 1.0)

    sample = transform({"image": image})["image"]  # CHW float
    input_batch = torch.from_numpy(sample).unsqueeze(0).to(device)

    prediction = model(input_batch)
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0).unsqueeze(0)
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)
    prediction = F.interpolate(
        prediction,
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    depth = prediction.squeeze(0).squeeze(0)
    if debug:
        finite = torch.isfinite(depth)
        if finite.any():
            vals = depth[finite]
            print(
                f"[DepthAnything][{debug_name}] raw depth tensor: "
                f"shape={tuple(depth.shape)}, dtype={depth.dtype}, "
                f"min={vals.min().item():.6f}, max={vals.max().item():.6f}, "
                f"mean={vals.mean().item():.6f}"
            )
        else:
            print(
                f"[DepthAnything][{debug_name}] raw depth tensor has no finite values; "
                f"shape={tuple(depth.shape)}, dtype={depth.dtype}"
            )
    return depth


def build_depth_anything_priors(
    cameras: List,
    variant: str = "vitl14",
    input_size: int = 518,
    device: Optional[str] = None,
    repo_path: Optional[str] = "Depth-Anything",
    normalize: bool = True,
    debug: bool = False,
):
    model, transform, device = load_depth_anything_model(
        variant=variant,
        input_size=input_size,
        device=device,
        repo_path=repo_path,
    )

    priors: List[torch.Tensor] = []
    for idx, cam in enumerate(tqdm(cameras, desc="Building Depth-Anything priors")):
        depth = infer_depth_anything(
            model,
            transform,
            cam.original_image,
            device=device,
            debug=debug and idx < 3,
            debug_name=Path(cam.image_name).stem,
        )
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        priors.append(depth.unsqueeze(0).cpu())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return priors
