
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

_TOPK_DEPTH_SAFE_K = 11
_topk_depth_warned = False


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, separate_sh=False, override_color=None,
           use_trained_exp=False, contrib=False, K=8, render_semantic=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    
    Args:
        render_semantic: If True, also render and return semantic features
    """
    # Create zero tensor for screen-space points (for gradients)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Setup rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        topk_depth_weight=getattr(pipe, "topk_depth_weight", 0.0),
        topk_depth_sigma=getattr(pipe, "topk_depth_sigma", 1.0),
        topk_depth_sort=getattr(pipe, "topk_depth_sort", False),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Covariance or scale/rotation
    scales, rotations, cov3D_precomp = None, None, None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # SHs or precomputed colors
    shs, colors_precomp = None, None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # ---------------- RASTERIZATION ---------------- #
    safe_k = K
    if contrib:
        use_depth_topk = (getattr(pipe, "topk_depth_weight", 0.0) > 0.0) or getattr(pipe, "topk_depth_sort", False)
        if use_depth_topk and K > _TOPK_DEPTH_SAFE_K:
            safe_k = _TOPK_DEPTH_SAFE_K
            global _topk_depth_warned
            if not _topk_depth_warned:
                print(f"[WARN] topk_contrib={K} too large for depth-aware Top-K; "
                      f"clamping to {safe_k} to avoid shared-memory overflow.")
                _topk_depth_warned = True
    if separate_sh:
        if contrib:
            # Call the updated rasterizer with top-K contributor tracking
            rendered_image, radii, depth_image, contrib_indices, contrib_opacities = rasterizer(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                contrib=True,
                K=safe_k
            )
        else:
            rendered_image, radii, depth_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp
            )
    else:
        if contrib:
            # Call the updated rasterizer with top-K contributor tracking
            rendered_image, radii, depth_image, contrib_indices, contrib_opacities = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                contrib=True,
                K=safe_k
            )
        else:
            rendered_image, radii, depth_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp
            )

    # Apply exposure if training
    if use_trained_exp and rendered_image is not None:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    # Clamp and package outputs
    if rendered_image is not None:
        rendered_image = rendered_image.clamp(0, 1)
        visibility_filter = (radii > 0).nonzero() if radii is not None else None
    else:
        visibility_filter = None

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth": depth_image
    }

    # Include contributor outputs if requested
    if contrib:
        out["contrib_indices"] = contrib_indices
        out["contrib_opacities"] = contrib_opacities

    # Render semantic features if requested and available
    if render_semantic and pc.get_semantic_mask is not None:
        try:
            out_semantic, out_semantic_weight = rasterizer.render_semantic(
                means3D=means3D,
                semantic_features=pc.get_semantic_mask,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp
            )
            out["semantic"] = out_semantic
            out["semantic_weight"] = out_semantic_weight
        except Exception as e:
            print(f"[WARNING] Semantic rendering failed: {e}")

    # Return the rendering package
    return out

