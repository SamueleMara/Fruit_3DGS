#
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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass
try:
    from diff_gaussian_rasterization import binary_mask_render_loss_cuda as _binary_mask_render_loss_cuda
except Exception:
    _binary_mask_render_loss_cuda = None


# -----------------------------
# FULL TRAINING LOSSES
# -----------------------------

# -----------------------------
# Appareance rendering losses
# -----------------------------

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

# -------------------------------------
# Binary mask render loss (weighted)
# -------------------------------------
def binary_mask_render_loss(gaussians_mask, contrib_indices, contrib_opacities, gt_mask, alpha_mask=None, pos_weight=1.0):
    """
    Compute differentiable binary mask loss for Gaussian contributions,
    optionally weighted by alpha_mask (e.g., mask-center weights).

    Args:
        gaussians_mask: [num_gaussians] learnable scalar per Gaussian (0..1)
        contrib_indices: [H, W, K] top-K Gaussian indices per pixel
        contrib_opacities: [H, W, K] alpha values of Gaussians per pixel
        gt_mask: [H, W] binary ground-truth mask (0 or 1)
        alpha_mask: optional [H, W] float mask to limit loss region

    Returns:
        scalar loss
    """
    H, W, K = contrib_indices.shape
    device = gaussians_mask.device

    valid_mask = (contrib_indices >= 0)
    safe_indices = torch.clamp(contrib_indices, min=0).long()

    # Treat semantic field as logits and optimize through probabilities.
    # This keeps training numerically stable and consistent with semantic CUDA rendering,
    # which also applies a sigmoid to semantic values.
    gaussians_prob = torch.sigmoid(gaussians_mask)
    f_i = gaussians_prob[safe_indices]  # [H, W, K]
    f_i = f_i * valid_mask.float()
    contrib_opacities = contrib_opacities * valid_mask.float()

    # Front-to-back alpha compositing
    alpha_prod = torch.cumprod(1.0 - contrib_opacities, dim=2)
    alpha_prod = torch.cat([torch.ones((H, W, 1), device=device), alpha_prod[:, :, :-1]], dim=2)
    F_rendered = (f_i * contrib_opacities * alpha_prod).sum(dim=2)  # [H, W]

    F_rendered = torch.clamp(F_rendered, 0.0, 1.0)

    if alpha_mask is not None:
        if alpha_mask.ndim == 3:
            alpha_mask = alpha_mask.squeeze(0)
        F_rendered = F_rendered * alpha_mask

    # Ensure same shape as gt_mask
    if F_rendered.shape != gt_mask.shape:
        F_rendered = F_rendered.squeeze(0)

    weight = None
    if pos_weight is not None and pos_weight != 1.0:
        weight = torch.ones_like(gt_mask, dtype=F_rendered.dtype, device=F_rendered.device)
        weight = weight + (pos_weight - 1.0) * gt_mask.float()
    if alpha_mask is not None:
        weight = alpha_mask if weight is None else (weight * alpha_mask)

    loss = F.binary_cross_entropy(F_rendered, gt_mask.float(), weight=weight)
    return loss





# -----------------------------
# CLUSTERING LOSSES
# -----------------------------


# -------------------------
# Utilities
# -------------------------
def safe_prob(p, eps=1e-8):
    return torch.clamp(p, eps, 1.0)

def softmax_logits(u, temperature=1.0):
    if temperature != 1.0:
        return F.softmax(u / temperature, dim=1)
    return F.softmax(u, dim=1)

# -----------------------------
# Compute mask-center weights
# -----------------------------
def compute_mask_center_weights(mask_tensor):
    """
    Given a mask [H, W] (binary), compute a weight map emphasizing center pixels.
    Returns float tensor [H, W] with values in [0,1].
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np

    mask_np = mask_tensor.cpu().numpy().astype(np.bool_)
    dist_map = distance_transform_edt(mask_np)
    if dist_map.max() > 0:
        dist_map = dist_map / dist_map.max()  # normalize to 0-1
    return torch.from_numpy(dist_map).float().to(mask_tensor.device)


# -------------------------
# Losses
# -------------------------
def loss_label_ce(p, y, mask, weight=None, eps=1e-8):
    # Convert logits to safe probs (avoid log(0))
    p = safe_prob(p, eps)

    # Standard cross-entropy: −Σ y * log(p) over classes
    ce = -(y * torch.log(p)).sum(dim=1)

    # Optional per-sample weighting
    if weight is not None:
        ce = ce * weight

    # Mask out invalid samples
    ce = ce * mask
    return ce.mean()


def loss_pairwise_symmetric_kl(p, pairs_j, pairs_k, weights=None, eps=1e-8):
    # Probabilities for paired indices
    p_j = safe_prob(p[pairs_j], eps)
    p_k = safe_prob(p[pairs_k], eps)

    # KL(p_j || p_k)
    kl_jk = (p_j * (torch.log(p_j) - torch.log(p_k))).sum(dim=1)

    # KL(p_k || p_j)
    kl_kj = (p_k * (torch.log(p_k) - torch.log(p_j))).sum(dim=1)

    # Symmetric KL = KL(j,k) + KL(k,j)
    loss = kl_jk + kl_kj

    # Optional pair weights
    if weights is not None:
        loss = loss * weights

    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=p.device)


def loss_propagation(p, A):
    # (I - A) p enforces consistency with affinity graph
    I_minus_A = torch.eye(A.size(0), device=p.device) - A
    diff = I_minus_A @ p

    # L2 norm per node averaged
    return (diff * diff).sum(dim=1).mean()


def loss_smoothness(q, Kmat):
    # Pairwise differences: [G,G,K]
    diff = q.unsqueeze(1) - q.unsqueeze(0)

    # Squared L2 distance per pair
    dist2 = (diff * diff).sum(dim=2)

    # Weighted by kernel matrix Kmat
    return (Kmat * dist2).mean()


def loss_marginal_entropy(p, eps=1e-8):
    # Compute class marginals m_k = 1/N Σ p_ik
    m = safe_prob(p.mean(dim=0), eps)

    # Σ m log m (negative entropy)
    return (m * torch.log(m)).sum()


def binary_mask_render_loss(gaussians_mask, contrib_indices, contrib_opacities, gt_mask, alpha_mask=None, pos_weight=1.0):
    # Prefer fully CUDA implementation when available (forward + backward in extension).
    use_cuda_impl = (
        _binary_mask_render_loss_cuda is not None
        and gaussians_mask.is_cuda
        and contrib_indices.is_cuda
        and contrib_opacities.is_cuda
        and gt_mask.is_cuda
    )
    if use_cuda_impl:
        # Match CUDA semantic rendering behavior: semantics are raw logits that are
        # transformed with sigmoid before compositing.
        gaussians_prob = torch.sigmoid(gaussians_mask)
        alpha = alpha_mask
        if alpha is not None and alpha.ndim == 3:
            alpha = alpha.squeeze(0)
        return _binary_mask_render_loss_cuda(
            gaussians_prob,
            contrib_indices,
            contrib_opacities,
            gt_mask,
            alpha_mask=alpha,
            pos_weight=1.0 if pos_weight is None else float(pos_weight),
        )

    # Fallback (pure PyTorch implementation)
    H, W, K = contrib_indices.shape
    device = gaussians_mask.device
    valid_mask = (contrib_indices >= 0)
    safe_indices = torch.clamp(contrib_indices, min=0).long()
    gaussians_prob = torch.sigmoid(gaussians_mask)
    f_i = gaussians_prob[safe_indices]
    f_i = f_i * valid_mask.float()
    contrib_opacities = contrib_opacities * valid_mask.float()
    alpha_prod = torch.cumprod(1.0 - contrib_opacities, dim=2)
    alpha_prod = torch.cat([torch.ones((H, W, 1), device=device), alpha_prod[:, :, :-1]], dim=2)
    F_rendered = (f_i * contrib_opacities * alpha_prod).sum(dim=2)
    F_rendered = torch.clamp(F_rendered, 0.0, 1.0)

    if alpha_mask is not None:
        if alpha_mask.ndim == 3:
            alpha_mask = alpha_mask.squeeze(0)
        F_rendered = F_rendered * alpha_mask

    if F_rendered.shape != gt_mask.shape:
        F_rendered = F_rendered.squeeze(0)

    weight = None
    if pos_weight is not None and pos_weight != 1.0:
        weight = torch.ones_like(gt_mask, dtype=F_rendered.dtype, device=F_rendered.device)
        weight = weight + (pos_weight - 1.0) * gt_mask.float()
    if alpha_mask is not None:
        weight = alpha_mask if weight is None else (weight * alpha_mask)

    return F.binary_cross_entropy(F_rendered, gt_mask.float(), weight=weight)


# ============================================================
#               Explicit Gradients w.r.t p_j
# ============================================================

def grad_label_ce(p, y, mask, weight=None, eps=1e-8):
    # Safe probabilities
    p = safe_prob(p, eps)

    # d/dp of CE = − y / p
    g = -(y / torch.clamp(p, min=1e-8))

    if weight is not None:
        g = g * weight.unsqueeze(1)

    g = g * mask.unsqueeze(1)
    return g


def grad_pairwise_symmetric_kl(p, pairs_j, pairs_k, weights=None, eps=1e-8):
    # Safe probability
    p = safe_prob(p, eps)
    N, K = p.shape

    g = torch.zeros_like(p)

    pj = torch.clamp(p[pairs_j], min=1e-8)
    pk = torch.clamp(p[pairs_k], min=1e-8)

    # Grad of KL(pj || pk) wrt pj and pk
    grad_j = torch.log(pj) - torch.log(pk) + 1 - pk / pj
    grad_k = torch.log(pk) - torch.log(pj) + 1 - pj / pk

    # Remove NaN / inf (rare but safe)
    grad_j = torch.nan_to_num(grad_j, nan=0.0, posinf=0.0, neginf=0.0)
    grad_k = torch.nan_to_num(grad_k, nan=0.0, posinf=0.0, neginf=0.0)

    if weights is not None:
        grad_j = grad_j * weights.unsqueeze(1)
        grad_k = grad_k * weights.unsqueeze(1)

    # Accumulate gradients for each index
    g.index_add_(0, pairs_j, grad_j)
    g.index_add_(0, pairs_k, grad_k)

    # Normalize for stability
    g = g / max(1.0, pairs_j.numel())
    return g


def grad_propagation(p, A):
    # Gradient of ||(I - A)p||^2
    I_minus_A = torch.eye(A.size(0), device=p.device) - A

    # 2 (I-A)^T (I-A) p / N
    g = 2 * (I_minus_A.T @ (I_minus_A @ p)) / p.size(0)
    return g


def grad_marginal_entropy(p, eps=1e-8):
    # m_k = marginal distribution over clusters
    N, K = p.shape
    m = safe_prob(p.mean(dim=0), eps)

    # d/dp_i = (log m + 1) / N
    gm = (torch.log(m) + 1.0) / N
    return gm.unsqueeze(0).repeat(N, 1)


def grad_smoothness(q, Kmat):
    G, K = q.shape

    # diff[g1,g2] = q1 - q2
    diff = q.unsqueeze(1) - q.unsqueeze(0)

    # Gradient wrt q[g]: Σ_{g2} K[g,g2] * 2(q[g] - q[g2])
    grad = 2.0 * (Kmat.unsqueeze(2) * diff).sum(dim=1) / max(1.0, G)
    return grad


# ============================================================
#   Aggregate point-level grads → Gaussian-level grads
# ============================================================

def aggregate_gaussian_grads(r_point_idx, r_gauss_idx, r_vals, g_point, N_seg):
    """
    Aggregate gradients from points to gaussian segments.
    r_gauss_idx MUST lie in [0, N_seg).
    """
    device = g_point.device
    K = g_point.size(1)

    # Select gradient for each contributing point
    g_j_selected = g_point[r_point_idx]        # [M,K]

    # Multiply by scalar contribution r_vals per mapping
    contrib = r_vals.unsqueeze(1) * g_j_selected

    # Accumulate into per-gaussian gradient
    G_accum = torch.zeros((N_seg, K), device=device)
    G_accum.index_add_(0, r_gauss_idx, contrib)
    return G_accum


def logits_grad_from_q_grads(u, G_agg, temperature=1.0):
    # Convert NaNs/Infs to zeros for safety
    G_agg = torch.nan_to_num(G_agg, nan=0.0, posinf=0.0, neginf=0.0)

    # q = softmax(u)
    q = softmax_logits(u, temperature)

    # inner = q ⋅ G_agg (vector)
    inner = (q * G_agg).sum(dim=1, keepdim=True)

    # Softmax-Jacobian product: q * (G_agg − inner)
    grad_u = q * (G_agg - inner)
    return grad_u


# ============================================================
#          Total cluster loss + gradients pipeline
# ============================================================

def total_cluster_loss(
    gaussians,
    r_point_idx,
    r_gauss_idx,
    r_vals,
    p_j,
    q_i,
    pair_j=None,
    pair_k=None,
    pair_weights=None,
    A=None,
    Kmat=None,
    gaussians_mask=None,
    contrib_indices=None,
    contrib_opacities=None,
    gt_mask=None,
    alpha_mask=None,
    use_label_ce=True,
    use_pair_kl=False,
    use_prop=False,
    use_smooth=False,
    use_marg=False,
    use_instance_render=False,
    debug=False
):
    """
    Compute total instance-field clustering loss and the gradients w.r.t
    Gaussian logits q_i.

    p_j: point-level cluster probabilities
    q_i: gaussian-level logits
    """

    device = p_j.device
    N_seg = gaussians.get_xyz.shape[0]
    K = q_i.shape[1]

    loss_vals = {}
    total_loss = torch.tensor(0.0, device=device)

    # Gradients for point assignments (p_j)
    grad_p = torch.zeros_like(p_j)

    # Gradients for gaussian logits (q_i)
    grad_q_gauss = torch.zeros_like(q_i)

    # -------------------------
    # Label Cross-Entropy
    # -------------------------
    if use_label_ce:
        L_label = loss_label_ce(p_j, p_j, torch.ones(p_j.shape[0], device=device))
        loss_vals['label_ce'] = L_label
        total_loss += L_label

        grad_p += grad_label_ce(p_j, p_j, torch.ones(p_j.shape[0], device=device))
    else:
        loss_vals['label_ce'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Pairwise Symmetric KL
    # -------------------------
    if use_pair_kl and pair_j is not None and pair_k is not None:
        L_pair = loss_pairwise_symmetric_kl(p_j, pair_j, pair_k, pair_weights)
        loss_vals['pair_kl'] = L_pair
        total_loss += L_pair

        grad_p += grad_pairwise_symmetric_kl(p_j, pair_j, pair_k, pair_weights)
    else:
        loss_vals['pair_kl'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Graph Propagation Loss
    # -------------------------
    if use_prop and A is not None:
        L_prop = loss_propagation(p_j, A)
        loss_vals['prop'] = L_prop
        total_loss += L_prop

        grad_p += grad_propagation(p_j, A)
    else:
        loss_vals['prop'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Gaussian Logit Smoothness
    # -------------------------
    if use_smooth and Kmat is not None:
        L_smooth = loss_smoothness(q_i, Kmat)
        loss_vals['smooth'] = L_smooth
        total_loss += L_smooth

        grad_q_gauss += grad_smoothness(q_i, Kmat)
    else:
        loss_vals['smooth'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Marginal Entropy Regularizer
    # -------------------------
    if use_marg:
        L_marg = loss_marginal_entropy(p_j)
        loss_vals['marg'] = L_marg
        total_loss += L_marg

        grad_p += grad_marginal_entropy(p_j)
    else:
        loss_vals['marg'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Optional Rendering-based Loss
    # -------------------------
    if use_instance_render and gaussians_mask is not None:
        L_render = binary_mask_render_loss(
            gaussians_mask, contrib_indices, contrib_opacities, gt_mask, alpha_mask
        )
        loss_vals['instance_render'] = L_render
        total_loss += L_render

        # Rendering loss gradient contribution (placeholder = ones)
        grad_q_gauss += aggregate_gaussian_grads(
            r_point_idx, r_gauss_idx, r_vals,
            torch.ones_like(p_j),     # dummy grad
            N_seg
        )
    else:
        loss_vals['instance_render'] = torch.tensor(0.0, device=device)

    # ============================================================
    # Aggregate point-level gradients → gaussian-level gradients
    # ============================================================
    grad_q_from_p = aggregate_gaussian_grads(
        r_point_idx, r_gauss_idx, r_vals, grad_p, N_seg
    )

    # Convert aggregated q-gradients to logit-gradients through softmax jacobian
    grad_q_gauss += logits_grad_from_q_grads(q_i, grad_q_from_p)

    return total_loss, loss_vals, grad_q_gauss
