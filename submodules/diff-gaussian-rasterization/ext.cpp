/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#define TORCH_USE_ENUM_TO_INT_CAST
#include <torch/extension.h>
#include "rasterize_points.h"

// --- PYBIND11 MODULE ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Original bindings
    m.def("rasterize_gaussians", &RasterizeGaussiansCUDA,
          "Standard Gaussian rasterizer");
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA,
          "Backward pass for Gaussian rasterizer");
    m.def("mark_visible", &markVisible,
          "Mark visible 3D points");

    // Top-K contribution rasterizer
    m.def("rasterize_gaussians_with_contrib",
          &RasterizeGaussiansCUDA_WithContrib,
          "Rasterizer that returns pixel→gaussian contributor indices with top-K tracking");
}

