ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# Required for RTX 5090 build
ENV TORCH_CUDA_ARCH_LIST="12.0"

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Conda config + TOS
RUN conda config --set always_yes yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environment with a modern Python
RUN conda create -n gsplat_depth python=3.10 -y

# Update environment using environment.yml
COPY environment.yml ./
RUN conda env update -n gsplat_depth -f environment.yml

ENV PATH=/opt/conda/envs/gsplat_depth/bin:$PATH

# Copy submodules needed for editable installs
WORKDIR /workspace
COPY submodules/ /workspace/submodules/

# Install PyTorch nightly for CUDA 12.8 FIRST (must be before pytorch3d and other torch-dependent packages)
# Use explicit upgrade to ensure installation completes
RUN pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify torch installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

# Install PyTorch3D (compatible with PyTorch nightly and CUDA 12.8)
# Disable build isolation so the build can see the already-installed torch.
RUN python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

# Install additional Python packages
RUN python -m pip install faiss-cpu \
    && python -m pip install scikit-optimize \
    && python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization \
    && python -m pip install --no-build-isolation -e submodules/simple-knn

