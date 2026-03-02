
# 🍋🍎🍏 FRUIT3DGS 🍏🍎🍋 
## Fruit Counting and Localization with 3D Gaussian Splatting and Contrastive Learning

<p align="center">
<b>Fruit3DGS: a general fruit counting and localization framework leveraging semantic-guided 3D Gaussian Splatting and contrastive learning</b><br>
Samuele Mara, Angelo Moroncelli, Marco Maccarini, Loris Roveda
</p>

---

## 📄 Paper

**Fruit3DGS: Fruit Counting and Localization with 3D Gaussian Splatting and Contrastive Learning**  


---

## 🖼 Graphical Abstract

<p align="center">
<img src="assets/Fruit3DGS_visual_abstract.png" width="1000">
</p>

---

## 📌 Overview

Fruit3DGS is a **fruit-agnostic 3D perception framework** for:

- Fruit counting from unordered multi-view RGB images  
- Accurate 3D fruit localization  
- Robotic picking integration  
- Generalization across fruit types and orchard conditions  

Unlike fruit-specific pipelines, Fruit3DGS:

- Does **not rely on geometric fruit templates**
- Does **not require fruit-specific retraining**
- Works across apples, lemons, sunflowers, and multi-fruit scenes
- Produces metrically accurate 3D outputs  

---

## 🧠 Built on 3D Gaussian Splatting

This repository extends:

**3D Gaussian Splatting for Real-Time Radiance Field Rendering**  
Kerbl et al., ACM TOG 2023  

Official repository:  
https://github.com/graphdeco-inria/gaussian-splatting  

Please cite both the original 3DGS paper and our work.

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  volume  = {42},
  number  = {4},
  year    = {2023}
}
```

---

## 🏗 Methodology

### 1️⃣ Robot-Friendly Dataset Acquisition
- Franka EMIKA Panda  
- Intel RealSense D405 (end-effector mounted)  
- Hand–eye calibration  
- Ground-truth 3D target measurement  

### 2️⃣ Data Preparation
- COLMAP SfM initialization  
- Undistorted multi-view dataset  
- Mask generation (Grounded-SAM)  

### 3️⃣ Semantic-Aware 3DGS Training (`train_sem.py`)
- Per-Gaussian semantic coefficients  
- BCE mask supervision  
- Top-K Gaussian contributor export  
- Foreground filtering  

### 4️⃣ Instance Embedding Field Training
- Learnable embedding vector per Gaussian  
- Contrastive InfoNCE loss  
- Semantic-gated smoothness constraint  

### 5️⃣ Fruit-Agnostic Instance Clustering (`instance_embedding_clustering.py`)
- kNN graph construction  
- Embedding + geometry metric  
- DBSCAN / HDBSCAN  
- Bayesian Optimization of hyperparameters  

### 6️⃣ 3D Oriented Bounding Box Estimation
- PCA-based OBB fitting  
- 3D centroid extraction  
- JSON export for robotic picking  

---

## 📊 Quantitative Results

### 🍏 Fruit Counting Results (Apple Benchmarks)

| Dataset   | FruitNeRF | FruitLangGS | Fruit3DGS (HDBSCAN) | Fruit3DGS (Ours) | Ground Truth |
|------------|------------|-------------|----------------------|------------------|--------------|
| Tree01     | 173        | **173**     | 193                  | 187              | 179 |
| Tree02     | **112**    | 110         | 158                  | 108              | 113 |
| Tree03     | 264        | **287**     | 304                  | 273              | 291 |
| Fuji-SfM   | 1459       | 1443        | 2795                 | **1445**         | 1455 |

---

### 🌍 Generalization to Unseen Datasets

| Dataset        | FruitNeRF | Fruit3DGS (HDBSCAN) | Fruit3DGS (Ours) | Ground Truth |
|----------------|------------|----------------------|------------------|--------------|
| Lemons         | 124        | 3                    | **3**            | 3 |
| Sunflowers     | 109        | 8                    | **6**            | 5 |
| Multi-Fruit    | 18         | 8                    | **7**            | 6 |
| Brandenburg    | 0          | 1199                 | **2227**         | 3293 |

---

### 📍 Localization Results – Lemon Scene

| Pipeline              | Mean Error (cm) ↓ | Min Error (cm) ↓ | Max Error (cm) ↓ |
|------------------------|-------------------|------------------|------------------|
| FruitNeRF              | 172.1             | 164.3            | 186.1 |
| **Fruit3DGS (Ours)**   | **6.4**           | **5.4**          | **7.4** |

---

### 📍 Localization Results – Multi-Fruit Scene

| Pipeline              | Mean Error (cm) ↓ | Min Error (cm) ↓ | Max Error (cm) ↓ |
|------------------------|-------------------|------------------|------------------|
| FruitNeRF              | --                | --               | -- |
| **Fruit3DGS (Ours)**   | **3.2**           | **2.2**          | **4.9** |

---

## 🐳 Docker Setup

```bash
xhost +local:docker

docker run -it   --gpus '"device=all"'   --ipc=host   --memory=110g   --memory-swap=110g   --memory-reservation=90g   -e DISPLAY=$DISPLAY   -e NVIDIA_DRIVER_CAPABILITIES=all   -v /tmp/.X11-unix:/tmp/.X11-unix   -v <path_to_workspace>:/workspace   -p 7007:7007   --name fruit_gs   fruit_gs:latest   /bin/bash
```

---

## 🔧 Training

### Semantic Training

```bash
python train_sem.py   -s <COLMAP_dataset>   --mask_dir <mask_dir>   --topk_contrib 2   --percent_dense 0.001   --sem_threshold 0.3
```

### Instance Clustering

```bash
python instance_embedding_clustering.py   --colmap_dir <dataset>   --model_dir <model_output>   --mask_dir <semantic_masks>   --mask_inst_dir <instance_masks>   --cluster_alg dbscan
```

---

## 📦 Dataset

The Fruit3DGS localization dataset includes:

- Multi-view RGB captures  
- Camera intrinsics & extrinsics  
- Ground-truth 3D target positions  
- Robotic acquisition setup  

Repository & dataset:  
https://github.com/SamueleMara/Fruit_3DGS  

---

## 📜 Citation

```bibtex
@article{mara2026fruit3dgs,
  title   = {Fruit3DGS: Fruit Counting and Localization with 3D Gaussian Splatting and Contrastive Learning},
  author  = {Mara, Samuele and Moroncelli, Angelo and Maccarini, Marco and Roveda, Loris},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}
```

---

## 📬 Contact

Samuele Mara  
IDSIA / SUPSI / USI  
GitHub: https://github.com/SamueleMara  
