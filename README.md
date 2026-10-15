# MiRAGE
Official implementation of “Parameter-Efficient Remote Sensing Image Scene Classification via Mixture of Recursive Cross-Attention and Cross-Modal Semantic Initialization” (MiRAGE).

## 📌 Method Overview

**MiRAGE** (*Mixture of Recursive Attention for Geographic Encoding*) is a unified parameter-efficient framework for Remote Sensing (RS) scene classification:
- **Parameter-Efficient Fine-Tuning (PEFT):** Adapts a pretrained `CLIP ViT-L/14@336` vision backbone using Low-Rank Adaptation (LoRA).
- **Dual-Head Architecture:** Employs a Global Recursive Head (GRH) to capture global context from the class token and a Spatial Recursive Head (SRH) utilizing multi-head cross-attention over patch tokens to capture local spatial textures. 
- **Mixture-of-Recursions (MoR):** A shared neural router dynamicallly determines an input-image-dependent recursion depth ($t \in [1, 4]$), to adapt the computation graph.
- **Cross-Modal Semantic Initialization (CMSI):** Transfers semantic priors by initializing head classifiers with class- and branch-specific CLIP text embeddings.
- **Inter-Classifier Consistency Loss:** Employs a symmetric Kullback-Leibler (KL) divergence loss between GRH and SRH logits to encourage mutual agreement while preserving their complementary representations.

---

## 📁 Repository Structure

```text
.
├── data/              # Benchmark dataset directories (AID, UCMerced, NWPU-RESISC45)
├── models/            # Directory to store trained checkpoints (download link below)
├── splits/            # Text files listing image names for each train/test partition
├── main.py            # Model training and multi-seed evaluation script
├── test.py            # Standalone inference and checkpoint evaluation script
├── requirements.txt   # Pinned environment dependencies
└── README.md
```

---

## 🛠️ Environment Setup

### Option 1: Primary Environment (NWPU, UCMerced, AID $T_r = 0.5$)
Tested on **Python 3.9**:

```bash
conda create -n mirage python=3.9 -y
conda activate mirage
pip install -r requirements.txt
```

### Option 2: Alternative Environment (AID $T_r = 0.2$)
If you are reproducing the AID $T_r = 0.2$ experiments under its original setup (**Python 3.10**):

```bash
conda create -n mor_sydha python=3.10 -y
conda activate mor_sydha
# Uncomment the alternative block in requirements.txt, then run:
pip install -r requirements.txt
```

---

## 💾 Dataset Preparation

Download the benchmark datasets from their original sources and extract them into the `data/` directory:

```text
data/
├── UCMerced_LandUse/
│   └── Images/
│       ├── agricultural/
│       ├── airplane/
│       └── ... (21 classes, 100 images each)
├── AID/
│   ├── Airport/
│   ├── BareLand/
│   └── ... (30 classes, 10,000 images total)
└── NWPU-RESISC45/
    ├── airplane/
    ├── airport/
    └── ... (45 classes, 700 images each)
```

> **Note on `splits/`:**  
> The `splits/` folder contains reference `.txt` files listing the exact image filenames used for each train/test split. By default, `main.py` generates reproducible stratified splits using fixed random seeds (`seed=0, 2, 4`), but the text files are provided for external reference or if you want to use the same sets of images.

---

## 🏋️ Training

Train MiRAGE using the built-in dataset presets. Each command automatically runs 3 independent runs across seeds (`0`, `2`, `4`):

### 1. UC Merced (UCM)
```bash
# 80% Train, 20% Test (Tr = 0.8)
python main.py --preset ucmerced --train_ratio 0.8 --batch_size 16 --num_runs 3 --seed 0

# 50% Train, 50% Test (Tr = 0.5)
python main.py --preset ucmerced --train_ratio 0.5 --batch_size 16 --num_runs 3 --seed 0
```

### 2. Aerial Image Dataset (AID)
```bash
# 50% Train, 50% Test (Tr = 0.5)
python main.py --preset aid --train_ratio 0.5 --batch_size 16 --num_runs 3 --seed 0

# 20% Train, 80% Test (Tr = 0.2)
python main.py --preset aid --train_ratio 0.2 --batch_size 16 --num_runs 3 --seed 0
```

### 3. NWPU-RESISC45 (NWPU)
```bash
# 80% Train, 20% Test (Tr = 0.8)
python main.py --preset nwpu --train_ratio 0.8 --batch_size 16 --num_runs 3 --seed 0

# 20% Train, 80% Test (Tr = 0.2)
python main.py --preset nwpu --train_ratio 0.2 --batch_size 16 --num_runs 3 --seed 0

# 10% Train, 90% Test (Tr = 0.1)
python main.py --preset nwpu --train_ratio 0.1 --batch_size 16 --num_runs 3 --seed 0
```

---

## 🔍 Pretrained Checkpoints & Evaluation

Pretrained weights are available for download:
- 🔗 **[Google Drive - Download Pretrained Models](https://drive.google.com/your-drive-link-here)**

Place the downloaded `.pth` checkpoint files into the `models/` directory.

To evaluate a saved model checkpoint on the test partition using hard routing ($t_h$):
```bash
python main.py \
  --preset aid \
  --eval \
  --checkpoint_path ./models/final_model_aid_seed_0.pth \
  --hard-routing-eval
```

---

## 📊 Benchmark Results

Classification accuracies (%) compared across standard train-test ratios ($T_r$) over 3 independent runs:

| Dataset | Training Ratio ($T_r$) | MiRAGE (Ours) | Prior Best | Best Competing Method |
| :--- | :---: | :---: | :---: | :--- |
| **UC Merced** ($N_c = 21$) | 0.8 | 99.60 ± 0.22 | **99.86 ± 0.10** | NAFF-HNN |
| | 0.5 | **99.49 ± 0.12** | 99.33 ± 0.32 | HFFT-PD-PVT |
| **AID** ($N_c = 30$) | 0.5 | **98.89 ± 0.02** | 98.51 ± 0.12 | LoRA-CLIP |
| | 0.2 | **98.61 ± 0.07** | 98.14 ± 0.09 | LoRA-CLIP |
| **NWPU-RESISC45** ($N_c = 45$) | 0.8 | **97.75 ± 0.11** | 97.51 ± 0.15 | SceneFormer |
| | 0.2 | **96.67 ± 0.10** | 96.27 ± 0.25 | Bai et al. (2026) |
| | 0.1 | **96.04 ± 0.05** | 95.18 ± 0.15 | LoRA-CLIP |

---

## 📜 Citation

If you find this work or code useful in your research, please cite:

```bibtex
@article{kynshi2026parameter,
  title={Parameter-efficient remote sensing image scene classification via mixture of recursive cross-attention and cross-modal semantic initialization},
  author={Kynshi, Junestarfield Lyngdoh and Borah, Kashmiri and Nag, Kaustuv},
  journal={Machine Learning with Applications},
  pages={101041},
  year={2026},
  publisher={Elsevier}
}
```
