# Fetal MoAt Net — Federated Learning Framework for Privacy-Preserving Fetal Ultrasound Analysis

This repository implements a lightweight federated deep learning framework for multi-class fetal diagnostic plane classification from 2D B-mode ultrasound images, addressing the dual challenges of computational efficiency and patient data privacy in prenatal care settings. The proposed model, Fetal MoAt Net, integrates a MobileNetV2 backbone with a custom MoAt attention module, combining scaled dot-product spatial attention with a learnable residual scaling factor to enhance diagnostically relevant regions while suppressing ultrasound-specific speckle noise, all within 4.1M parameters and 1.2B FLOPs. The federated learning pipeline enables collaborative multi-institutional training without sharing raw patient data. Four aggregation strategies are benchmarked — FedAvg, FedProx, SCAFFOLD, and Contrastive Prototype FL across 5, 10, and 20 virtual clients under both IID and Non-IID (Dirichlet α = 0.5) data distributions on two public datasets: FPUS23 (4-class phantom) and FETAL_PLANES_DB (6-class real maternal-fetal). The repository is designed for direct reproducibility of the associated journal submission and as a reusable baseline toolkit for federated medical image classification research.


---

## Repository Structure

```
fetal_federated/
├── data/
│   └── dataset.py          # Dataset classes for FPUS23 and FETAL_PLANES_DB
├── models/
│   ├── moat_net.py         # Fetal MoAt Net (MobileNetV2 + MoAt attention)
│   └── baselines.py        # ResNet101, DenseNet121, VGG16, InceptionV3, Xception wrappers
├── strategies/
│   ├── fedavg.py           # FedAvg aggregation strategy
│   ├── fedprox.py          # FedProx (Li et al., MLSys 2020)
│   ├── scaffold.py         # SCAFFOLD (Karimireddy et al., ICML 2020)
│   └── proto_fl.py         # Contrastive Prototype FL (Fiorentino et al., 2025)
├── clients/
│   └── fl_client.py        # Flower FL client (shared across all strategies)
├── utils/
│   ├── partition.py        # IID / Non-IID data partitioning (Dirichlet)
│   ├── metrics.py          # Accuracy, AUC, F1, confusion matrix utilities
│   └── visualization.py    # Training curves, ROC/PR curves, Grad-CAM
├── results/                # Saved metrics, plots, checkpoints
├── scripts/
│   └── download_data.sh    # Instructions for downloading public datasets
├── train_centralized.py    # Centralized training baseline
├── train_federated.py      # Main federated training entry point
├── evaluate.py             # Standalone evaluation on saved checkpoints
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Images | Classes | Access |
|---|---|---|---|
| FPUS23 | 5,265 | 4 (AC, BPD, FL, No Plane) | https://github.com/bharathprabakaran/FPUS23 |
| FETAL_PLANES_DB | 12,400 | 6 (Brain, Abdomen, Femur, Thorax, Cervix, Other) | https://doi.org/10.5281/zenodo.3904280 |

Place downloaded datasets under:
```
data/FPUS23/         (subfolders: AC_PLANE, BPD_PLANE, FL_PLANE, NO_PLANE)
data/FETAL_PLANES_DB/ (subfolders per class label)
```

---

## Installation

```bash
git clone https://github.com/<your-username>/fetal_federated.git
cd fetal_federated
pip install -r requirements.txt
```

---

## Quick Start

### 1. Centralized Training
```bash
python train_centralized.py --dataset fpus23 --epochs 50 --batch_size 32
python train_centralized.py --dataset fetal_planes_db --epochs 50 --batch_size 32
```

### 2. Federated Training — All Strategies
```bash
# FedAvg
python train_federated.py --strategy fedavg --dataset fpus23 --num_clients 10 --rounds 100 --partition iid

# FedProx
python train_federated.py --strategy fedprox --dataset fpus23 --num_clients 10 --rounds 100 --partition non_iid --mu 0.01

# SCAFFOLD
python train_federated.py --strategy scaffold --dataset fpus23 --num_clients 10 --rounds 100 --partition non_iid

# Contrastive Prototype FL (Fiorentino et al. 2025)
python train_federated.py --strategy proto_fl --dataset fpus23 --num_clients 10 --rounds 100 --partition non_iid

# Sweep: all strategies × all client counts × both partitions
python train_federated.py --strategy all --num_clients 5 10 20 --partition iid non_iid --dataset fpus23
```

### 3. Evaluate a Saved Checkpoint
```bash
python evaluate.py --checkpoint results/fedavg_fpus23_10c_iid/best_model.pt --dataset fpus23
```

---

## Federated Learning Strategies

| Strategy | Reference | Key Mechanism |
|---|---|---|
| FedAvg | McMahan et al., 2017 | Weighted average of client updates |
| FedProx | Li et al., MLSys 2020 | Proximal regularization (μ·‖w−wᵍ‖²) on client objective |
| SCAFFOLD | Karimireddy et al., ICML 2020 | Control variates to correct client drift |
| Proto-FL | Fiorentino et al., IJCARS 2025 | SimCLR embeddings + prototype sharing for noisy labels |

---

## Model

**Fetal MoAt Net** = MobileNetV2 backbone (frozen base) + MoAt Custom Attention Block + Classifier Head.

MoAt Attention:
- Three 1×1 conv projections → Q, K, V
- Scaled dot-product attention: A = softmax(QKᵀ/√d)
- Learnable residual scaling γ
- Fout = conv₁ₓ₁(concat(F, γ·Fattention))

Parameters: **4.1M** | FLOPs: **1.2B**

---

## Results (Reported in Paper)

| Strategy | Dataset | Clients | Setting | Accuracy (%) | AUC (%) |
|---|---|---|---|---|---|
| Centralized | FPUS23 | — | — | 96.0 | 100.0 |
| FedAvg | FPUS23 | 10 | IID | 94.7 | 98.6 |
| FedProx | FPUS23 | 10 | IID | — | — |
| SCAFFOLD | FPUS23 | 10 | IID | — | — |
| Proto-FL | FPUS23 | 10 | IID | — | — |

_(Run experiments to populate remaining entries)_

---

## Citation

```bibtex
@article{veningston2025fetalmoatnet_fl,
  title={A Lightweight Federated Deep Learning Framework for Privacy-Preserving Fetal Ultrasound Analysis},
  author={Veningston, K. and Mushtaq, Gazala},
  journal={International Journal of Information Technology},
  year={2025}
}

@article{mushtaq2025fetalmoatnet,
  title={Fetal MoAt Net: a light-weight deep learning model for fetal diagnostic plane classification},
  author={Mushtaq, Gazala and Veningston, K.},
  journal={International Journal of Computers and Applications},
  doi={10.1080/1206212X.2025.2543550},
  year={2025}
}
```

---

## References

- Li T. et al., "Federated Optimization in Heterogeneous Networks (FedProx)," MLSys 2020.
- Karimireddy S.P. et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," ICML 2020.
- Fiorentino M.C. et al., "Contrastive prototype federated learning against noisy labels in fetal standard plane detection," Int J CARS, 2025.
- Jiang Y. et al., "From pretraining to privacy: Federated ultrasound foundation model," npj Digital Medicine, 2025.
- Sivasubramanian A. et al., "Efficient feature extraction using light-weight CNN attention," Phys Eng Sci Med, 2025.
- Mushtaq G. and Veningston K., "Fetal MoAt Net," Int J Comput Appl, 2025.
