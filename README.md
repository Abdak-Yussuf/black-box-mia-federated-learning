# Evaluating Black-Box Membership Inference Attacks in Federated Learning Systems

> **DS Lab 2** — Data Science Laboratory Project  
> **Author:** Abdihakim Yussuf Abdi  
> **Date:** March 2026

---

## Overview

This project investigates **privacy vulnerabilities in Federated Learning (FL)** systems by evaluating the effectiveness of **black-box Membership Inference Attacks (MIA)** against models trained using the Federated Averaging (FedAvg) algorithm.

A Membership Inference Attack attempts to determine whether a specific data sample was used during model training — a serious privacy threat in sensitive domains such as healthcare and finance. This study empirically measures how well such attacks succeed against federated models, and how experimental conditions such as the number of clients, model complexity, and dataset difficulty affect attack success.

---

## Research Questions

| ID | Question |
|---|---|
| **RQ1** | Can a black-box MIA successfully infer whether a sample was used during training of a federated learning model? |
| **RQ2** | How does the number of federated clients affect the success rate of membership inference attacks? |
| **RQ3** | How does model complexity influence privacy leakage in federated learning systems? |

---

## Project Structure

```
Data-Science-Lab-/
├── Notebooks/
│   ├── 01_fl_training.ipynb           # FL simulation with FedAvg on CIFAR-10
│   ├── 02_mia_attack.ipynb            # MIA pipeline + RQ1, RQ2, RQ3 on CIFAR-10
│   └── 03_cifar100_experiments.ipynb  # Full pipeline replicated on CIFAR-100
├── Papers/
│   ├── shokri2017_membership_inference.pdf
│   ├── mcmahan2017_fedavg.pdf
│   └── nasr2019_privacy_analysis.pdf
├── results/
│   ├── cifar10_rq1_attack_results.png
│   ├── cifar10_rq2_rq3_results.png
│   ├── cifar10_all_results.json
│   ├── cifar100_rq1_results.png
│   ├── cifar100_rq2_rq3_results.png
│   └── cifar100_all_results.json
├── .gitignore
└── README.md
```

---

## Methodology

### 1. Federated Learning Simulation
- **Dataset:** CIFAR-10 (50,000 training / 10,000 test images, 10 classes) and CIFAR-100 (100 classes)
- **Partitioning:** IID — data shuffled and split equally across clients
- **Algorithm:** FedAvg — clients train locally and send weight updates to central server
- **Configuration:** 5 clients, 10 communication rounds, 3 local epochs per round

### 2. Black-Box Membership Inference Attack
- **Threat model:** Attacker has query access only — observes softmax output probabilities, no access to model weights or gradients
- **Attack dataset:** 5,000 member samples (from training set) + 5,000 non-member samples (from test set)
- **Features:** Full softmax output vector (10-dim for CIFAR-10, 100-dim for CIFAR-100)
- **Attacker model:** Logistic regression binary classifier (member=1, non-member=0)

### 3. Experimental Conditions
- **RQ2:** Varied number of clients — 2, 5, 10
- **RQ3:** Varied model complexity — small CNN (2 conv blocks) vs large CNN (4 conv layers)

### 4. Evaluation Metrics
- Attack Accuracy, Precision, Recall, AUC (random-chance baseline = 0.50)

---

## Results

### CIFAR-10 Results

#### RQ1 — Base Attack (5 clients, small model)

| Metric | Attacker Score | Random Baseline |
|---|---|---|
| Accuracy | 0.5033 | 0.50 |
| Precision | 0.5051 | 0.50 |
| Recall | 0.3313 | 0.50 |
| **AUC** | **0.5128** | **0.50** |

#### RQ2 — Number of Clients vs Attack AUC

| Clients | Global Model Accuracy | Attack AUC |
|---|---|---|
| 2 | 0.7121 | 0.4833 |
| 5 | 0.6768 | 0.4829 |
| 10 | 0.6232 | 0.4816 |

#### RQ3 — Model Complexity vs Privacy Leakage

| Model | Global Accuracy | Attack AUC |
|---|---|---|
| Small CNN (~200K params) | 0.6792 | 0.4788 |
| Large CNN (~1.2M params) | 0.7539 | 0.4784 |

---

### Key Findings

- **RQ1:** The black-box MIA achieved an AUC of 0.51 — barely above random chance — indicating that FedAvg provides meaningful resistance against black-box membership inference attacks under IID conditions.
- **RQ2:** Increasing the number of clients had negligible effect on attack success. AUC remained below 0.50 across all client configurations, suggesting that even a small federation provides privacy protection.
- **RQ3:** Model complexity did not increase privacy leakage. The larger model achieved higher accuracy but similar AUC — FedAvg appears to mask memorisation signals regardless of model size.

---

## Environment & Dependencies

All experiments were run on **Google Colab** with a T4 GPU.

| Package | Version |
|---|---|
| Python | 3.12 |
| TensorFlow / Keras | 2.19.0 |
| scikit-learn | latest |
| NumPy | latest |
| Matplotlib | latest |

---

## How to Run

### Option 1 — Google Colab (recommended)
1. Open each notebook in Google Colab
2. Set runtime to GPU: `Runtime → Change runtime type → T4 GPU`
3. Run notebooks in order: `01` → `02` → `03`
4. Notebooks 02 and 03 are **standalone** — they train their own models internally

### Option 2 — Local machine
```bash
pip install tensorflow scikit-learn matplotlib numpy
jupyter notebook
```

---

## References

- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. *IEEE Symposium on Security and Privacy*, 3–18.

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*, 1273–1282.

- Nasr, M., Shokri, R., & Houmansadr, A. (2019). Comprehensive privacy analysis of deep learning. *IEEE Symposium on Security and Privacy*, 739–753.

---

## Future Work

This study establishes a baseline for black-box MIA in IID federated learning settings. Planned extensions include:

- **Non-IID data partitioning** — evaluating attack success under realistic heterogeneous data distributions
- **White-box attacks** — comparing black-box vs white-box attack success rates
- **Defense mechanisms** — implementing differential privacy and measuring the privacy-utility tradeoff
- **Additional datasets** — extending experiments to medical imaging datasets

---

*This project is the foundation for an ongoing thesis on privacy attacks and defenses in federated learning systems.*
