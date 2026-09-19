# Evaluating Black-Box Membership Inference Attacks in Federated Learning Systems

**Abdihakim Yussuf Abdi**  
Data Science MSc — Eötvös Loránd University (ELTE), Budapest, 2026  
Supervisor: Kiss Péter

---

## Overview

This project presents a systematic empirical evaluation of membership inference attacks (MIAs) against federated learning models trained using the Federated Averaging (FedAvg) algorithm. The evaluation is structured across three tiers of attacker capability and seven research questions, implemented across seven self-contained reproducible notebooks on CIFAR-10 and CIFAR-100 datasets under both IID and Non-IID data distributions.

The central question: **can an adversary determine whose data was used to train a federated learning model, using only the model's prediction outputs?**

---

## Key Findings

- FedAvg resists trained black-box classifier attacks under IID conditions — logistic regression AUC near 0.50 across all configurations
- The primary vulnerability channel is **prediction loss**, not confidence vector patterns — loss-based attack achieves AUC 0.577 (CIFAR-10) and 0.618 (CIFAR-100)
- A realistic **shadow model attack** using CIFAR-100 as shadow data (no real training data access) achieves AUC 0.540 and member recall **78.5%** — outperforming the unrealistic baseline
- The **white-box gradient attack** achieves the highest AUC of 0.615 under IID — confirming gradient access provides substantially more information than softmax outputs alone
- **Non-IID data heterogeneity** consistently amplifies all attack types — loss AUC increases by +0.026 on CIFAR-10 under moderate Non-IID
- **DP-SGD** successfully suppresses attacks to near-random chance but incurs severe accuracy costs — hyperparameter tuning required for a practical operating point

---

## Three-Tier Attacker Capability Hierarchy

| Tier | Attack | IID AUC | Non-IID AUC | Realistic? |
|---|---|---|---|---|
| — | Random chance | 0.500 | 0.500 | — |
| T1 | Black-box LR (unrealistic) | ~0.492 | — | No |
| T1 | Loss-based (no training) | 0.577 | 0.565 | Yes |
| T3 | Shadow model (CIFAR-100) | 0.540 | 0.549 | Yes |
| T3 | White-box gradient | 0.615 | 0.613 | Yes* |

*Realistic in the compromised server threat model.

---

## Project Structure

```
.
├── notebooks/
│   ├── 01_cifar10_experiments.ipynb      # RQ1, RQ2, RQ3 — CIFAR-10 baseline
│   ├── 02_cifar100_experiments.ipynb     # RQ1, RQ2, RQ3 — CIFAR-100 baseline
│   ├── 03a_noniid_cifar10.ipynb          # RQ4 — Non-IID CIFAR-10
│   ├── 03b_noniid_cifar100.ipynb         # RQ4 — Non-IID CIFAR-100
│   ├── 04_dp_defence.ipynb               # RQ5 — DP-SGD defence
│   ├── 05_shadow_model_attack.ipynb      # RQ6 — Shadow model attack
│   └── 06_whitebox_attack.ipynb          # RQ7 — White-box gradient attack
├── figures/                              # All generated figures (PNG)
├── results/                              # All result files (JSON)
├── report/                               # Thesis LaTeX source
├── Papers/                               # Reference papers
└── README.md
```

---

## Research Questions

**Tier 1 — Controlled Black-Box Experiments**
- **RQ1:** Can a black-box MIA successfully infer training membership against a FedAvg model?
- **RQ2:** How does the number of federated clients affect attack success?
- **RQ3:** How does model complexity influence privacy leakage?

**Tier 2 — Distribution and Defence Experiments**
- **RQ4:** How does Non-IID data partitioning affect membership inference attack success?
- **RQ5:** Can DP-SGD effectively suppress the loss-based attack, and at what cost to utility?

**Tier 3 — Realistic and Advanced Attacks**
- **RQ6:** Can a shadow model attack using similar-domain data (no real training data access) successfully infer membership?
- **RQ7:** How much additional membership information does white-box gradient access provide?

---

## Datasets

| Dataset | Classes | Train | Test | Task difficulty |
|---|---|---|---|---|
| CIFAR-10 | 10 | 50,000 | 10,000 | Moderate (~72% accuracy) |
| CIFAR-100 | 100 | 50,000 | 10,000 | Hard (~38% accuracy) |

Both downloaded automatically via `keras.datasets` — no manual upload required.

---

## Federated Learning Setup

- **Algorithm:** FedAvg (McMahan et al. 2017)
- **Default config:** 5 clients, 10 rounds, 3 local epochs, batch size 64, Adam lr=0.001
- **Model:** Small CNN (~200K params) — two conv blocks + dense layer
- **IID partitioning:** Random equal split across clients
- **Non-IID partitioning:** Class-based cyclic assignment (5/10 classes per client for CIFAR-10)

---

## Attack Methods

| Method | Features used | Training data needed | Notebook |
|---|---|---|---|
| Random baseline | — | No | 01, 02 |
| Threshold attack | Max softmax confidence | No | 01, 02 |
| Loss-based attack | Cross-entropy loss scalar | No | 01–06 |
| Black-box LR | 6 architecture-agnostic features | Yes (real data) | 01, 02 |
| Shadow model | 6 architecture-agnostic features | Yes (shadow data only) | 05 |
| White-box gradient | Per-layer gradient norms + loss + confidence | Yes (real data) | 06 |

The 6 architecture-agnostic features: max confidence, prediction entropy, top-3 confidence values, loss proxy.

---

## Differential Privacy Defence

DP-SGD implemented from scratch using TensorFlow `GradientTape` — no external DP library. Per-sample gradient clipping (C=1.0) and Gaussian noise injection evaluated at four privacy budgets:

| Configuration | ε | σ | Global Acc | Loss AUC |
|---|---|---|---|---|
| No DP | ∞ | 0.0 | ~67% | ~0.516 |
| Weak DP | 50 | 0.5 | TBD | TBD |
| Moderate DP | 10 | 1.1 | ~12% | ~0.465 |
| Strong DP | 1 | 3.0 | ~17% | ~0.498 |

---

## How to Run

All notebooks run on Google Colab with a T4 GPU. No file uploads required.

1. Open any notebook in Google Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Select **Runtime → Run all**

**Run order for complete results:**
1. Notebooks 01, 02, 03a, 03b, 04 — independent, any order
2. Notebook 05 — run before Notebook 06
3. Notebook 06 — loads shadow results from Notebook 05

---

## Environment

- Python 3.x
- TensorFlow 2.19.0
- scikit-learn
- NumPy, Matplotlib
- Google Colab T4 GPU
- Random seed: 42 (fixed globally for reproducibility)

---

## References

- McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks from Decentralized Data
- Shokri et al. (2017) — Membership Inference Attacks Against Machine Learning Models
- Nasr et al. (2019) — Comprehensive Privacy Analysis of Deep Learning
- Yeom et al. (2018) — Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting
- Bai et al. (2024) — Membership Inference Attacks and Defenses in Federated Learning: A Survey

---

## License

This project is submitted as part of an MSc thesis at Eötvös Loránd University. All code is original and reproducible.