# Evaluating Black-Box Membership Inference Attacks in Federated Learning

## Overview

This project empirically evaluates black-box Membership Inference Attacks (MIA) against federated learning models trained using the Federated Averaging (FedAvg) algorithm.

A Membership Inference Attack tries to determine whether a specific data sample was used during model training — a serious privacy threat in domains like healthcare and finance. In a medical context, simply confirming that a patient's record was in a training dataset can reveal a sensitive diagnosis without ever accessing the record itself.

> **Core question:** Can an adversary who can only observe a model's softmax output probabilities determine whether a specific sample was used in training?

---

## Research Questions

| ID  | Question |
|-----|----------|
| RQ1 | Can a black-box MIA successfully infer whether a sample was used during training of a federated learning model? |
| RQ2 | How does the number of federated clients affect the success rate of membership inference attacks? |
| RQ3 | How does model complexity influence privacy leakage in federated learning systems? |
| RQ4 | How does Non-IID data partitioning affect attack success compared to IID conditions? |

---

## Key Results

### RQ1 — Baseline Attack (IID, 5 clients, 10 rounds)

| Attack Method        | CIFAR-10 AUC | CIFAR-100 AUC |
|----------------------|-------------|--------------|
| Random baseline      | 0.4969      | 0.4969       |
| Threshold attack     | 0.5355      | 0.5408       |
| Loss-based attack    | 0.5768      | 0.6180       |
| Logistic regression  | 0.4967      | 0.5162       |

### RQ2 — Number of Clients (CIFAR-10)

| Clients | Global Accuracy | Attack AUC |
|---------|----------------|------------|
| 2       | 0.7069         | 0.4932     |
| 5       | 0.6784         | 0.5123     |
| 10      | 0.6415         | 0.5027     |

### RQ3 — Model Complexity (CIFAR-10)

| Model                    | Global Accuracy | Attack AUC |
|--------------------------|----------------|------------|
| Small CNN (~200K params) | 0.6811         | 0.4946     |
| Large CNN (~1.2M params) | 0.7442         | 0.5031     |

### RQ4 — Non-IID vs IID

| Dataset   | Condition                        | Loss AUC | Change |
|-----------|----------------------------------|----------|--------|
| CIFAR-10  | IID                              | 0.5401   | —      |
| CIFAR-10  | Non-IID (5/10 classes per client)  | 0.5644   | +0.024 |
| CIFAR-100 | IID                              | 0.5606   | —      |
| CIFAR-100 | Non-IID (50/100 classes per client)| 0.5636   | +0.003 |

---

## Main Findings

- **FedAvg is the dominant privacy mechanism.** Weight averaging reduces individual memorisation signals regardless of client count or model size — explaining why RQ2 and RQ3 show no consistent effect.
- **The key signal is prediction loss, not confidence patterns.** Logistic regression (AUC ≈ 0.50) fails, while the loss-based method succeeds (AUC 0.577–0.618).
- **Dataset difficulty amplifies leakage.** Harder tasks (lower accuracy) produce larger loss differences — CIFAR-100 consistently shows higher attack AUC than CIFAR-10.
- **Non-IID increases leakage when client specialisation is strong.** Moderate Non-IID increases attack success, while mild Non-IID has minimal effect.

---

## Repository Structure

```
black-box-mia-federated-learning/
│
├── architecture/
│   └── architecture.png
│
├── notebooks/
│   ├── 01_cifar10_experiments.ipynb    # CIFAR-10: FL training + MIA + RQ1/2/3
│   ├── 02_cifar100_experiments.ipynb   # CIFAR-100: FL training + MIA + dataset comparison
│   ├── 03a_noniid_cifar10.ipynb        # Non-IID vs IID on CIFAR-10 (RQ4)
│   └── 03b_noniid_cifar100.ipynb       # Non-IID vs IID on CIFAR-100 (RQ4)
│
├── results/
│   ├── cifar10_all_results.json
│   ├── cifar100_all_results.json
│   ├── cifar10_noniid_results.json
│   └── cifar100_noniid_results.json
│
├── report/
│   └── final_report.pdf
│
├── .gitignore
└── README.md
```

---

## How to Run

All notebooks are fully self-contained — no file uploads required. Everything runs on Google Colab with a free T4 GPU.

### Setup

1. Open [Google Colab](https://colab.research.google.com)
2. Go to `Runtime → Change runtime type → T4 GPU`
3. Open the notebook and click `Runtime → Run all`

### Notebook Order and Runtimes

| Notebook | Purpose | Est. Runtime |
|----------|---------|-------------|
| `01_cifar10_experiments.ipynb`  | Main CIFAR-10 pipeline (RQ1/2/3)         | ~25–30 min |
| `02_cifar100_experiments.ipynb` | CIFAR-100 pipeline + dataset comparison   | ~20–25 min |
| `03a_noniid_cifar10.ipynb`      | Non-IID CIFAR-10 (RQ4)                   | ~15 min    |
| `03b_noniid_cifar100.ipynb`     | Non-IID CIFAR-100 (RQ4)                  | ~20 min    |

> **Important:** Run `03a` and `03b` in separate Colab sessions. Running both together exhausts the free-tier RAM (~12 GB). Always Factory Reset the runtime before starting each one.

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Framework | TensorFlow 2.19.0 |
| Hardware | Google Colab T4 GPU |
| Random seed | 42 (fixed for full reproducibility) |
| FL clients (base) | 5 |
| FL rounds (base) | 10 |
| Local epochs (base) | 3 |
| Batch size | 64 |
| Optimiser | Adam (lr = 0.001) |
| Aggregation | FedAvg (weighted by dataset size) |
| Attack samples | 5,000 members + 5,000 non-members (balanced) |
| Attacker model | Logistic Regression (max_iter = 1000) |
| IID partitioning | Random shuffle + equal split (NumPy array_split) |
| Non-IID partitioning | Class-based cyclic assignment (5/10 or 50/100 classes per client) |

---

## Environment

| Package | Version |
|---------|---------|
| Python | 3.12 |
| TensorFlow / Keras | 2.19.0 |
| scikit-learn | latest |
| NumPy | latest |
| Matplotlib | latest |

---


## References

- Bai, L., et al. (2024). Membership inference attacks and defenses in federated learning: A survey. *ACM Computing Surveys*. arXiv:2412.06157
- Zhu, G., et al. (2024). FedMIA: An effective membership inference attack exploiting the "all for one" principle. *CVPR 2024*, 20643–20653
- Dayal, S., et al. (2023). Comparative analysis of membership inference attacks in federated and centralised learning. *Information*, 14(11), 620
- McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*, 1273–1282
- Nasr, M., et al. (2019). Comprehensive privacy analysis of deep learning. *IEEE S&P*, 739–753

Full reference list in the LaTeX report.

---

*This project is the foundation for an ongoing thesis on privacy attacks and defences in federated learning systems.*