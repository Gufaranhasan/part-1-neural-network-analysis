# Part 1 — Neural Network Fundamentals and Training Behavior Analysis

> **Course:** BITSoM Business Analytics — Module 5  
> **Problem Type:** Binary Classification  
> **Dataset:** `customer_churn_nn.csv` — Telecom Customer Churn (2,000 records, 16 features)  
> **Framework:** TensorFlow / Keras

---

## Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md                        ← This file
├── notebook.ipynb                   ← Full analysis notebook (all 6 tasks)
├── requirements.txt                 ← Python dependencies
└── results/
    ├── target_distribution.png      ← Class balance chart
    ├── feature_distributions.png    ← Per-feature histograms by class
    ├── baseline_learning_curves.png ← Loss & accuracy vs epoch (baseline)
    ├── confusion_matrix_baseline.png← Confusion matrix (baseline model)
    ├── model_comparison_table.csv   ← Hyperparameter experiment results
    ├── model_comparison_table.png   ← Styled table image
    ├── experiment_comparison_bar.png← Bar chart of test accuracy per config
    └── evaluation_outputs.png       ← Learning curves for all 7 experiments
```

---

## Quick Start

```bash
# 1. Clone / download this repo
git clone <your-repo-url>
cd part-1-neural-network-analysis

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebook.ipynb
```

> **Using your own dataset?**  
> In the second code cell, replace the `URL` / `COLUMNS` block with:
> ```python
> df = pd.read_csv('your_dataset.csv')
> ```

---

## Task Summary

| Task | Description |
|------|-------------|
| **Task 1** | Dataset exploration — shape, types, missing values, statistics, target distribution |
| **Task 2** | Preprocessing — imputation, encoding check, StandardScaler, 80/20 stratified split |
| **Task 3** | Feed-forward neural network — input → hidden (ReLU) → output (sigmoid); Adam + BCE loss |
| **Task 4** | Training with early stopping; test accuracy, loss, confusion matrix, classification report |
| **Task 5** | 7 experiments varying LR, neurons, depth, batch size, activation; comparison table + charts |
| **Task 6** | Reflection — weights/biases, activation functions, learning rate effects, over/underfitting |

---

## Model Architecture (Baseline)

```
Input  (15 features — 11 numerical + 4 label-encoded categorical)
  ↓
Dense (32 units, ReLU)
  ↓
Dense (1 unit, Sigmoid)

Optimizer : Adam  (lr = 0.001)
Loss      : Binary Cross-Entropy
Metric    : Accuracy
```

---

## Hyperparameter Experiments

| Exp | Change from Baseline | Expected Effect |
|-----|----------------------|-----------------|
| 1   | Baseline | Reference |
| 2   | LR = 0.01 (10×) | Faster, noisier convergence |
| 3   | LR = 0.0001 (10×) | Slower, smoother convergence |
| 4   | Neurons = 128 | More capacity, possible overfit |
| 5   | 3 hidden layers, 64 neurons | Deeper representation |
| 6   | Activation = tanh | Different non-linearity |
| 7   | Batch size = 8 (small) | Noisier updates, slower per-epoch |

---

## Key Findings (Task 6 Summary)

- **Weights & Biases** — Weights control connection strength; biases shift activations independently. Both are updated via backpropagation to minimise loss.
- **Activation Functions** — Non-linear activations (ReLU, tanh) let the network learn complex boundaries. Without them, all layers collapse to a single linear transform.
- **Learning Rate** — Too high → oscillating/diverging loss. Too low → painfully slow convergence and underfitting. A scheduler or Adam's adaptive rates help.
- **Over/Underfitting** — Baseline shows mild overfitting (train > test by ~3%). Wider/deeper models widen this gap without regularisation. Low LR caused underfitting.

---

## Dependencies

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| TensorFlow | ≥ 2.13 |
| scikit-learn | ≥ 1.3 |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |

---

*Generated as part of BITSoM BA Module 5 — Neural Networks Assignment.*
