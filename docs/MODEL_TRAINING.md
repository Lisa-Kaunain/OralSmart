# OralSmart AI — Model Training & Evaluation Documentation

## 1. Model Architecture

* **Backbone:** MobileNetV3-Small (lightweight deep convolutional neural network optimized for fast edge and CPU inference).
* **Weights:** Pretrained on ImageNet-1K (`MobileNet_V3_Small_Weights.DEFAULT`), fine-tuned with transfer learning on the Zenodo intraoral caries benchmark.
* **Classifier Head:**
  - Dropout layer ($p=0.3$) for regularization.
  - Linear projection layer: $1024 \to 3$ classes (`No Visible Dental Caries`, `Dental Caries (Permanent Dentition)`, `Dental Caries (Primary Dentition)`).

---

## 2. Training Pipeline & Hyperparameters

* **Hardware Used:** 12th Gen Intel Core i7-12700H (14 cores / 20 logical threads), 16 GB RAM.
* **Device:** CPU (CPUExecutionProvider).
* **Optimizer:** AdamW with weight decay $1\times 10^{-4}$.
* **Learning Rate:** $1\times 10^{-3}$ with Cosine Annealing learning rate schedule.
* **Loss Function:** Weighted Cross-Entropy Loss to counter class frequency imbalance:
  - Class 0 (No Visible Caries): $1.5\times$
  - Class 1 (Permanent Caries D): $1.0\times$
  - Class 2 (Primary Caries d): $1.8\times$
* **Batch Size:** 16
* **Epochs:** 12
* **Random Seed:** 42 (fixed for reproducibility)

---

## 3. Training Progress & Validation Results

Model checkpoints were evaluated after each epoch on the independent validation split ($N=29$). The best checkpoint was automatically saved based on Validation Macro F1-score:

| Epoch | Train Loss | Train Accuracy | Train Macro F1 | Val Loss | Val Accuracy | Val Macro F1 | Checkpoint Saved |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.8436 | 62.5% | 0.607 | 0.9373 | 69.0% | 0.514 | Yes |
| 2 | 0.3358 | 85.0% | 0.844 | 1.0049 | 65.5% | 0.645 | Yes |
| 3 | 0.3121 | 89.2% | 0.888 | 0.9864 | 72.4% | 0.700 | Yes |
| 4 | 0.1400 | 96.7% | 0.966 | 1.1330 | 79.3% | 0.767 | Yes |
| 5 | 0.0395 | 99.2% | 0.991 | 1.4261 | 75.9% | 0.753 | — |
| **6** | **0.0609** | **96.7%** | **0.965** | **1.3438** | **82.8%** | **0.821** | **Best Checkpoint** |
| 7 | 0.0364 | 98.3% | 0.984 | 1.4557 | 79.3% | 0.791 | — |
| 8 | 0.0728 | 98.3% | 0.982 | 1.2854 | 82.8% | 0.821 | — |
| 9 | 0.0296 | 99.2% | 0.990 | 1.1293 | 82.8% | 0.821 | — |
| 10 | 0.0075 | 100.0% | 1.000 | 1.0754 | 82.8% | 0.809 | — |
| 11 | 0.0200 | 99.2% | 0.991 | 1.0591 | 82.8% | 0.809 | — |
| 12 | 0.0129 | 99.2% | 0.991 | 1.0408 | 82.8% | 0.809 | — |

---

## 4. Final Evaluation on Held-Out Test Split

Evaluated strictly on the held-out test split ($N=30$) which was never accessed during model training or hyperparameter tuning:

### Overall Metrics:
* **Total Test Samples:** 30
* **Overall Accuracy:** **86.67%** (26/30 correct)
* **Macro Precision:** **0.9030**
* **Macro Recall:** **0.8315**
* **Macro F1-Score:** **0.8456**

### Per-Class Detailed Performance:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **No Visible Dental Caries** | 0.91 | 1.00 | **0.95** | 10 |
| **Dental Caries (Permanent Dentition)** | 0.80 | 0.92 | **0.86** | 13 |
| **Dental Caries (Primary Dentition)** | 1.00 | 0.57 | **0.73** | 7 |

### Confusion Matrix:

```
                      Predicted Normal    Predicted Permanent (D)    Predicted Primary (d)
Actual Normal:               10                      0                         0
Actual Permanent (D):         1                     12                         0
Actual Primary (d):           0                      3                         4
```

* **Zero false positives on normal teeth:** All 10 healthy control images were correctly identified as having no visible caries.
* **High permanent caries sensitivity:** 92.3% of permanent caries teeth were successfully identified.
* **Primary caries confusion:** 3 primary teeth cases were classified as permanent caries, typical of mixed dentition photographs where primary molars and erupting permanent teeth appear side by side.

---

## 5. ONNX Export & Parity Verification

* **Export Target:** `web/instance/models/dental_classifier.onnx` (File size: 6.09 MB)
* **Metadata Target:** `web/instance/models/dental_classifier.labels.json`
* **Opset Version:** 14 (universal ONNX Runtime CPU compatibility)
* **Dynamic Axes:** Batch size dimension is dynamic (`{0: 'batch_size'}`)
* **Numerical Parity:** PyTorch vs. ONNX Runtime maximum absolute output deviation: **$0.000001$ ($1\times 10^{-6}$)**
