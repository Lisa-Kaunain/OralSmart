# OralSmart AI — AI Dental Screening & Patient-Dentist Platform

**OralSmart AI** is a specialized intraoral dental photograph screening and personalized oral healthcare platform. The system bridges patients and licensed dentists: patients can upload intraoral photographs taken from mobile phones for preliminary visual caries screening powered by an ONNX-optimized deep learning model, while qualified dentists review images, establish professional diagnoses, and formulate customized treatment plans.

---

## Key Features

* **Intraoral AI Screening:** Real-time visual screening for dental caries using a MobileNetV3 model trained on the clinical Zenodo intraoral caries benchmark.
* **Dual Patient & Dentist Portals:** Complete role-based authentication, personalized dashboard, and screening history tracking.
* **Clinician Review Workflow:** Seamless queue where dentists review patient photographs and AI preliminary findings to submit official clinical evaluations and follow-up schedules.
* **ONNX Runtime Acceleration:** Fast, low-latency CPU and GPU inference without external API dependencies.
* **Clinical Safety & Medical Ethics:** Strictly transparent preliminary explanations; prevents diagnostic overreach, automated prescribing, or unsupervised treatment claims.

---

## Supported Screening Classes

Trained on the **Annotated Intraoral Image Dataset for Dental Caries Detection** (Zenodo Record 14827784, CC-BY-4.0):

1. **`No Visible Dental Caries`** — Healthy dentition with no observable enamel breakdown or cavitation.
2. **`Dental Caries (Permanent Dentition)`** — Visual patterns consistent with carious lesions in secondary/permanent teeth (WHO DMFT Index).
3. **`Dental Caries (Primary Dentition)`** — Visual patterns suggestive of decay in deciduous/primary baby teeth (WHO dmft Index).

---

## Directory Structure

```
OralSmartAI/
├── dataset/                        # Dataset manifest, splits & evaluation logs
│   ├── train/                      # 120 training images & annotations
│   ├── valid/                      # 29 validation images & annotations
│   ├── test/                       # 30 held-out test images & annotations
│   ├── checkpoints/                # Best model PyTorch checkpoint (best_model.pth)
│   ├── dataset_metadata.json       # Formal dataset metadata & license
│   └── test_evaluation_report.json # Detailed held-out test evaluation report
├── docs/                           # Documentation
│   ├── DATASET.md                  # Dataset source, license & taxonomy
│   ├── MODEL_TRAINING.md           # Model architecture, training log & metrics
│   └── RUNNING.md                  # Setup, running & testing instructions
├── scripts/                        # Reproducible ML & Data pipelines
│   ├── prepare_dataset.py          # Zenodo dataset downloader & preprocessor
│   ├── train.py                    # Model training with transfer learning & validation
│   ├── evaluate.py                 # Comprehensive held-out test evaluation
│   └── export_onnx.py              # Export PyTorch weights to ONNX with parity checks
├── tests/                          # Automated tests
│   └── test_oralsmart_complete.py  # 12 comprehensive unit and integration tests
├── web/                            # Flask web application
│   ├── database/
│   │   └── schema.sql              # Database reference schema
│   ├── instance/
│   │   ├── models/                 # Exported ONNX model & labels
│   │   │   ├── dental_classifier.onnx
│   │   │   └── dental_classifier.labels.json
│   │   └── uploads/                # Patient image upload storage
│   ├── services/
│   │   ├── ai_service.py           # ONNX Runtime inference & clinical guidance
│   │   └── recommendation_service.py
│   ├── static/                     # CSS stylesheets & client JavaScript
│   ├── templates/                  # Jinja2 HTML templates for patient & dentist
│   ├── app.py                      # Flask entry point and application routes
│   ├── config.py                   # App configuration
│   └── requirements.txt            # Python dependencies
└── README.md
```

---

## Evaluation Results (Held-Out Test Set, $N=30$)

* **Overall Accuracy:** **86.67%**
* **Macro Precision:** **0.9030**
* **Macro Recall:** **0.8315**
* **Macro F1-Score:** **0.8456**
* **Inference Speed:** ~18ms per image on CPU via ONNX Runtime

---

## Quick Start

### 1. Install Dependencies
```powershell
python -m pip install -r web/requirements.txt
python -m pip install torch torchvision onnx onnxruntime onnxscript pytest requests
```

### 2. Run Test Suite
```powershell
python -m pytest -v tests/test_oralsmart_complete.py
```

### 3. Start Application
```powershell
cd web
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.
