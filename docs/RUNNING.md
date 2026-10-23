# OralSmart AI — Setup & Execution Guide

## 1. System Requirements

* **Operating System:** Windows 10/11, macOS, or Linux.
* **Python Version:** Python 3.10 to 3.13 (Verified on Python 3.13.9 64-bit AMD64).
* **RAM:** 4 GB minimum (8 GB+ recommended).
* **Disk Space:** ~500 MB for application, dependencies, model, and dataset samples.

---

## 2. Environment Setup

### Step 1: Navigate to Project Directory
```powershell
cd C:\Users\lisak\.gemini\antigravity\scratch\OralSmartAI
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r web/requirements.txt
python -m pip install torch torchvision onnx onnxruntime onnxscript pytest requests
```

---

## 3. Dataset Preparation, Model Training & Export

If you wish to re-run the end-to-end dataset preparation and training:

### A. Prepare the Zenodo Benchmark Dataset
Downloads and validates intraoral photographs from Zenodo Record 14827784:
```powershell
python scripts/prepare_dataset.py --train-samples 120 --valid-samples 30 --test-samples 30 --workers 2
```

### B. Train the Dental Screening Model
```powershell
python scripts/train.py --epochs 12 --batch-size 16 --lr 0.001 --seed 42
```
Checkpoints are saved to `dataset/checkpoints/best_model.pth`.

### C. Evaluate on the Held-Out Test Set
```powershell
python scripts/evaluate.py
```
Outputs report to `dataset/test_evaluation_report.json`.

### D. Export to ONNX
```powershell
python scripts/export_onnx.py
```
Exports `dental_classifier.onnx` and `dental_classifier.labels.json` to `web/instance/models/`.

---

## 4. Running the Flask Application

### Quick Start:
```powershell
cd web
python app.py
```
The application will start on `http://127.0.0.1:5000`.

### Running with Custom Port / Production Mode:
```powershell
$env:FLASK_APP = "web\app.py"
$env:FLASK_RUN_PORT = "5000"
python -m flask run --port 5000
```

---

## 5. End-to-End Testing & Verification

Run the comprehensive automated test suite (12 tests covering AI service, ONNX inference, Flask routes, authentication, screening storage, and dentist workflows):
```powershell
python -m pytest -v tests/test_oralsmart_complete.py
```

### Manual Workflow Verification:
1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Click **Create account** and register as a **Patient** (e.g., `patient@example.com` / `Password123!`).
3. Click **New Screening** on the patient dashboard.
4. Upload any intraoral tooth photograph (`.jpg`, `.jpeg`, or `.png`).
5. View the resulting **AI Preliminary Screening Report**:
   - Shows the detected condition (`No Visible Dental Caries`, `Dental Caries (Permanent Dentition)`, or `Dental Caries (Primary Dentition)`).
   - Shows model confidence percentage (e.g., `85.4%`).
   - Displays clear, clinically sound preliminary explanations and recommended next steps.
   - Includes medical disclaimers emphasizing that photographic AI is not a dental diagnosis.
6. Click **Request dentist review**.
7. Log out and register a **Dentist** account (e.g., `dentist@example.com` / `Password123!`).
8. The screening appears in the dentist's pending queue on `/dentist`.
9. The dentist can click **Review**, input professional findings, enter a customized treatment plan and follow-up date, and save.
10. The patient logs back in and views the updated dentist finding and follow-up appointment date directly on their report.
