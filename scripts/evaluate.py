#!/usr/bin/env python3
"""Evaluation Script for OralSmart AI Dental Screening Model.

Evaluates the trained model on the held-out test dataset, computing:
- Precision, Recall, and F1-score (macro and per-class)
- Accuracy
- Confusion Matrix
- Category breakdown (No Visible Dental Caries, Permanent Dentition Caries, Primary Dentition Caries)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_model")


def load_model(checkpoint_path, num_classes=3, device="cpu"):
    """Load model from saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_dataset(model, test_manifest, dataset_root, class_names, device="cpu"):
    """Run model inference on all test images and compute comprehensive metrics."""
    test_manifest = Path(test_manifest)
    dataset_root = Path(dataset_root)

    with open(test_manifest, "r", encoding="utf-8") as f:
        samples = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_classes = len(class_names)
    y_true = []
    y_pred = []
    y_probs = []
    per_sample_results = []

    logger.info("Evaluating %d held-out test samples...", len(samples))

    with torch.no_grad():
        for item in samples:
            image_path = dataset_root / item["local_path"]
            with Image.open(image_path) as img:
                tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            true_class = int(item["label_id"])

            y_true.append(true_class)
            y_pred.append(pred_class)
            y_probs.append(probs.tolist())

            per_sample_results.append({
                "file_name": item["file_name"],
                "true_label": class_names[true_class],
                "predicted_label": class_names[pred_class],
                "confidence": float(probs[pred_class]),
                "probabilities": {class_names[i]: float(probs[i]) for i in range(num_classes)},
                "correct": bool(true_class == pred_class),
            })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float((y_true == y_pred).mean())

    # Build Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Per-class metrics
    class_metrics = {}
    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        class_metrics[class_names[c]] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

    report = {
        "dataset": "Zenodo Intraoral Dental Caries Benchmark (Record 14827784)",
        "total_test_samples": len(y_true),
        "overall_accuracy": round(accuracy, 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class_metrics": class_metrics,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist(),
        },
        "sample_evaluations": per_sample_results[:10],
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OralSmart AI model on held-out test set")
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "checkpoints" / "best_model.pth"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parent.parent / "dataset"),
        help="Path to dataset root",
    )
    parser.add_argument(
        "--output-report",
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "test_evaluation_report.json"),
        help="Path to save evaluation report",
    )
    args = parser.parse_args()

    class_names = [
        "No Visible Dental Caries",
        "Dental Caries (Permanent Dentition)",
        "Dental Caries (Primary Dentition)",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.checkpoint, num_classes=len(class_names), device=device)
    test_manifest = Path(args.dataset_dir) / "test" / "annotations.json"

    report = evaluate_dataset(model, test_manifest, args.dataset_dir, class_names, device=device)

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("==================================================")
    logger.info("HELD-OUT TEST SET EVALUATION REPORT")
    logger.info("==================================================")
    logger.info("Total Test Samples: %d", report["total_test_samples"])
    logger.info("Overall Accuracy:   %.2f%%", report["overall_accuracy"] * 100)
    logger.info("Macro Precision:    %.4f", report["macro_precision"])
    logger.info("Macro Recall:       %.4f", report["macro_recall"])
    logger.info("Macro F1-Score:     %.4f", report["macro_f1"])
    logger.info("--------------------------------------------------")
    for cls_name, m in report["per_class_metrics"].items():
        logger.info(
            "%-36s | Prec: %.2f | Rec: %.2f | F1: %.2f | Support: %d",
            cls_name,
            m["precision"],
            m["recall"],
            m["f1_score"],
            m["support"],
        )
    logger.info("--------------------------------------------------")
    logger.info("Confusion Matrix: %s", report["confusion_matrix"]["matrix"])
    logger.info("Report written to %s", args.output_report)

