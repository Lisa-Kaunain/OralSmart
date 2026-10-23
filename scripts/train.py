#!/usr/bin/env python3
"""Dental Image Screening Model Training Pipeline.

Trains a deep transfer learning model on intraoral photographs to classify
dental caries findings (No Visible Dental Caries, Permanent Dentition Caries,
Primary Dentition Caries) based on clinical annotations from the Zenodo benchmark.
"""

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_model")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OralSmartDataset(Dataset):
    """Dataset loader for OralSmart AI intraoral photographs."""

    def __init__(self, manifest_file, dataset_root, transform=None):
        self.manifest_file = Path(manifest_file)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = self.dataset_root / item["local_path"]

        # Load RGB image
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(item["label_id"])
        return image, label, str(image_path)


def get_transforms(img_size=224):
    """Data augmentation for training and normalization for validation/testing."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        # Standard ImageNet normalization matching ONNX runtime preprocessing
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def build_model(num_classes=3, pretrained=True):
    """Build a MobileNetV3-Small transfer learning classifier."""
    logger.info("Initializing MobileNetV3-Small architecture (num_classes=%d)", num_classes)
    try:
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            logger.info("Loaded pretrained MobileNetV3 weights.")
        else:
            model = models.mobilenet_v3_small(weights=None)
    except Exception as e:
        logger.warning("Could not download weights online (%s). Initializing fresh model.", e)
        model = models.mobilenet_v3_small(weights=None)

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def compute_metrics(y_true, y_pred, num_classes=3):
    """Compute accuracy, per-class precision, recall, and macro F1."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": [float(x) for x in f1s],
    }


def train_model(
    dataset_dir,
    output_dir,
    epochs=15,
    batch_size=16,
    learning_rate=1e-3,
    seed=42,
    num_classes=3,
):
    """Train the dental classifier with validation monitoring and checkpointing."""
    set_seed(seed)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    train_manifest = dataset_dir / "train" / "annotations.json"
    valid_manifest = dataset_dir / "valid" / "annotations.json"

    train_tf, val_tf = get_transforms()
    train_dataset = OralSmartDataset(train_manifest, dataset_dir, transform=train_tf)
    valid_dataset = OralSmartDataset(valid_manifest, dataset_dir, transform=val_tf)

    logger.info(
        "Loaded datasets: Train=%d samples, Validation=%d samples",
        len(train_dataset),
        len(valid_dataset),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = build_model(num_classes=num_classes).to(device)

    # Class weighting to address class imbalance
    labels_in_train = [int(s["label_id"]) for s in train_dataset.samples]
    class_counts = [labels_in_train.count(c) for c in range(num_classes)]
    logger.info("Class distribution in training set: %s", class_counts)

    weights = [
        max(class_counts) / max(c, 1) for c in class_counts
    ]
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info("Class loss weights: %s", weights)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    training_history = []

    logger.info("Starting training loop for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Training Phase
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())

        scheduler.step()
        train_loss = train_loss / len(train_dataset)
        train_metrics = compute_metrics(train_targets, train_preds, num_classes)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, targets, _ in valid_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())

        val_loss = val_loss / len(valid_dataset)
        val_metrics = compute_metrics(val_targets, val_preds, num_classes)
        dt = time.time() - t0

        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Train Loss: %.4f, Acc: %.3f, F1: %.3f | Val Loss: %.4f, Acc: %.3f, F1: %.3f",
            epoch,
            epochs,
            dt,
            train_loss,
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_loss,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        training_history.append(record)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_weights = copy.deepcopy(model.state_dict())
            checkpoint_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_weights,
                    "val_metrics": val_metrics,
                    "val_loss": val_loss,
                    "num_classes": num_classes,
                    "architecture": "mobilenet_v3_small",
                },
                checkpoint_path,
            )
            logger.info("Saved new best model checkpoint to %s (Val F1: %.4f)", checkpoint_path, best_val_f1)

    # Save training logs
    log_file = output_dir / "training_history.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    logger.info("Training complete. Best Validation F1: %.4f", best_val_f1)
    return str(output_dir / "best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OralSmart AI Dental Screening Model")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parent.parent / "dataset"),
        help="Path to prepared dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "checkpoints"),
        help="Path to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

