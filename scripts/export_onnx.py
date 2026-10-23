#!/usr/bin/env python3
"""ONNX Exporter and Verifier for OralSmart AI Dental Screening Model.

Exports trained PyTorch weights to an optimized ONNX model and generates
the corresponding label metadata JSON matching the Flask backend's expectations.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_onnx")


def export_model_to_onnx(
    checkpoint_path,
    output_onnx_path,
    output_labels_path,
    class_names,
    input_size=(224, 224),
    opset_version=14,
):
    """Export PyTorch checkpoint to ONNX and verify numerical parity."""
    checkpoint_path = Path(checkpoint_path)
    output_onnx_path = Path(output_onnx_path)
    output_labels_path = Path(output_labels_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    output_labels_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    num_classes = len(class_names)
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)

    logger.info("Exporting to ONNX format at %s...", output_onnx_path)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )
    except TypeError:
        # Fallback if dynamo argument is not accepted in older torch
        torch.onnx.export(
            model,
            dummy_input,
            str(output_onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    logger.info("Validating ONNX model integrity...")
    onnx_proto = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_proto)
    logger.info("ONNX model structure is valid.")

    logger.info("Verifying inference parity between PyTorch and ONNX Runtime...")
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(pytorch_output, ort_output, rtol=1e-3, atol=1e-4)
    logger.info("Parity verified successfully! Maximum absolute deviation: %.6f", np.max(np.abs(pytorch_output - ort_output)))

    # Write labels and configuration JSON
    metadata = {
        "classes": class_names,
        "output_type": "logits",
        "architecture": checkpoint.get("architecture", "mobilenet_v3_small"),
        "input_shape": [1, 3, input_size[0], input_size[1]],
        "color_channel_order": "RGB",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "dataset": "Annotated Intraoral Image Dataset for Dental Caries Detection (Zenodo Record 14827784)",
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
        "val_metrics": checkpoint.get("val_metrics", {}),
    }

    with open(output_labels_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model labels to %s", output_labels_path)
    logger.info("Export completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained dental model to ONNX")
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "checkpoints" / "best_model.pth"),
        help="Path to trained PyTorch checkpoint",
    )
    parser.add_argument(
        "--output-model",
        default=str(Path(__file__).resolve().parent.parent / "web" / "instance" / "models" / "dental_classifier.onnx"),
        help="Target ONNX model path",
    )
    parser.add_argument(
        "--output-labels",
        default=str(Path(__file__).resolve().parent.parent / "web" / "instance" / "models" / "dental_classifier.labels.json"),
        help="Target labels JSON path",
    )
    args = parser.parse_args()

    class_names = [
        "No Visible Dental Caries",
        "Dental Caries (Permanent Dentition)",
        "Dental Caries (Primary Dentition)",
    ]

    export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output_model,
        output_labels_path=args.output_labels,
        class_names=class_names,
    )

