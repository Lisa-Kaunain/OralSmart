"""Run preliminary dental image classification with an optional ONNX model."""

from functools import lru_cache
import json
import math
import os
from pathlib import Path

from PIL import Image


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "instance" / "models" / "dental_classifier.onnx"
DEFAULT_LABELS_PATH = Path(__file__).resolve().parent.parent / "instance" / "models" / "dental_classifier.labels.json"


def _model_paths():
    model_path = Path(os.environ.get("DENTAL_MODEL_PATH", DEFAULT_MODEL_PATH))
    labels_path = Path(os.environ.get("DENTAL_MODEL_LABELS_PATH", DEFAULT_LABELS_PATH))
    return model_path, labels_path


def _unavailable_result():
    return {
        "condition": "AI model unavailable — no prediction",
        "confidence": None,
        "explanation": (
            "No trained dental image model and class-label configuration were found. "
            "This image was saved but no condition was identified."
        ),
        "recommendation": (
            "No AI screening recommendation is available. You may request a qualified "
            "dentist to review the image."
        ),
        "mode": "unavailable",
    }


@lru_cache(maxsize=2)
def _load_model_session(model_path, modified_ns, file_size):
    del modified_ns, file_size
    try:
        import onnxruntime
    except ImportError as error:
        raise RuntimeError(
            "The ONNX model is configured, but onnxruntime is missing. "
            "Install it with 'pip install onnxruntime'."
        ) from error

    return onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _load_labels(labels_path):
    with labels_path.open(encoding="utf-8") as labels_file:
        metadata = json.load(labels_file)

    classes = metadata.get("classes")
    output_type = metadata.get("output_type")
    normalization = metadata.get("normalization")

    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(label, str) or not label.strip() for label in classes)
    ):
        raise ValueError("The model labels file must contain a non-empty 'classes' list.")
    if output_type not in {"logits", "probabilities"}:
        raise ValueError("'output_type' in the model labels file must be 'logits' or 'probabilities'.")
    return classes, output_type, normalization


def _preprocess_image(image_path, input_shape, normalization=None):
    import numpy

    if len(input_shape) != 4:
        raise ValueError("The ONNX model input must have NCHW shape [1, 3, height, width].")
    batch, channels, height, width = input_shape
    if (isinstance(batch, int) and batch != 1) or (isinstance(channels, int) and channels != 3):
        raise ValueError("The ONNX model input must accept one RGB image in NCHW format.")
    height = height if isinstance(height, int) and height > 0 else 224
    width = width if isinstance(width, int) and width > 0 else 224

    with Image.open(image_path) as image:
        resized = image.convert("RGB").resize((width, height))
        pixels = numpy.asarray(resized, dtype=numpy.float32) / 255.0

    if normalization and isinstance(normalization, dict):
        mean = numpy.array(normalization.get("mean", [0.485, 0.456, 0.406]), dtype=numpy.float32)
        std = numpy.array(normalization.get("std", [0.229, 0.224, 0.225]), dtype=numpy.float32)
        pixels = (pixels - mean) / std

    return numpy.expand_dims(numpy.transpose(pixels, (2, 0, 1)), axis=0).astype(numpy.float32)


def _probabilities(output, output_type, class_count):
    import numpy

    scores = numpy.asarray(output, dtype=numpy.float64)
    if scores.size != class_count or not numpy.isfinite(scores).all():
        raise ValueError("The ONNX model output does not match the configured class labels.")
    scores = scores.reshape(class_count)

    if output_type == "probabilities":
        if (scores < 0).any() or not math.isclose(float(scores.sum()), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("The model output is not a valid probability distribution.")
        return scores / scores.sum()

    shifted_scores = scores - scores.max()
    exponentials = numpy.exp(shifted_scores)
    return exponentials / exponentials.sum()


def _format_clinical_explanation_and_guidance(condition):
    """Generate clinically accurate explanations and guidance without diagnostic or prescription overreach."""
    if condition == "Dental Caries (Permanent Dentition)":
        explanation = (
            "Visual AI preliminary screening identified surface features consistent with potential "
            "carious lesions in secondary (permanent) teeth. Note: Standard photographs cannot visualize "
            "interproximal decay between teeth or subgingival margins, and visual screening is not a clinical diagnosis."
        )
        recommendation = (
            "Consult a qualified dentist for a clinical evaluation. The dentist will perform a physical visual-tactile "
            "examination and obtain dental radiographs to verify tooth structure and determine whether preventive or "
            "restorative treatment is warranted."
        )
    elif condition == "Dental Caries (Primary Dentition)":
        explanation = (
            "Visual AI preliminary screening identified findings suggestive of dental decay in primary (deciduous/baby) "
            "dentition. Primary teeth have thinner enamel and require prompt dental monitoring."
        )
        recommendation = (
            "Schedule an appointment with a dentist or pediatric dental specialist. Prompt professional intervention helps "
            "preserve primary dentition and protects developing permanent teeth."
        )
    elif condition == "No Visible Dental Caries":
        explanation = (
            "No visible carious lesions or overt enamel cavitation were detected on the exposed surfaces of this screening image. "
            "Note that early demineralization or interdental cavities may not be visible in surface photographs."
        )
        recommendation = (
            "Continue daily oral health maintenance: brush twice daily with fluoride toothpaste, clean between teeth daily, "
            "and schedule regular professional dental check-ups every six months."
        )
    else:
        explanation = (
            f"The configured ONNX dental classifier assigned its highest score to '{condition}'. "
            "This automated screening is for informational guidance only and is not a dental diagnosis."
        )
        recommendation = (
            "Discuss this preliminary result with a licensed dentist. The AI classifier does not determine dental prescriptions "
            "or treatment plans."
        )
    return explanation, recommendation


def analyze_dental_image(image_path):
    """Return an ONNX classifier result, or an explicit no-model result."""
    if not image_path:
        raise ValueError("An image path is required.")

    model_path, labels_path = _model_paths()
    if not model_path.is_file() and not labels_path.is_file():
        return _unavailable_result()
    if not model_path.is_file() or not labels_path.is_file():
        raise RuntimeError(
            "Both the configured ONNX model and its class-label JSON file are required."
        )

    classes, output_type, normalization = _load_labels(labels_path)
    model_stat = model_path.stat()
    model_session = _load_model_session(
        str(model_path.resolve()), model_stat.st_mtime_ns, model_stat.st_size
    )
    model_input = model_session.get_inputs()[0]
    input_tensor = _preprocess_image(image_path, model_input.shape, normalization=normalization)
    output = model_session.run(None, {model_input.name: input_tensor})[0]
    class_probabilities = _probabilities(output, output_type, len(classes))
    predicted_index = int(class_probabilities.argmax())
    condition = classes[predicted_index]
    confidence = float(class_probabilities[predicted_index])

    explanation, recommendation = _format_clinical_explanation_and_guidance(condition)

    return {
        "condition": condition,
        "confidence": confidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "mode": "onnx",
        "probabilities": {classes[i]: float(class_probabilities[i]) for i in range(len(classes))},
    }
