
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection_engine.models.tabular_transformer import TabularTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "detection_engine"
    / "models"
    / "best_transformer.pt"
)

NUM_FEATURES = 147

# CUDA is preferred. CPU is the fallback.
# No TPU or XLA backend is used.
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEVICE INFORMATION
# ============================================================

def get_device_info():
    """
    Return the selected PyTorch execution device.
    """

    info = {
        "device": str(DEVICE),
        "device_type": DEVICE.type,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    if DEVICE.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(
            DEVICE
        )
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(
                DEVICE
            ).total_memory / (1024 ** 3),
            2,
        )
    else:
        info["gpu_name"] = None
        info["gpu_memory_gb"] = None

    return info


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained Transformer on the selected device.

    CUDA is used whenever available; otherwise CPU.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Transformer checkpoint not found: {MODEL_PATH}"
        )

    model = TabularTransformer(
        num_features=NUM_FEATURES,
        embedding_dim=64,
        num_heads=4,
        num_layers=3,
        feedforward_dim=128,
        dropout=0.1,
        num_classes=2,
    )

    # Load checkpoint onto CPU first, then transfer
    # the fully initialized model to CUDA if available.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(model, features):
    """
    Run Transformer inference.

    Supports:
        Single sample: (147,)
        Batch:         (N, 147)

    Returns:
        predicted_class: CPU tensor
        confidence:      CPU tensor
        probabilities:   CPU tensor

    The model computation runs on the model's device.
    """

    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(
            features,
            dtype=torch.float32,
        )
    else:
        features = features.to(
            dtype=torch.float32
        )

    if features.ndim == 1:
        features = features.unsqueeze(0)

    if features.ndim != 2:
        raise ValueError(
            "Features must have shape (147,) "
            "or (N, 147)."
        )

    if features.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features, "
            f"received {features.shape[1]}."
        )

    # Detect the actual model device rather than assuming
    # the global DEVICE still matches the model.
    model_device = next(
        model.parameters()
    ).device

    features = features.to(
        device=model_device,
        dtype=torch.float32,
    )

    model.eval()

    with torch.inference_mode():
        logits = model(features)

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1,
        )

    return (
        predicted_class.cpu(),
        confidence.cpu(),
        probabilities.cpu(),
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("TRANSFORMER INFERENCE")
    print("=" * 70)

    info = get_device_info()

    print(f"\nPyTorch version : {info['pytorch_version']}")
    print(f"CUDA build      : {info['cuda_version']}")
    print(f"Selected device : {info['device']}")

    if info["device_type"] == "cuda":
        print(f"GPU              : {info['gpu_name']}")
        print(
            f"GPU memory       : "
            f"{info['gpu_memory_gb']} GB"
        )
    else:
        print("Running on CPU.")

    print("\nLoading trained Transformer...")

    model = load_model()

    print("Model loaded successfully.")
    print(
        "Model device:",
        next(model.parameters()).device,
    )

    # Synthetic input checks the inference pipeline.
    # It is NOT a model accuracy evaluation.
    test_sample = torch.randn(
        NUM_FEATURES,
        dtype=torch.float32,
    )

    prediction, confidence, probabilities = predict(
        model,
        test_sample,
    )

    predicted_class = int(prediction[0].item())

    class_name = (
        "Normal" if predicted_class == 0 else "Attack"
    )

    print("\n" + "-" * 70)
    print("SINGLE-SAMPLE PREDICTION")
    print("-" * 70)

    print(f"Predicted class : {predicted_class}")
    print(f"Prediction      : {class_name}")
    print(f"Confidence      : {confidence[0].item():.4f}")

    print(
        f"Normal          : "
        f"{probabilities[0, 0].item():.4f}"
    )

    print(
        f"Attack          : "
        f"{probabilities[0, 1].item():.4f}"
    )

    # Batch inference check.
    batch = torch.randn(
        8,
        NUM_FEATURES,
        dtype=torch.float32,
    )

    batch_predictions, batch_confidence, batch_probs = predict(
        model,
        batch,
    )

    assert batch_predictions.shape == (8,)
    assert batch_confidence.shape == (8,)
    assert batch_probs.shape == (8, 2)

    print("\nBatch inference : PASSED (8 samples)")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        print(
            "GPU memory allocated:",
            round(
                torch.cuda.memory_allocated(
                    DEVICE
                ) / (1024 ** 2),
                2,
            ),
            "MB",
        )

    print("\n" + "=" * 70)
    print("TRANSFORMER INFERENCE TEST COMPLETED")
    print("=" * 70)
