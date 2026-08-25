import os
import sys

import torch
import torch.nn.functional as F

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

from detection_engine.models.tabular_transformer import TabularTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "detection_engine/models/best_transformer.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = TabularTransformer(
        num_features=147,
        embedding_dim=64,
        num_heads=4,
        num_layers=3,
        feedforward_dim=128,
        dropout=0.1,
        num_classes=2
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(model, features):

    # Convert input to tensor.
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(
            features,
            dtype=torch.float32
        )

    # Add batch dimension if a single sample is supplied.
    if features.dim() == 1:
        features = features.unsqueeze(0)

    features = features.to(DEVICE)

    with torch.no_grad():

        logits = model(features)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        )

        confidence = torch.max(
            probabilities,
            dim=1
        ).values

    return (
        predicted_class.cpu(),
        confidence.cpu(),
        probabilities.cpu()
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("TRANSFORMER INFERENCE")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print("\nLoading trained Transformer...")

    model = load_model()

    print("✓ Model loaded successfully.")

    # --------------------------------------------------------
    # Test with one realistic processed feature vector.
    # --------------------------------------------------------

    test_sample = torch.randn(
        147,
        dtype=torch.float32
    )

    prediction, confidence, probabilities = predict(
        model,
        test_sample
    )

    predicted_class = prediction.item()
    predicted_confidence = confidence.item()

    class_name = (
        "Normal"
        if predicted_class == 0
        else "Attack"
    )

    print("\n" + "-" * 70)
    print("PREDICTION")
    print("-" * 70)

    print(f"\nPredicted class : {predicted_class}")
    print(f"Prediction      : {class_name}")
    print(
        f"Confidence      : {predicted_confidence:.4f}"
    )

    print("\nClass probabilities:")

    print(
        f"Normal : {probabilities[0][0].item():.4f}"
    )

    print(
        f"Attack : {probabilities[0][1].item():.4f}"
    )

    print("\n" + "=" * 70)
    print("TRANSFORMER INFERENCE TEST COMPLETED")
    print("=" * 70)