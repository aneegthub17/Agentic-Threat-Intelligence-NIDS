
from datetime import datetime, timezone

import torch


CLASS_NAMES = {
    0: "Normal",
    1: "Attack",
}


class DetectionAgent:
    """
    Validates and standardizes the output of the existing
    Transformer-based detection engine.

    This agent does not retrain or modify the model.
    """

    def process(self, prediction, confidence, probabilities):
        # Validate the prediction output.
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Prediction must be a PyTorch tensor.")

        if prediction.numel() != 1:
            raise ValueError("Expected exactly one prediction.")

        predicted_class = int(prediction.item())

        if predicted_class not in CLASS_NAMES:
            raise ValueError("Unexpected predicted class.")

        # Validate confidence.
        if not isinstance(confidence, torch.Tensor):
            raise TypeError("Confidence must be a PyTorch tensor.")

        if confidence.numel() != 1:
            raise ValueError("Expected exactly one confidence value.")

        confidence_value = float(confidence.item())

        if not 0.0 <= confidence_value <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        # Validate class probabilities.
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("Probabilities must be a PyTorch tensor.")

        if probabilities.shape != (1, 2):
            raise ValueError(
                "Expected probabilities with shape (1, 2)."
            )

        probs = probabilities[0].detach().cpu().tolist()

        if not all(
            isinstance(p, (int, float))
            and 0.0 <= p <= 1.0
            for p in probs
        ):
            raise ValueError("Invalid class probabilities.")

        if abs(sum(probs) - 1.0) > 1e-4:
            raise ValueError(
                "Class probabilities must sum to approximately 1."
            )

        if abs(max(probs) - confidence_value) > 1e-4:
            raise ValueError(
                "Confidence does not match the maximum class probability."
            )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predicted_class": predicted_class,
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": confidence_value,
            "probabilities": {
                "normal": float(probs[0]),
                "attack": float(probs[1]),
            },
            "status": "valid",
        }
