
import torch


class FeatureExplainer:
    """Generate local feature attributions for a PyTorch classifier."""

    def __init__(self, model):
        self.model = model

    def explain(self, features, feature_names=None, target_class=None,
                top_k=10):
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.detach().clone().float()

        if features.dim() == 1:
            features = features.unsqueeze(0)

        if features.dim() != 2 or features.shape[0] != 1:
            raise ValueError("Expected one sample with shape [1, n_features]")

        num_features = features.shape[1]

        if num_features == 0:
            raise ValueError("At least one feature is required")

        if feature_names is None:
            feature_names = [
                f"feature_{i}" for i in range(num_features)
            ]

        if len(feature_names) != num_features:
            raise ValueError(
                "Number of feature names must match input features"
            )

        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Feature names must be unique")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = features.device

        features = features.to(device)
        features.requires_grad_(True)

        was_training = self.model.training
        self.model.eval()

        try:
            # Do not use torch.no_grad(): gradients are required for XAI.
            with torch.enable_grad():
                logits = self.model(features)

                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise ValueError(
                        "Model must return logits with shape [1, n_classes]"
                    )

                num_classes = logits.shape[1]

                if num_classes < 2:
                    raise ValueError("Model must have at least two classes")

                predicted_class = int(logits.argmax(dim=1).item())

                if target_class is None:
                    target_class = predicted_class

                if (
                    not isinstance(target_class, int)
                    or isinstance(target_class, bool)
                    or not 0 <= target_class < num_classes
                ):
                    raise ValueError("target_class is outside the class range")

                target_logit = logits[0, target_class]

                gradients = torch.autograd.grad(
                    outputs=target_logit,
                    inputs=features,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                # Gradient × input attribution.
                attributions = (features * gradients)[0].detach().cpu()

                probabilities = torch.softmax(
                    logits.detach(), dim=1
                )[0].cpu()

        finally:
            self.model.train(was_training)

        scores = attributions.tolist()

        feature_attributions = [
            {
                "feature": str(name),
                "attribution": float(score),
            }
            for name, score in zip(feature_names, scores)
        ]

        top_features = sorted(
            feature_attributions,
            key=lambda item: abs(item["attribution"]),
            reverse=True,
        )[:min(top_k, num_features)]

        return {
            "method": "gradient_x_input",
            "predicted_class": predicted_class,
            "predicted_label": (
                "Normal" if predicted_class == 0 else
                "Attack" if predicted_class == 1 else
                f"Class_{predicted_class}"
            ),
            "target_class": target_class,
            "target_label": (
                "Normal" if target_class == 0 else
                "Attack" if target_class == 1 else
                f"Class_{target_class}"
            ),
            "class_probabilities": [
                float(value) for value in probabilities.tolist()
            ],
            "feature_attributions": feature_attributions,
            "top_features": top_features,
            "interpretation_note": (
                "Attributions are local gradient-times-input scores for "
                "the selected class logit. Positive scores indicate a "
                "positive local contribution under this method; negative "
                "scores indicate a negative local contribution. They are "
                "not causal explanations or calibrated risk scores."
            ),
        }
