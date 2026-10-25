
import unittest

import torch
import torch.nn as nn

from security_orchestrator.xai.feature_explainer import FeatureExplainer


class TinyClassifier(nn.Module):
    """Small deterministic model for testing feature attribution."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)

        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([
                    [1.0, 0.0, -1.0],
                    [0.0, 2.0, 1.0],
                ])
            )

    def forward(self, features):
        return self.linear(features)


class TestFeatureExplainer(unittest.TestCase):

    def setUp(self):
        self.model = TinyClassifier()
        self.explainer = FeatureExplainer(self.model)

    def test_returns_attributions_for_each_feature(self):
        result = self.explainer.explain(
            [1.0, 2.0, 3.0],
            feature_names=["a", "b", "c"],
        )

        self.assertEqual(len(result["feature_attributions"]), 3)
        self.assertEqual(len(result["top_features"]), 3)
        self.assertEqual(result["method"], "gradient_x_input")

    def test_top_features_sorted_by_absolute_attribution(self):
        result = self.explainer.explain(
            [1.0, 2.0, 3.0],
            feature_names=["a", "b", "c"],
        )

        scores = [
            abs(item["attribution"])
            for item in result["top_features"]
        ]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rejects_mismatched_feature_names(self):
        with self.assertRaises(ValueError):
            self.explainer.explain(
                [1.0, 2.0, 3.0],
                feature_names=["a", "b"],
            )

    def test_rejects_invalid_target_class(self):
        with self.assertRaises(ValueError):
            self.explainer.explain(
                [1.0, 2.0, 3.0],
                target_class=5,
            )

    def test_preserves_model_training_state(self):
        self.model.train()

        self.explainer.explain([1.0, 2.0, 3.0])

        self.assertTrue(self.model.training)

    def test_produces_finite_attributions(self):
        result = self.explainer.explain([1.0, 2.0, 3.0])

        for item in result["feature_attributions"]:
            self.assertTrue(
                torch.isfinite(torch.tensor(item["attribution"])).item()
            )


if __name__ == "__main__":
    unittest.main()
