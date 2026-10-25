
import unittest

import pandas as pd
import torch

from detection_engine.inference.predict import load_model, predict
from security_orchestrator.xai.feature_explainer import FeatureExplainer


class TestRealExplanation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = "datasets/processed/test_processed.csv"
        cls.model = load_model()
        cls.explainer = FeatureExplainer(cls.model)

        # Read only one row; the test CSV is large.
        sample = pd.read_csv(cls.data_path, nrows=1)

        if "target" not in sample.columns:
            raise ValueError("Processed dataset must contain target")

        cls.feature_names = [
            column for column in sample.columns if column != "target"
        ]
        cls.actual_label = int(sample["target"].iloc[0])
        cls.features = torch.tensor(
            sample[cls.feature_names].to_numpy(),
            dtype=torch.float32,
        )

    def test_real_model_explanation(self):
        prediction, confidence, probabilities = predict(
            self.model, self.features
        )

        result = self.explainer.explain(
            self.features,
            feature_names=self.feature_names,
            top_k=10,
        )

        self.assertEqual(result["predicted_class"], prediction.item())
        self.assertEqual(len(result["feature_attributions"]), 147)
        self.assertEqual(len(result["top_features"]), 10)
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(item["attribution"])).item()
                for item in result["feature_attributions"]
            )
        )

        print("\nReal Transformer XAI result")
        print("Actual label:", self.actual_label)
        print("Prediction:", result["predicted_label"])
        print("Confidence:", confidence.item())
        print("Probabilities:", probabilities.tolist())
        print("Top contributing features:")
        for item in result["top_features"]:
            print(
                f"  {item['feature']}: "
                f"{item['attribution']:.6f}"
            )


if __name__ == "__main__":
    unittest.main()
