
import unittest
import torch

from security_orchestrator.agents.detection_agent import DetectionAgent


class TestDetectionAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DetectionAgent()

    def test_valid_attack_prediction(self):
        result = self.agent.process(
            torch.tensor([1]),
            torch.tensor([0.9]),
            torch.tensor([[0.1, 0.9]])
        )

        self.assertEqual(result["prediction"], "Attack")
        self.assertEqual(result["status"], "valid")
        self.assertAlmostEqual(result["confidence"], 0.9)

    def test_valid_normal_prediction(self):
        result = self.agent.process(
            torch.tensor([0]),
            torch.tensor([0.8]),
            torch.tensor([[0.8, 0.2]])
        )

        self.assertEqual(result["prediction"], "Normal")

    def test_invalid_class(self):
        with self.assertRaises(ValueError):
            self.agent.process(
                torch.tensor([2]),
                torch.tensor([0.9]),
                torch.tensor([[0.1, 0.9]])
            )

    def test_inconsistent_confidence(self):
        with self.assertRaises(ValueError):
            self.agent.process(
                torch.tensor([1]),
                torch.tensor([0.7]),
                torch.tensor([[0.1, 0.9]])
            )


if __name__ == "__main__":
    unittest.main()
