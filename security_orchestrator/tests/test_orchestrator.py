
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from security_orchestrator.orchestrator import SecurityOrchestrator
from security_orchestrator.agents.memory_agent import MemoryAgent


class TestSecurityOrchestrator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        memory = MemoryAgent(
            Path(self.temp_dir.name) / "incidents.jsonl"
        )

        # Inject a placeholder model; prediction is mocked in these tests.
        self.orchestrator = SecurityOrchestrator(
            model=object(),
            memory_agent=memory,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("security_orchestrator.orchestrator.predict")
    def test_attack_flows_through_all_agents(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([1]),
            torch.tensor([0.91]),
            torch.tensor([[0.09, 0.91]]),
        )

        result = self.orchestrator.analyze(
            [0.0] * 147,
            traffic_features={"protocol": "TCP"},
        )

        self.assertEqual(
            result["detection"]["prediction"], "Attack"
        )
        self.assertEqual(
            result["risk_assessment"]["risk_level"], "MEDIUM"
        )
        self.assertTrue(
            result["risk_assessment"]["requires_review"]
        )
        self.assertEqual(
            result["memory_record"]["prediction"], "Attack"
        )
        mock_predict.assert_called_once()

    @patch("security_orchestrator.orchestrator.predict")
    def test_normal_flows_through_all_agents(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([0]),
            torch.tensor([0.85]),
            torch.tensor([[0.85, 0.15]]),
        )

        result = self.orchestrator.analyze([0.0] * 147)

        self.assertEqual(
            result["detection"]["prediction"], "Normal"
        )
        self.assertEqual(
            result["risk_assessment"]["risk_level"], "LOW"
        )
        self.assertEqual(
            result["memory_record"]["prediction"], "Normal"
        )

    @patch("security_orchestrator.orchestrator.predict")
    def test_attack_generates_report(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([1]),
            torch.tensor([0.91]),
            torch.tensor([[0.09, 0.91]]),
        )

        result = self.orchestrator.analyze(
            [0.0] * 147,
            traffic_features={"protocol": "TCP"},
        )

        report = result["report"]

        self.assertEqual(
            report["detection"]["prediction"], "Attack"
        )
        self.assertEqual(
            report["risk_assessment"]["risk_level"], "MEDIUM"
        )
        self.assertEqual(report["report_status"], "preliminary")
        self.assertTrue(report["report_id"])
        self.assertTrue(report["recommendations"])
        self.assertEqual(result["memory_record"]["prediction"], "Attack")

    @patch("security_orchestrator.orchestrator.predict")
    def test_normal_generates_report(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([0]),
            torch.tensor([0.85]),
            torch.tensor([[0.85, 0.15]]),
        )

        result = self.orchestrator.analyze([0.0] * 147)

        report = result["report"]

        self.assertEqual(
            report["detection"]["prediction"], "Normal"
        )
        self.assertEqual(
            report["risk_assessment"]["risk_level"], "LOW"
        )
        self.assertEqual(report["report_status"], "preliminary")
        self.assertTrue(report["report_id"])
        self.assertTrue(report["recommendations"])
        self.assertEqual(result["memory_record"]["prediction"], "Normal")


if __name__ == "__main__":
    unittest.main()
