
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from security_orchestrator.orchestrator import SecurityOrchestrator
from security_orchestrator.agents.memory_agent import MemoryAgent
from security_orchestrator.rag.retriever import build_local_retriever


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
        self.assertEqual(
            result["memory_record"]["prediction"], "Attack"
        )

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
        self.assertEqual(
            result["memory_record"]["prediction"], "Normal"
        )

    @patch("security_orchestrator.orchestrator.predict")
    def test_xai_explanation_is_included_in_report(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([1]),
            torch.tensor([0.91]),
            torch.tensor([[0.09, 0.91]]),
        )

        explanation = {
            "method": "gradient_x_input",
            "predicted_class": 1,
            "predicted_label": "Attack",
            "top_features": [
                {"feature": "src_packets", "attribution": 0.75}
            ],
        }

        self.orchestrator.explainer = unittest.mock.Mock()
        self.orchestrator.explainer.explain.return_value = explanation

        result = self.orchestrator.analyze(
            [0.0] * 147,
            feature_names=[f"feature_{i}" for i in range(147)],
        )

        self.assertEqual(result["xai_status"], "completed")
        self.assertEqual(result["xai_explanation"], explanation)
        self.assertEqual(
            result["report"]["xai_explanation"], explanation
        )
        self.orchestrator.explainer.explain.assert_called_once()

    @patch("security_orchestrator.orchestrator.predict")
    def test_xai_failure_does_not_block_detection(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([0]),
            torch.tensor([0.85]),
            torch.tensor([[0.85, 0.15]]),
        )

        self.orchestrator.explainer = unittest.mock.Mock()
        self.orchestrator.explainer.explain.side_effect = RuntimeError(
            "Test explanation failure"
        )

        result = self.orchestrator.analyze([0.0] * 147)

        self.assertEqual(result["detection"]["prediction"], "Normal")
        self.assertEqual(result["xai_status"], "unavailable")
        self.assertEqual(
            result["xai_explanation"]["reason"],
            "Explanation failed: RuntimeError",
        )
        self.assertEqual(
            result["report"]["detection"]["prediction"], "Normal"
        )
        self.assertEqual(
            result["report"]["xai_explanation"]["status"],
            "unavailable",
        )

    @patch("security_orchestrator.orchestrator.predict")
    def test_xai_receives_feature_names(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([0]),
            torch.tensor([0.85]),
            torch.tensor([[0.85, 0.15]]),
        )

        self.orchestrator.explainer = unittest.mock.Mock()
        self.orchestrator.explainer.explain.return_value = {
            "method": "gradient_x_input",
            "top_features": [],
        }

        names = [f"feature_{i}" for i in range(147)]

        self.orchestrator.analyze(
            [0.0] * 147,
            feature_names=names,
        )

        self.orchestrator.explainer.explain.assert_called_once_with(
            [0.0] * 147,
            feature_names=names,
            top_k=10,
        )

    @patch("security_orchestrator.orchestrator.predict")
    def test_rag_evidence_flows_through_orchestrator(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([1]),
            torch.tensor([0.91]),
            torch.tensor([[0.09, 0.91]]),
        )

        fake_evidence = {
            "document_id": "nist-800-61-r3",
            "title": "NIST SP 800-61 Rev. 3: Incident Response",
            "source": "NIST Computer Security Resource Center",
            "source_url": (
                "https://csrc.nist.gov/pubs/sp/800/61/r3/final"
            ),
            "content": "Incident response guidance.",
            "matched_terms": ["incident", "response"],
            "retrieval_score": 0.75,
            "evidence_type": "retrieved_reference",
        }

        fake_retriever = Mock()
        fake_retriever.retrieve.return_value = [
            {
                "document_id": "nist-800-61-r3",
                "title": "NIST SP 800-61 Rev. 3: Incident Response",
                "source": "NIST Computer Security Resource Center",
                "source_url": (
                    "https://csrc.nist.gov/pubs/sp/800/61/r3/final"
                ),
                "content": "Incident response guidance.",
                "matched_terms": ["incident", "response"],
                "score": 0.75,
            }
        ]

        orchestrator = SecurityOrchestrator(
            model=object(),
            memory_agent=self.orchestrator.memory_agent,
            retriever=fake_retriever,
        )

        result = orchestrator.analyze(
            [0.0] * 147,
            traffic_features={
                "investigation_context": (
                    "incident response cybersecurity risk management"
                )
            },
        )

        self.assertTrue(
            result["threat_analysis"]["retrieval_performed"]
        )
        self.assertEqual(
            result["threat_analysis"]["sources"][0]["document_id"],
            fake_evidence["document_id"],
        )
        self.assertEqual(
            result["threat_analysis"]["sources"][0]["source_url"],
            fake_evidence["source_url"],
        )
        fake_retriever.retrieve.assert_called_once()

    @patch("security_orchestrator.orchestrator.predict")
    def test_rag_evidence_is_included_in_final_report(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([1]),
            torch.tensor([0.91]),
            torch.tensor([[0.09, 0.91]]),
        )

        fake_retriever = Mock()
        fake_retriever.retrieve.return_value = [
            {
                "document_id": "nist-800-61-r3",
                "title": "NIST SP 800-61 Rev. 3: Incident Response",
                "source": "NIST Computer Security Resource Center",
                "source_url": (
                    "https://csrc.nist.gov/pubs/sp/800/61/r3/final"
                ),
                "content": "Incident response guidance.",
                "matched_terms": ["incident", "response"],
                "score": 0.75,
            }
        ]

        orchestrator = SecurityOrchestrator(
            model=object(),
            memory_agent=self.orchestrator.memory_agent,
            retriever=fake_retriever,
        )

        result = orchestrator.analyze(
            [0.0] * 147,
            traffic_features={
                "investigation_context": (
                    "incident response cybersecurity risk management"
                )
            },
        )

        report_threat = result["report"]["threat_analysis"]

        self.assertTrue(report_threat["retrieval_performed"])
        self.assertEqual(len(report_threat["sources"]), 1)
        self.assertEqual(
            report_threat["sources"][0]["document_id"],
            "nist-800-61-r3",
        )
        self.assertEqual(
            report_threat["sources"][0]["source_url"],
            "https://csrc.nist.gov/pubs/sp/800/61/r3/final",
        )
        fake_retriever.retrieve.assert_called_once()

    @patch("security_orchestrator.orchestrator.predict")
    def test_real_local_rag_reaches_final_report(self, mock_predict):
        mock_predict.return_value = (
            torch.tensor([1]),
            torch.tensor([0.91]),
            torch.tensor([[0.09, 0.91]]),
        )

        retriever = build_local_retriever()

        orchestrator = SecurityOrchestrator(
            model=object(),
            memory_agent=self.orchestrator.memory_agent,
            retriever=retriever,
        )

        result = orchestrator.analyze(
            [0.0] * 147,
            traffic_features={
                "investigation_context": (
                    "incident response cybersecurity risk management"
                )
            },
        )

        report_threat = result["report"]["threat_analysis"]

        self.assertTrue(report_threat["retrieval_performed"])
        self.assertGreater(len(report_threat["sources"]), 0)
        self.assertEqual(
            report_threat["sources"][0]["document_id"],
            "nist-800-61-r3",
        )
        self.assertTrue(
            report_threat["sources"][0]["source_url"].startswith("https://")
        )


if __name__ == "__main__":
    unittest.main()
