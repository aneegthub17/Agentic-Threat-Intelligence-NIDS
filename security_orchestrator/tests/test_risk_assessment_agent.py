
import unittest

from security_orchestrator.agents.risk_assessment_agent import (
    RiskAssessmentAgent,
)


class TestRiskAssessmentAgent(unittest.TestCase):

    def setUp(self):
        self.agent = RiskAssessmentAgent()

    def test_attack_requires_review(self):
        record = {
            "status": "valid",
            "prediction": "Attack",
            "confidence": 0.99,
        }

        result = self.agent.process(record)

        self.assertEqual(result["risk_level"], "MEDIUM")
        self.assertTrue(result["requires_review"])
        self.assertEqual(
            result["assessment_type"], "preliminary_rule_based"
        )

    def test_normal_gets_preliminary_low_risk(self):
        record = {
            "status": "valid",
            "prediction": "Normal",
            "confidence": 0.85,
        }

        result = self.agent.process(record)

        self.assertEqual(result["risk_level"], "LOW")
        self.assertFalse(result["requires_review"])

    def test_rejects_unvalidated_record(self):
        with self.assertRaises(ValueError):
            self.agent.process({
                "prediction": "Attack",
                "confidence": 0.9,
            })

    def test_rejects_mismatched_threat_analysis(self):
        record = {
            "status": "valid",
            "prediction": "Attack",
            "confidence": 0.9,
        }

        analysis = {"prediction": "Normal"}

        with self.assertRaises(ValueError):
            self.agent.process(record, analysis)


if __name__ == "__main__":
    unittest.main()
