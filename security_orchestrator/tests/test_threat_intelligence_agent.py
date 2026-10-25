
import unittest

from security_orchestrator.agents.threat_intelligence_agent import (
    ThreatIntelligenceAgent,
)


class TestThreatIntelligenceAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ThreatIntelligenceAgent()

    def test_attack_summary(self):
        record = {
            "status": "valid",
            "prediction": "Attack",
        }

        result = self.agent.process(record)

        self.assertEqual(result["prediction"], "Attack")
        self.assertIn("further", result["summary"].lower())
        self.assertFalse(result["retrieval_performed"])
        self.assertEqual(result["sources"], [])

    def test_normal_summary(self):
        record = {
            "status": "valid",
            "prediction": "Normal",
        }

        result = self.agent.process(record)

        self.assertEqual(result["prediction"], "Normal")
        self.assertIn("does not guarantee", result["summary"])

    def test_rejects_unvalidated_record(self):
        with self.assertRaises(ValueError):
            self.agent.process({"prediction": "Attack"})

    def test_includes_supplied_features(self):
        record = {
            "status": "valid",
            "prediction": "Attack",
        }

        result = self.agent.process(
            record,
            {"packet_count": 120, "protocol": "TCP"},
        )

        self.assertEqual(
            result["observed_features"]["packet_count"], 120
        )
        self.assertEqual(
            result["observed_features"]["protocol"], "TCP"
        )


if __name__ == "__main__":
    unittest.main()
