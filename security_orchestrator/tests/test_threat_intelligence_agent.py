
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

    def test_retrieval_is_used_when_configured(self):

        class FakeRetriever:
            def __init__(self):
                self.called = False
                self.query = None
                self.top_k = None

            def retrieve(self, query, top_k=3):
                self.called = True
                self.query = query
                self.top_k = top_k

                return [
                    {
                        "document_id": "nist-800-61-r3",
                        "title": "NIST Incident Response Guidance",
                        "source": "NIST",
                        "source_url": (
                            "https://csrc.nist.gov/pubs/sp/800/61/r3/final"
                        ),
                        "content": "Incident response guidance.",
                        "matched_terms": ["incident"],
                        "score": 0.5,
                    }
                ]

        retriever = FakeRetriever()
        agent = ThreatIntelligenceAgent(retriever=retriever)

        result = agent.process(
            {
                "status": "valid",
                "prediction": "Attack",
            },
            traffic_features={
                "incident_response": "incident response",
            },
        )

        self.assertTrue(retriever.called)
        self.assertIn("incident response", retriever.query)
        self.assertEqual(retriever.top_k, 3)
        self.assertTrue(result["retrieval_performed"])

        self.assertEqual(
            result["sources"][0]["document_id"],
            "nist-800-61-r3",
        )
        self.assertEqual(
            result["sources"][0]["source_url"],
            "https://csrc.nist.gov/pubs/sp/800/61/r3/final",
        )

    def test_empty_retrieval_returns_no_sources(self):

        class EmptyRetriever:
            def retrieve(self, query, top_k=3):
                return []

        agent = ThreatIntelligenceAgent(retriever=EmptyRetriever())

        result = agent.process(
            {
                "status": "valid",
                "prediction": "Normal",
            }
        )

        self.assertTrue(result["retrieval_performed"])
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
