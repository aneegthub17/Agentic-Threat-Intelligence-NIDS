
import json
import tempfile
import unittest
from pathlib import Path

from security_orchestrator.agents.memory_agent import MemoryAgent


class TestMemoryAgent(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "memory.jsonl"
        self.agent = MemoryAgent(path)

        self.detection = {
            "status": "valid",
            "prediction": "Attack",
            "confidence": 0.91,
        }

        self.risk = {"risk_level": "MEDIUM"}

        self.threat = {
            "summary": "Further threat investigation is required."
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_store_and_retrieve(self):
        stored = self.agent.store(
            self.detection, self.risk, self.threat
        )

        recent = self.agent.get_recent()

        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["prediction"], "Attack")
        self.assertEqual(recent[0]["risk_level"], "MEDIUM")
        self.assertEqual(stored["summary"], self.threat["summary"])

    def test_recent_records_are_newest_first(self):
        self.agent.store(self.detection, self.risk)
        self.agent.store(
            {**self.detection, "prediction": "Normal"},
            {"risk_level": "LOW"},
        )

        recent = self.agent.get_recent(limit=2)

        self.assertEqual(recent[0]["prediction"], "Normal")
        self.assertEqual(recent[1]["prediction"], "Attack")

    def test_empty_memory(self):
        self.assertEqual(self.agent.get_recent(), [])

    def test_rejects_unvalidated_detection(self):
        with self.assertRaises(ValueError):
            self.agent.store(
                {"prediction": "Attack", "confidence": 0.9},
                self.risk,
            )

    def test_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            self.agent.get_recent(limit=0)


if __name__ == "__main__":
    unittest.main()
