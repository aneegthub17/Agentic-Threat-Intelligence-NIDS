
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from security_orchestrator.orchestrator import SecurityOrchestrator
from security_orchestrator.agents.memory_agent import MemoryAgent


class TestRealInference(unittest.TestCase):

    def test_real_sample_through_orchestrator(self):
        df = pd.read_csv(
            "datasets/processed/test_processed.csv",
            nrows=1,
        )

        features = df.drop(columns=["target"]).iloc[0].to_numpy(
            dtype="float32"
        )

        actual_target = int(df["target"].iloc[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = MemoryAgent(
                Path(temp_dir) / "integration_memory.jsonl"
            )

            orchestrator = SecurityOrchestrator(
                memory_agent=memory
            )

            result = orchestrator.analyze(features)

            self.assertIn(
                result["detection"]["prediction"],
                {"Normal", "Attack"},
            )
            self.assertIn(
                result["risk_assessment"]["risk_level"],
                {"LOW", "MEDIUM", "HIGH"},
            )

            self.assertEqual(
                result["detection"]["predicted_class"],
                actual_target,
                "The first sample's prediction differs from its label.",
            )

            self.assertEqual(len(memory.get_recent()), 1)

            print("\nReal-model orchestrator result:")
            print("Actual label:", actual_target)
            print("Detection:", result["detection"])
            print("Risk:", result["risk_assessment"])


if __name__ == "__main__":
    unittest.main()
