
import unittest

from security_orchestrator.reporting.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = ReportGenerator()

        self.detection = {
            "status": "valid",
            "prediction": "Attack",
            "predicted_class": 1,
            "confidence": 0.91,
            "probabilities": {"normal": 0.09, "attack": 0.91},
        }

        self.threat_analysis = {
            "summary": "Potential malicious traffic detected.",
            "observed_features": {},
            "retrieval_performed": False,
            "sources": [],
        }

        self.risk_assessment = {
            "risk_level": "MEDIUM",
            "rationale": "Potential attack detected.",
            "assessment_type": "preliminary_rule_based",
            "requires_review": True,
        }

    def test_attack_report(self):
        report = self.generator.generate(
            self.detection,
            self.threat_analysis,
            self.risk_assessment,
        )

        self.assertEqual(report["detection"]["prediction"], "Attack")
        self.assertEqual(report["risk_assessment"]["risk_level"], "MEDIUM")
        self.assertEqual(report["report_status"], "preliminary")
        self.assertTrue(report["recommendations"])
        self.assertEqual(report["threat_analysis"]["sources"], [])

    def test_normal_report(self):
        detection = {
            **self.detection,
            "prediction": "Normal",
            "predicted_class": 0,
            "confidence": 0.90,
        }

        risk = {
            **self.risk_assessment,
            "risk_level": "LOW",
            "requires_review": False,
        }

        report = self.generator.generate(
            detection, self.threat_analysis, risk
        )

        self.assertEqual(report["detection"]["prediction"], "Normal")
        self.assertEqual(report["risk_assessment"]["risk_level"], "LOW")

    def test_invalid_detection_rejected(self):
        detection = {**self.detection, "status": "invalid"}

        with self.assertRaises(ValueError):
            self.generator.generate(
                detection, self.threat_analysis, self.risk_assessment
            )

    def test_invalid_confidence_rejected(self):
        detection = {**self.detection, "confidence": 1.5}

        with self.assertRaises(ValueError):
            self.generator.generate(
                detection, self.threat_analysis, self.risk_assessment
            )

    def test_report_has_unique_ids(self):
        first = self.generator.generate(
            self.detection, self.threat_analysis, self.risk_assessment
        )
        second = self.generator.generate(
            self.detection, self.threat_analysis, self.risk_assessment
        )

        self.assertNotEqual(first["report_id"], second["report_id"])


if __name__ == "__main__":
    unittest.main()
