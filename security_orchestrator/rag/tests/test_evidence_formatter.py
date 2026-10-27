
import unittest

from security_orchestrator.rag.evidence_formatter import EvidenceFormatter


class TestEvidenceFormatter(unittest.TestCase):

    def setUp(self):
        self.sample_result = {
            "document_id": "nist-800-61-r3",
            "title": "NIST Incident Response Guidance",
            "source": "NIST",
            "source_url": "https://csrc.nist.gov/pubs/sp/800/61/r3/final",
            "content": "Incident response supports cybersecurity risk management.",
            "matched_terms": ["incident", "response"],
            "score": 0.75,
        }

    def test_formats_retrieved_evidence(self):
        evidence = EvidenceFormatter.format_results(
            [self.sample_result]
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(
            evidence[0]["document_id"], "nist-800-61-r3"
        )
        self.assertEqual(
            evidence[0]["evidence_type"], "retrieved_reference"
        )

    def test_preserves_source_url(self):
        evidence = EvidenceFormatter.format_results(
            [self.sample_result]
        )

        self.assertEqual(
            evidence[0]["source_url"],
            self.sample_result["source_url"],
        )

    def test_preserves_score_and_matched_terms(self):
        evidence = EvidenceFormatter.format_results(
            [self.sample_result]
        )

        self.assertEqual(evidence[0]["retrieval_score"], 0.75)
        self.assertEqual(
            evidence[0]["matched_terms"],
            ["incident", "response"],
        )

    def test_empty_results(self):
        self.assertEqual(
            EvidenceFormatter.format_results([]), []
        )

    def test_rejects_non_list_input(self):
        with self.assertRaises(TypeError):
            EvidenceFormatter.format_results("invalid")

    def test_rejects_missing_required_fields(self):
        incomplete = {
            "document_id": "test-001",
            "title": "Test document",
        }

        with self.assertRaises(ValueError):
            EvidenceFormatter.format_results([incomplete])

    def test_rejects_non_dictionary_result(self):
        with self.assertRaises(ValueError):
            EvidenceFormatter.format_results(["invalid"])


if __name__ == "__main__":
    unittest.main()
