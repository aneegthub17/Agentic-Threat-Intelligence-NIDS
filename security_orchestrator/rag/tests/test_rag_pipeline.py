
import unittest
from pathlib import Path

from security_orchestrator.rag.document_store import DocumentStore
from security_orchestrator.rag.retriever import LocalRetriever
from security_orchestrator.rag.evidence_formatter import EvidenceFormatter


class TestLocalRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        knowledge_path = (
            Path(__file__).resolve().parents[1]
            / "knowledge_base"
            / "documents.json"
        )
        cls.documents = DocumentStore(knowledge_path).load_documents()
        cls.retriever = LocalRetriever(cls.documents)

    def test_retrieval_flows_into_evidence_formatter(self):
        results = self.retriever.retrieve(
            "incident response cybersecurity risk management",
            top_k=2,
        )
        evidence = EvidenceFormatter.format_results(results)

        self.assertTrue(evidence)
        self.assertEqual(
            evidence[0]["document_id"], "nist-800-61-r3"
        )
        self.assertTrue(evidence[0]["source_url"].startswith("https://"))
        self.assertEqual(
            evidence[0]["evidence_type"], "retrieved_reference"
        )

    def test_unrelated_query_produces_no_evidence(self):
        results = self.retriever.retrieve(
            "volcanic geology astronomy"
        )
        evidence = EvidenceFormatter.format_results(results)

        self.assertEqual(evidence, [])

    def test_evidence_contains_traceability_fields(self):
        results = self.retriever.retrieve(
            "adversary techniques tactics",
            top_k=2,
        )
        evidence = EvidenceFormatter.format_results(results)

        self.assertTrue(evidence)

        for item in evidence:
            self.assertIn("document_id", item)
            self.assertIn("source", item)
            self.assertIn("source_url", item)
            self.assertIn("content", item)
            self.assertIn("retrieval_score", item)
            self.assertIn("matched_terms", item)


if __name__ == "__main__":
    unittest.main()
