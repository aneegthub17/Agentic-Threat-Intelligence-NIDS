
import unittest
from pathlib import Path

from security_orchestrator.rag.document_store import DocumentStore
from security_orchestrator.rag.retriever import LocalRetriever


class TestLocalRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        knowledge_path = (
            Path(__file__).resolve().parents[1]
            / "knowledge_base"
            / "documents.json"
        )
        cls.documents = DocumentStore(knowledge_path).load_documents()
        cls.retriever = LocalRetriever(cls.documents)

    def test_loads_knowledge_base(self):
        self.assertEqual(len(self.documents), 3)

    def test_retrieves_relevant_document(self):
        results = self.retriever.retrieve(
            "incident response cybersecurity risk management"
        )
        self.assertTrue(results)
        self.assertEqual(results[0]["document_id"], "nist-800-61-r3")

    def test_returns_source_metadata(self):
        results = self.retriever.retrieve(
            "adversary techniques tactics"
        )
        self.assertTrue(results)
        self.assertIn("source", results[0])
        self.assertIn("title", results[0])
        self.assertIn("matched_terms", results[0])
        self.assertIn("source_url", results[0])
        self.assertTrue(
            results[0]["source_url"].startswith("https://")
        )
    def test_no_match_returns_empty_list(self):
        results = self.retriever.retrieve(
            "volcanic geology astronomy"
        )
        self.assertEqual(results, [])

    def test_rejects_empty_query(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve(" ")

    def test_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("network traffic", top_k=0)


if __name__ == "__main__":
    unittest.main()
