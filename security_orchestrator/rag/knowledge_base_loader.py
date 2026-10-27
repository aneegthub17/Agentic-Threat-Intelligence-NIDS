
from pathlib import Path

from security_orchestrator.rag.document_store import DocumentStore


def load_knowledge_base():
    """Load and validate the local cybersecurity knowledge base."""
    knowledge_base_path = (
        Path(__file__).resolve().parent
        / "knowledge_base"
        / "documents.json"
    )

    store = DocumentStore(knowledge_base_path)
    return store.load_documents()
