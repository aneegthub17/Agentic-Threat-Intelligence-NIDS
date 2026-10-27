
import re


class LocalRetriever:
    """Retrieve documents using weighted keyword overlap."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "this", "to", "was", "what", "when", "with",
    }

    GENERIC_TERMS = {
        "attack", "network", "intrusion", "detection", "security",
        "cybersecurity", "traffic", "sample", "system",
    }

    def __init__(self, documents, min_score=0.25):
        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        if (
            not isinstance(min_score, (int, float))
            or isinstance(min_score, bool)
            or not 0 <= min_score <= 1
        ):
            raise ValueError("min_score must be between 0 and 1")

        self.documents = documents
        self.min_score = float(min_score)

    @classmethod
    def _tokens(cls, text):
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {
            word for word in words
            if word not in cls.STOP_WORDS and len(word) > 1
        }

    def retrieve(self, query, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")

        query_tokens = self._tokens(query) - self.GENERIC_TERMS

        if not query_tokens:
            return []

        results = []

        for document in self.documents:
            title_tokens = self._tokens(document["title"])
            content_tokens = self._tokens(document["content"])

            title_matches = query_tokens & title_tokens
            content_matches = query_tokens & content_tokens

            matched_terms = title_matches | content_matches

            if not matched_terms:
                continue

            # A title match contributes twice as much as a body match.
            weighted_matches = (
                2 * len(title_matches)
                + len(content_matches - title_matches)
            )

            score = weighted_matches / (2 * len(query_tokens))

            if score < self.min_score:
                continue

            results.append({
                "document_id": document["id"],
                "title": document["title"],
                "source": document["source"],
                "source_url": document.get("source_url", ""),
                "content": document["content"],
                "matched_terms": sorted(matched_terms),
                "score": min(score, 1.0),
            })

        results.sort(
            key=lambda item: (-item["score"], item["document_id"])
        )

        return results[:top_k]


def build_local_retriever(min_score=0.25):
    """Build a retriever using the validated local knowledge base."""
    from security_orchestrator.rag.knowledge_base_loader import (
        load_knowledge_base,
    )

    documents = load_knowledge_base()
    return LocalRetriever(documents, min_score=min_score)
