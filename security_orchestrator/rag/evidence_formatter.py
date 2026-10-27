
class EvidenceFormatter:
    """Format retrieved documents as traceable incident evidence."""

    REQUIRED_FIELDS = {
        "document_id",
        "title",
        "source",
        "content",
        "score",
    }

    @staticmethod
    def format_results(results):
        if not isinstance(results, list):
            raise TypeError("results must be a list")

        evidence = []

        for result in results:
            if not isinstance(result, dict):
                raise ValueError("Each result must be a dictionary")

            missing = EvidenceFormatter.REQUIRED_FIELDS - result.keys()
            if missing:
                raise ValueError(
                    f"Retrieved result is missing fields: {sorted(missing)}"
                )

            item = {
                "document_id": result["document_id"],
                "title": result["title"],
                "source": result["source"],
                "source_url": result.get("source_url", ""),
                "content": result["content"],
                "matched_terms": result.get("matched_terms", []),
                "retrieval_score": result["score"],
                "evidence_type": "retrieved_reference",
            }

            evidence.append(item)

        return evidence
