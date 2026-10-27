
class ThreatIntelligenceAgent:
    """
    Analyse validated detections and optionally retrieve local
    cybersecurity references for additional context.
    """

    def __init__(self, retriever=None):
        self.retriever = retriever

    def process(self, detection_record, traffic_features=None):
        if not isinstance(detection_record, dict):
            raise TypeError("Detection record must be a dictionary.")

        if detection_record.get("status") != "valid":
            raise ValueError("Detection record has not passed validation.")

        prediction = detection_record.get("prediction")

        if prediction not in {"Normal", "Attack"}:
            raise ValueError("Unsupported prediction.")

        if prediction == "Attack":
            summary = (
                "The Transformer classified this traffic sample as an "
                "attack. Further threat-intelligence investigation is "
                "required to identify relevant context."
            )
        else:
            summary = (
                "The Transformer classified this traffic sample as normal. "
                "This result does not guarantee that the traffic is benign."
            )

        feature_summary = {}

        if traffic_features is not None:
            if not isinstance(traffic_features, dict):
                raise TypeError("Traffic features must be a dictionary.")

            feature_summary = {
                str(key): value
                for key, value in traffic_features.items()
                if isinstance(value, (str, int, float, bool))
                or value is None
            }

        evidence = []

        if self.retriever is not None:
            query_parts = []

            for key, value in feature_summary.items():
                # Feature names provide context; use text values only
                # when they may help identify a relevant reference.
                query_parts.append(key.replace("_", " "))

                if isinstance(value, str):
                    query_parts.append(value)

            query = " ".join(query_parts).strip()

            # Retrieve references only when traffic context is available.
            retrieved = (
                self.retriever.retrieve(query, top_k=3)
                if query
                else []
            )

            from security_orchestrator.rag.evidence_formatter import (
                EvidenceFormatter,
            )

            evidence = EvidenceFormatter.format_results(retrieved)

        return {
            "prediction": prediction,
            "summary": summary,
            "observed_features": feature_summary,
            "retrieval_performed": self.retriever is not None,
            "sources": evidence,
        }
