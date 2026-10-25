
class ThreatIntelligenceAgent:
    """
    Analyses a validated NIDS detection record.

    This initial version does not perform external retrieval or
    infer a specific attack family from a binary prediction.
    """

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

        # Only summarise features actually supplied by the caller.
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

        return {
            "prediction": prediction,
            "summary": summary,
            "observed_features": feature_summary,
            "retrieval_performed": False,
            "sources": [],
        }
