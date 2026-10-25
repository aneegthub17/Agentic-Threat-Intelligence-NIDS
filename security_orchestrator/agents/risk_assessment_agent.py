
class RiskAssessmentAgent:
    """
    Assigns an initial risk level using the detection result
    and explicitly supplied evidence.

    This is a preliminary rule-based assessment, not a calibrated
    measure of real-world attack severity.
    """

    def process(self, detection_record, threat_analysis=None):
        if not isinstance(detection_record, dict):
            raise TypeError("Detection record must be a dictionary.")

        if detection_record.get("status") != "valid":
            raise ValueError("Detection record has not passed validation.")

        prediction = detection_record.get("prediction")

        if prediction not in {"Normal", "Attack"}:
            raise ValueError("Unsupported prediction.")

        if threat_analysis is not None:
            if not isinstance(threat_analysis, dict):
                raise TypeError("Threat analysis must be a dictionary.")

            if threat_analysis.get("prediction") != prediction:
                raise ValueError(
                    "Threat analysis and detection prediction do not match."
                )

        confidence = detection_record.get("confidence")

        if not isinstance(confidence, (int, float)):
            raise ValueError("A numeric confidence value is required.")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        if prediction == "Normal":
            risk_level = "LOW"
            rationale = (
                "The model predicted Normal. This is a preliminary "
                "screening result, not proof that traffic is benign."
            )

        else:
            # Conservative initial rule: all attack predictions
            # require review; confidence alone does not set severity.
            risk_level = "MEDIUM"
            rationale = (
                "The model predicted Attack. Review the available "
                "traffic evidence and threat intelligence to establish "
                "actual severity."
            )

        return {
            "prediction": prediction,
            "risk_level": risk_level,
            "model_confidence": float(confidence),
            "rationale": rationale,
            "assessment_type": "preliminary_rule_based",
            "requires_review": prediction == "Attack",
        }
