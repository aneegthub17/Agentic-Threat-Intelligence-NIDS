
from datetime import datetime, timezone
from uuid import uuid4


class ReportGenerator:
    """Generate structured incident reports from agent outputs."""

    def generate(
        self,
        detection,
        threat_analysis,
        risk_assessment,
        memory_record=None,
    ):
        if not isinstance(detection, dict):
            raise TypeError("detection must be a dictionary")

        if not isinstance(threat_analysis, dict):
            raise TypeError("threat_analysis must be a dictionary")

        if not isinstance(risk_assessment, dict):
            raise TypeError("risk_assessment must be a dictionary")

        if detection.get("status") != "valid":
            raise ValueError("Cannot generate a report from invalid detection")

        prediction = detection.get("prediction")
        if prediction not in ("Normal", "Attack"):
            raise ValueError("Unsupported detection prediction")

        confidence = detection.get("confidence")
        if not isinstance(confidence, (int, float)):
            raise ValueError("Detection confidence must be numeric")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")

        risk_level = risk_assessment.get("risk_level")
        if risk_level not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError("Unsupported risk level")

        if prediction == "Attack":
            summary = (
                "The Transformer detected potential malicious traffic. "
                "Further investigation is recommended."
            )
            recommendations = [
                "Review the relevant network traffic and security logs.",
                "Correlate the alert with trusted threat-intelligence sources.",
                "Verify the activity before deciding on a response.",
            ]
        else:
            summary = (
                "The Transformer classified the analyzed traffic as normal. "
                "This result does not guarantee that the traffic is benign."
            )
            recommendations = [
                "Continue routine monitoring of network activity.",
                "Investigate separately if other security indicators raise concern.",
            ]

        return {
            "report_id": str(uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_status": "preliminary",
            "detection": {
                "prediction": prediction,
                "predicted_class": detection.get("predicted_class"),
                "confidence": confidence,
                "probabilities": detection.get("probabilities", {}),
            },
            "threat_analysis": {
                "summary": threat_analysis.get("summary", ""),
                "observed_features": threat_analysis.get(
                    "observed_features", {}
                ),
                "retrieval_performed": threat_analysis.get(
                    "retrieval_performed", False
                ),
                "sources": threat_analysis.get("sources", []),
            },
            "risk_assessment": {
                "risk_level": risk_level,
                "rationale": risk_assessment.get("rationale", ""),
                "assessment_type": risk_assessment.get(
                    "assessment_type", "preliminary_rule_based"
                ),
                "requires_review": risk_assessment.get(
                    "requires_review", False
                ),
            },
            "recommendations": recommendations,
            "memory_record": memory_record,
            "limitations": [
                "The detection confidence is not a calibrated probability "
                "of real-world threat severity.",
                "Threat-intelligence retrieval is not performed unless "
                "explicitly implemented by the threat-analysis component.",
                "Risk assessment is preliminary and rule-based.",
                "The report does not independently confirm an attack family "
                "or identify a specific vulnerability.",
            ],
        }
