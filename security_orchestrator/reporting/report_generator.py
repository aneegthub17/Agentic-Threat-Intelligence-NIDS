
from datetime import datetime, timezone
from uuid import uuid4
import math


class ReportGenerator:
    """Generate evidence-aware preliminary NIDS incident reports."""

    @staticmethod
    def _recommendations(prediction, risk_level, patterns, family):
        recommendations = []

        if prediction == "Normal":
            return [
                "Continue routine network monitoring.",
                "Review other security alerts if independent indicators "
                "suggest suspicious activity.",
                "Do not treat a Normal prediction as proof of safety.",
            ]

        recommendations.extend([
            "Review the original network flow and relevant security logs.",
            "Correlate the detection with nearby flows, endpoints, "
            "and authentication or application events.",
            "Validate the alert before taking disruptive action.",
        ])

        pattern_names = {
            item.get("pattern")
            for item in patterns
            if isinstance(item, dict)
        }

        if "high_packet_rate" in pattern_names:
            recommendations.append(
                "Examine traffic volume across multiple flows and time "
                "windows to determine whether sustained flooding occurred."
            )

        if "high_byte_rate" in pattern_names:
            recommendations.append(
                "Review transferred data volume, destination context, "
                "and authorized workload activity."
            )

        if "asymmetric_packet_flow" in pattern_names:
            recommendations.append(
                "Investigate directional packet imbalance using "
                "connection logs and related bidirectional flows."
            )

        if "short_high_volume_flow" in pattern_names:
            recommendations.append(
                "Verify flow duration units and packet counts before "
                "interpreting the observed short-duration pattern."
            )

        if family not in ("Undetermined", "Not applicable", None):
            recommendations.append(
                "Treat the suggested attack family as an unverified "
                "hypothesis until corroborating evidence is available."
            )

        if risk_level == "HIGH":
            recommendations.extend([
                "Verify the reported critical indicators and preserve "
                "relevant forensic evidence.",
                "Escalate to the authorized incident-response team for "
                "containment decisions based on verified findings.",
            ])
        else:
            recommendations.append(
                "Escalate if independent investigation establishes "
                "compromise, disruption, or another critical indicator."
            )

        return recommendations

    @staticmethod
    def _as_dict(value):
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_list(value):
        return value if isinstance(value, list) else []

    def generate(
        self,
        detection,
        threat_analysis,
        risk_assessment,
        memory_record=None,
        xai_explanation=None,
    ):
        if not isinstance(detection, dict):
            raise TypeError("detection must be a dictionary")

        if not isinstance(threat_analysis, dict):
            raise TypeError("threat_analysis must be a dictionary")

        if not isinstance(risk_assessment, dict):
            raise TypeError("risk_assessment must be a dictionary")

        if detection.get("status") != "valid":
            raise ValueError(
                "Cannot generate a report from invalid detection"
            )

        prediction = detection.get("prediction")

        if prediction not in ("Normal", "Attack"):
            raise ValueError("Unsupported detection prediction")

        confidence = detection.get("confidence")

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(
                "Detection confidence must be a finite number "
                "between 0 and 1"
            )

        risk_level = risk_assessment.get("risk_level")

        if risk_level not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError("Unsupported risk level")

        if (
            threat_analysis.get("prediction", prediction)
            != prediction
        ):
            raise ValueError(
                "Threat analysis prediction does not match detection"
            )

        if (
            risk_assessment.get("prediction", prediction)
            != prediction
        ):
            raise ValueError(
                "Risk assessment prediction does not match detection"
            )

        if xai_explanation is None:
            xai_explanation = {
                "status": "not_provided",
                "method": None,
                "top_features": [],
                "interpretation_note": (
                    "No feature explanation was provided."
                ),
            }
        elif not isinstance(xai_explanation, dict):
            raise TypeError(
                "xai_explanation must be a dictionary"
            )

        observed_features = self._as_dict(
            threat_analysis.get("observed_features")
        )

        traffic_patterns = self._as_list(
            threat_analysis.get("traffic_patterns")
        )

        family_analysis = self._as_dict(
            threat_analysis.get("attack_family_analysis")
        )

        family = family_analysis.get(
            "family", "Undetermined"
        )

        sources = self._as_list(
            threat_analysis.get("sources")
        )

        critical_indicators = self._as_list(
            risk_assessment.get("critical_indicators")
        )

        if prediction == "Attack":
            summary = (
                "The binary Transformer classified the analyzed "
                "network flow as Attack. This is a model-generated "
                "alert requiring investigation, not independent "
                "confirmation of malicious activity."
            )
        else:
            summary = (
                "The binary Transformer classified the analyzed "
                "network flow as Normal. This result does not "
                "guarantee the traffic is benign."
            )

        if family not in ("Undetermined", "Not applicable"):
            family_note = (
                "An attack-family hypothesis was recorded, "
                "but it has not been independently verified."
            )
        else:
            family_note = (
                "No reliable attack-family classification "
                "is available from the current evidence."
            )

        recommendations = self._recommendations(
            prediction,
            risk_level,
            traffic_patterns,
            family,
        )

        report = {
            "report_id": str(uuid4()),
            "generated_at": (
                datetime.now(timezone.utc).isoformat()
            ),
            "report_status": "preliminary",
            "report_type": (
                "transformer_nids_investigation"
            ),
            "summary": summary,
            "detection": {
                "prediction": prediction,
                "predicted_class": detection.get(
                    "predicted_class"
                ),
                "confidence": float(confidence),
                "probabilities": self._as_dict(
                    detection.get("probabilities")
                ),
                "model_scope": (
                    "binary_normal_vs_attack"
                ),
            },
            "xai_explanation": xai_explanation,
            "threat_analysis": {
                "summary": threat_analysis.get(
                    "summary", ""
                ),
                "observed_features": observed_features,
                "traffic_patterns": traffic_patterns,
                "attack_family_analysis": family_analysis,
                "family_interpretation": family_note,
                "retrieval_performed": bool(
                    threat_analysis.get(
                        "retrieval_performed", False
                    )
                ),
                "retrieval_query": threat_analysis.get(
                    "retrieval_query", ""
                ),
                "sources": sources,
                "retrieval_interpretation": (
                    "Retrieved references provide general "
                    "investigation context. Their presence "
                    "does not independently validate the "
                    "model prediction or attack-family hypothesis."
                ),
            },
            "risk_assessment": {
                "risk_level": risk_level,
                "rationale": risk_assessment.get(
                    "rationale", ""
                ),
                "assessment_type": risk_assessment.get(
                    "assessment_type",
                    "preliminary_rule_based",
                ),
                "requires_review": risk_assessment.get(
                    "requires_review", False
                ),
                "severity_basis": risk_assessment.get(
                    "severity_basis", ""
                ),
                "critical_indicators": critical_indicators,
                "critical_indicator_count": len(
                    critical_indicators
                ),
            },
            "recommendations": recommendations,
            "memory_record": memory_record,
            "limitations": [
                "This is a research prototype, not a "
                "production intrusion-detection service.",
                "The Transformer performs binary classification "
                "and does not identify attack families.",
                "Model confidence is not a calibrated measure "
                "of real-world threat severity.",
                "Feature attributions describe model behavior; "
                "they do not establish causation.",
                "Numerical traffic-pattern thresholds are "
                "illustrative and not independently validated "
                "as attack-family decision boundaries.",
                "Attack-family hypotheses based on analyst "
                "context remain unverified.",
                "Local keyword-based retrieval supplies "
                "reference material, not independent "
                "evidence of compromise.",
                "Risk assessment is preliminary and rule-based. "
                "Any supplied critical indicators require "
                "independent verification.",
                "Dataset replay is not live network capture.",
            ],
        }

        return report
