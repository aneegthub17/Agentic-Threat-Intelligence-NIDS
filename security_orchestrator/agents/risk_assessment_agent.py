
import math
import re


class RiskAssessmentAgent:
    """
    Preliminary evidence-based network security risk assessment.

    LOW:
        Transformer predicts Normal.

    MEDIUM:
        Transformer predicts Attack, without independently
        corroborated high-severity evidence.

    HIGH:
        Transformer predicts Attack and explicit,
        independently observed critical evidence exists.

    This is a rule-based triage mechanism, not a calibrated
    estimate of real-world severity.
    """

    HIGH_SEVERITY_INDICATORS = {
        "confirmed_compromise": (
            "Confirmed system compromise"
        ),
        "confirmed_data_exfiltration": (
            "Confirmed data exfiltration"
        ),
        "confirmed_ransomware": (
            "Confirmed ransomware activity"
        ),
        "confirmed_service_disruption": (
            "Confirmed service disruption"
        ),
        "confirmed_privilege_escalation": (
            "Confirmed privilege escalation"
        ),
        "confirmed_lateral_movement": (
            "Confirmed lateral movement"
        ),
    }

    def __init__(self):
        pass

    @staticmethod
    def _validate_confidence(value):
        """
        Validate confidence from the detection agent.
        """

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                "A numeric confidence value is required."
            )

        confidence = float(value)

        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(
                "Confidence must be finite and between 0 and 1."
            )

        return confidence

    @staticmethod
    def _is_explicitly_confirmed(value):
        """
        Accept only explicit Boolean confirmation.

        Strings such as 'yes', 'true', or 'confirmed'
        are deliberately not treated as verified evidence.
        """

        return value is True

    def _extract_indicators(self, threat_analysis):
        """
        Extract independently supplied critical indicators.

        RAG reference text and model confidence are not
        considered confirmation of an incident.
        """

        if not threat_analysis:
            return []

        observed = threat_analysis.get(
            "observed_features",
            {}
        )

        if not isinstance(observed, dict):
            return []

        indicators = []

        for key, description in (
            self.HIGH_SEVERITY_INDICATORS.items()
        ):
            if self._is_explicitly_confirmed(
                observed.get(key)
            ):
                indicators.append(
                    {
                        "indicator": key,
                        "description": description,
                        "source": "explicit_observed_feature",
                    }
                )

        return indicators

    def process(
        self,
        detection_record,
        threat_analysis=None,
    ):
        """
        Evaluate the preliminary risk level.

        Preserves the existing process() interface
        used by SecurityOrchestrator.
        """

        if not isinstance(detection_record, dict):
            raise TypeError(
                "Detection record must be a dictionary."
            )

        if detection_record.get("status") != "valid":
            raise ValueError(
                "Detection record has not passed validation."
            )

        prediction = detection_record.get(
            "prediction"
        )

        if prediction not in {
            "Normal",
            "Attack",
        }:
            raise ValueError(
                "Unsupported prediction."
            )

        if threat_analysis is not None:

            if not isinstance(threat_analysis, dict):
                raise TypeError(
                    "Threat analysis must be a dictionary."
                )

            if (
                threat_analysis.get("prediction")
                != prediction
            ):
                raise ValueError(
                    "Threat analysis and detection "
                    "prediction do not match."
                )

        confidence = self._validate_confidence(
            detection_record.get("confidence")
        )

        critical_indicators = (
            self._extract_indicators(
                threat_analysis
            )
        )

        if prediction == "Normal":

            risk_level = "LOW"

            rationale = (
                "The Transformer predicted Normal. "
                "No attack was identified by this model. "
                "This is a preliminary screening result "
                "and does not establish that the traffic "
                "is benign."
            )

            requires_review = False

        elif critical_indicators:

            risk_level = "HIGH"

            descriptions = [
                item["description"]
                for item in critical_indicators
            ]

            rationale = (
                "The Transformer predicted Attack, "
                "and additional critical incident "
                "indicators were explicitly supplied: "
                + "; ".join(descriptions)
                + ". These indicators require independent "
                "verification and urgent analyst review."
            )

            requires_review = True

        else:

            risk_level = "MEDIUM"

            rationale = (
                "The Transformer predicted Attack. "
                "No independently confirmed critical "
                "incident indicators were supplied. "
                "The sample requires investigation "
                "to determine its actual severity."
            )

            requires_review = True

        return {
            "prediction": prediction,
            "risk_level": risk_level,
            "model_confidence": confidence,
            "rationale": rationale,
            "assessment_type": (
                "preliminary_rule_based"
            ),
            "requires_review": requires_review,
            "critical_indicators": (
                critical_indicators
            ),
            "critical_indicator_count": (
                len(critical_indicators)
            ),
            "severity_basis": (
                "explicit_critical_indicators"
                if risk_level == "HIGH"
                else "binary_detection_screening"
            ),
        }
