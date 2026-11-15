
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ThreatIntelligenceAgent:
    """
    Investigate validated binary Transformer detections.

    A separate Random Forest provides selective attack-family
    hypotheses. The validation policy determines which family
    predictions are eligible for display.

    Ground-truth labels must never enter model inference.
    """

    FAMILY_KEYWORDS = {
        "DoS": (
            "denial of service", "ddos", "dos", "flood"
        ),
        "Reconnaissance": (
            "reconnaissance", "port scan",
            "scanning", "enumeration"
        ),
        "Brute Force": (
            "brute force", "bruteforce",
            "password guessing", "credential stuffing"
        ),
        "Exploits": (
            "exploit", "vulnerability exploitation",
            "remote code execution"
        ),
        "Fuzzers": (
            "fuzzing", "fuzzer", "malformed input"
        ),
        "Backdoor": (
            "backdoor", "unauthorized remote access"
        ),
        "Shellcode": (
            "shellcode", "payload execution"
        ),
        "Worms": (
            "worm", "self propagating malware"
        ),
    }

    EXCLUDED_FIELDS = {
        "target",
        "label",
        "attack_category",
        "attack_cat",
        "ground_truth_attack_category",
        "actual_label",
        "dataset_source",
        "ground_truth",
        "true_label",
    }

    HIGH_PACKET_RATE = 10000.0
    HIGH_PACKET_COUNT = 1000.0
    HIGH_BYTE_RATE = 10000000.0
    ASYMMETRY_RATIO = 10.0
    MIN_ASYMMETRY_PACKETS = 100.0

    MODEL_PATH = (
        Path(__file__).resolve().parents[2]
        / "detection_engine"
        / "models"
        / "attack_family_rf.joblib"
    )

    _cached_family_artifact = None

    def __init__(self, retriever=None):
        self.retriever = retriever

    @classmethod
    def _clean_features(cls, traffic_features):
        if traffic_features is None:
            return {}

        if not isinstance(traffic_features, dict):
            raise TypeError(
                "Traffic features must be a dictionary."
            )

        cleaned = {}

        for key, value in traffic_features.items():
            key = str(key)

            if key.lower() in cls.EXCLUDED_FIELDS:
                continue

            if value is None:
                cleaned[key] = None
            elif isinstance(value, bool):
                cleaned[key] = value
            elif isinstance(value, str):
                cleaned[key] = value
            elif isinstance(value, (int, float, np.number)):
                number = float(value)
                cleaned[key] = (
                    number if math.isfinite(number) else None
                )

        return cleaned

    @staticmethod
    def _numeric(features, name):
        value = features.get(name)

        if value is None or isinstance(value, bool):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number) or number < 0:
            return None

        return number

    @staticmethod
    def _keyword_match(context, keyword):
        pattern = (
            r"(?<!\w)"
            + re.escape(keyword)
            + r"(?!\w)"
        )
        return bool(re.search(pattern, context))

    def _analyst_family_matches(self, features):
        context = features.get(
            "investigation_context", ""
        )

        if not isinstance(context, str):
            return []

        context = context.lower()
        matches = []

        for family, keywords in self.FAMILY_KEYWORDS.items():
            found = [
                keyword
                for keyword in keywords
                if self._keyword_match(context, keyword)
            ]

            if found:
                matches.append({
                    "family": family,
                    "keywords": found,
                })

        return matches

    def _numerical_pattern_analysis(self, features):
        """
        Describe observed traffic characteristics without
        treating them as confirmed attack-family evidence.
        """
        observations = []

        src_packets = self._numeric(
            features, "src_packets"
        )
        dst_packets = self._numeric(
            features, "dst_packets"
        )
        src_packet_rate = self._numeric(
            features, "src_packet_rate"
        )
        dst_packet_rate = self._numeric(
            features, "dst_packet_rate"
        )
        src_bytes_rate = self._numeric(
            features, "src_bytes_rate"
        )
        dst_bytes_rate = self._numeric(
            features, "dst_bytes_rate"
        )
        duration = self._numeric(
            features, "duration"
        )

        for direction, rate, packets in (
            ("source", src_packet_rate, src_packets),
            ("destination", dst_packet_rate, dst_packets),
        ):
            if (
                rate is not None
                and packets is not None
                and rate >= self.HIGH_PACKET_RATE
                and packets >= self.HIGH_PACKET_COUNT
            ):
                observations.append({
                    "pattern": "high_packet_rate",
                    "direction": direction,
                    "observed_packet_rate": rate,
                    "observed_packet_count": packets,
                    "screening_threshold": self.HIGH_PACKET_RATE,
                    "interpretation": (
                        "Elevated packet rate and count "
                        "within a single flow. This is not "
                        "sufficient to establish DoS."
                    ),
                })

        for direction, rate in (
            ("source", src_bytes_rate),
            ("destination", dst_bytes_rate),
        ):
            if (
                rate is not None
                and rate >= self.HIGH_BYTE_RATE
            ):
                observations.append({
                    "pattern": "high_byte_rate",
                    "direction": direction,
                    "observed_bytes_per_second": rate,
                    "screening_threshold": self.HIGH_BYTE_RATE,
                    "interpretation": (
                        "Elevated byte rate. This can occur "
                        "in legitimate traffic and does not "
                        "prove data exfiltration."
                    ),
                })

        if (
            src_packets is not None
            and dst_packets is not None
        ):
            larger = max(src_packets, dst_packets)
            smaller = min(src_packets, dst_packets)
            ratio = larger / max(smaller, 1.0)

            if (
                larger >= self.MIN_ASYMMETRY_PACKETS
                and ratio >= self.ASYMMETRY_RATIO
            ):
                observations.append({
                    "pattern": "asymmetric_packet_flow",
                    "source_packets": src_packets,
                    "destination_packets": dst_packets,
                    "packet_ratio": ratio,
                    "interpretation": (
                        "Strongly asymmetric packet counts. "
                        "The cause cannot be determined "
                        "from this flow alone."
                    ),
                })

        if (
            duration is not None
            and 0 < duration <= 1.0
            and src_packets is not None
            and src_packets >= self.HIGH_PACKET_COUNT
        ):
            observations.append({
                "pattern": "short_high_volume_flow",
                "duration": duration,
                "source_packets": src_packets,
                "interpretation": (
                    "High packet count over a short "
                    "reported duration. Duration units "
                    "require verification."
                ),
            })

        return observations

    @classmethod
    def _load_family_artifact(cls):
        """
        Load the family model once, not once per replay sample.

        Joblib files must come from a trusted local source.
        """
        if cls._cached_family_artifact is None:
            if not cls.MODEL_PATH.is_file():
                return None

            cls._cached_family_artifact = joblib.load(
                cls.MODEL_PATH
            )

        return cls._cached_family_artifact

    def _predict_family(self, features):
        """
        Predict a family using network-flow inputs only.

        Return the model's raw hypothesis and the
        validation-based eligibility decision.
        """
        artifact = self._load_family_artifact()

        if artifact is None:
            return {
                "status": "model_unavailable",
                "reason": "Family classifier not found.",
            }

        feature_columns = artifact["feature_columns"]

        missing = [
            name
            for name in feature_columns
            if name not in features
        ]

        if missing:
            return {
                "status": "missing_features",
                "reason": (
                    "Required network-flow features "
                    "are missing."
                ),
                "missing_features": missing,
            }

        row = {}

        for name in feature_columns:
            value = features[name]

            if name == "protocol":
                row[name] = (
                    str(value)
                    if value is not None
                    else "unknown"
                )
            else:
                row[name] = self._numeric(
                    features, name
                )

        frame = pd.DataFrame(
            [row],
            columns=feature_columns,
        )

        pipeline = artifact["pipeline"]
        probabilities = pipeline.predict_proba(frame)[0]

        classes = pipeline.classes_
        best_index = int(np.argmax(probabilities))

        predicted_family = str(classes[best_index])
        confidence = float(probabilities[best_index])

        policy = artifact.get(
            "validation_policy", {}
        )

        minimum_confidence = float(
            policy.get("minimum_confidence", 1.0)
        )

        family_details = policy.get(
            "families", {}
        ).get(predicted_family, {})

        eligible = bool(
            family_details.get("eligible", False)
        )

        accepted = (
            predicted_family != "Normal"
            and eligible
            and confidence >= minimum_confidence
        )

        return {
            "status": "completed",
            "predicted_family": predicted_family,
            "confidence": confidence,
            "minimum_confidence": minimum_confidence,
            "validation_eligible": eligible,
            "accepted": accepted,
            "validation_precision": family_details.get(
                "validation_precision"
            ),
            "validation_predictions": family_details.get(
                "validation_predictions", 0
            ),
        }

    def _investigate_family(
        self, prediction, features, patterns=None
    ):
        if prediction != "Attack":
            return {
                "family": "No attack indicated",
                "method": "binary_transformer",
                "evidence": [],
                "certainty": "not_assessed",
                "alternative_explanations": [
                    "A Normal prediction does not guarantee "
                    "the absence of malicious activity."
                ],
            }

        if patterns is None:
            patterns = self._numerical_pattern_analysis(
                features
            )

        analyst_matches = self._analyst_family_matches(
            features
        )

        try:
            model_result = self._predict_family(
                features
            )
        except Exception as exc:
            model_result = {
                "status": "inference_error",
                "reason": (
                    f"Family inference failed: "
                    f"{type(exc).__name__}"
                ),
            }

        if (
            model_result.get("status") == "completed"
            and model_result.get("accepted")
        ):
            family = model_result["predicted_family"]

            return {
                "family": f"Possible {family}",
                "method": "validated_random_forest",
                "certainty": "model_hypothesis",
                "confidence": model_result["confidence"],
                "predicted_family": family,
                "validation_precision": model_result[
                    "validation_precision"
                ],
                "evidence": [
                    {
                        "type": "multiclass_model_prediction",
                        "model": "RandomForestClassifier",
                        "predicted_family": family,
                        "confidence": model_result[
                            "confidence"
                        ],
                        "validation_precision": model_result[
                            "validation_precision"
                        ],
                        "validation_predictions": model_result[
                            "validation_predictions"
                        ],
                    }
                ] + patterns,
                "analyst_context_matches": analyst_matches,
                "alternative_explanations": [
                    "This is a learned family hypothesis, "
                    "not a confirmed attack classification.",
                    "The model can make incorrect predictions "
                    "even at high confidence.",
                    "Validation performance may not generalize "
                    "to new network environments.",
                ],
            }

        return {
            "family": "Attack type under investigation",
            "method": "selective_random_forest",
            "certainty": "insufficient_reliable_evidence",
            "confidence": model_result.get(
                "confidence"
            ),
            "predicted_family": model_result.get(
                "predicted_family"
            ),
            "model_status": model_result.get(
                "status"
            ),
            "evidence": patterns,
            "analyst_context_matches": analyst_matches,
            "alternative_explanations": [
                "The family model did not meet its "
                "validation-based acceptance requirements.",
                "An Attack prediction does not establish "
                "a specific attack family.",
            ],
        }

    def process(
        self,
        detection_record,
        traffic_features=None,
    ):
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

        if prediction not in {"Normal", "Attack"}:
            raise ValueError(
                "Unsupported prediction."
            )

        features = self._clean_features(
            traffic_features
        )

        patterns = self._numerical_pattern_analysis(
            features
        )

        family_analysis = self._investigate_family(
            prediction,
            features,
            patterns,
        )

        if prediction == "Attack":
            summary = (
                "The Transformer classified this network "
                "flow as Attack. Further investigation is "
                "required. A separate Random Forest provides "
                "a selective attack-family hypothesis when "
                "validation requirements are met. This does "
                "not establish compromise."
            )
        else:
            summary = (
                "The Transformer classified this network "
                "flow as Normal. This does not guarantee "
                "that the traffic is benign."
            )

        evidence = []
        retrieval_query = ""

        if self.retriever is not None:
            if prediction == "Attack":
                query_parts = [
                    "incident response",
                    "investigation",
                    "containment",
                    "adversary behavior",
                ]
            else:
                query_parts = [
                    "network monitoring",
                    "incident response",
                ]

            pattern_names = {
                item["pattern"]
                for item in patterns
            }

            if "high_packet_rate" in pattern_names:
                query_parts.extend([
                    "traffic volume",
                    "network availability",
                ])

            if "high_byte_rate" in pattern_names:
                query_parts.extend([
                    "network traffic",
                    "data transfer investigation",
                ])

            if "asymmetric_packet_flow" in pattern_names:
                query_parts.append(
                    "network flow investigation"
                )

            context = features.get(
                "investigation_context"
            )

            if (
                isinstance(context, str)
                and context.strip()
            ):
                query_parts.append(
                    context.strip()
                )

            if (
                family_analysis.get("method")
                == "validated_random_forest"
            ):
                query_parts.append(
                    family_analysis["predicted_family"]
                )

            retrieval_query = " ".join(
                query_parts
            )

            retrieved = self.retriever.retrieve(
                retrieval_query,
                top_k=3,
            )

            from security_orchestrator.rag.evidence_formatter import (
                EvidenceFormatter,
            )

            evidence = EvidenceFormatter.format_results(
                retrieved
            )

        return {
            "prediction": prediction,
            "summary": summary,
            "observed_features": features,
            "traffic_patterns": patterns,
            "attack_family_analysis": family_analysis,
            "retrieval_performed": (
                self.retriever is not None
            ),
            "retrieval_query": retrieval_query,
            "sources": evidence,
        }
