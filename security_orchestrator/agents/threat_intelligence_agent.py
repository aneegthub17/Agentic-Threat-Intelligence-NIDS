
import math
import re


class ThreatIntelligenceAgent:
    """
    Investigate validated binary Transformer detections.

    Numerical observations describe traffic characteristics.
    They are not validated attack-family classifications.

    Ground-truth labels must not enter agent reasoning.
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

    # Illustrative screening thresholds, not validated
    # intrusion or attack-family decision boundaries.
    HIGH_PACKET_RATE = 10000.0
    HIGH_PACKET_COUNT = 1000.0
    HIGH_BYTE_RATE = 10000000.0

    ASYMMETRY_RATIO = 10.0
    MIN_ASYMMETRY_PACKETS = 100.0

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

            elif isinstance(value, (int, float)):
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
        Describe measurable flow characteristics.

        Do not assign an attack family from these patterns.
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
                    "screening_threshold": (
                        self.HIGH_PACKET_RATE
                    ),
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
                    "screening_threshold": (
                        self.HIGH_BYTE_RATE
                    ),
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

    def _investigate_family(
        self, prediction, features, patterns=None
    ):
        """
        Only analyst-provided context can suggest a family.

        No numerical rule assigns an attack family.
        """

        if prediction != "Attack":
            return {
                "family": "Not applicable",
                "method": "not_applicable",
                "evidence": [],
                "certainty": "not_assessed",
                "alternative_explanations": [],
            }

        if patterns is None:
            patterns = self._numerical_pattern_analysis(
                features
            )

        matches = self._analyst_family_matches(
            features
        )

        if len(matches) == 1:
            match = matches[0]

            return {
                "family": match["family"],
                "method": "analyst_context_keyword_match",
                "evidence": [
                    {
                        "type": "analyst_context",
                        "matched_keywords": (
                            match["keywords"]
                        ),
                    }
                ] + patterns,
                "certainty": "unverified_hypothesis",
                "alternative_explanations": [
                    "The analyst description is not "
                    "independent verification.",
                    "Traffic patterns can also occur "
                    "during benign activity.",
                ],
            }

        if len(matches) > 1:
            return {
                "family": "Undetermined",
                "method": "ambiguous_analyst_context",
                "evidence": patterns,
                "certainty": "unknown",
                "alternative_explanations": [
                    "Multiple attack families were "
                    "mentioned in the analyst context."
                ],
            }

        return {
            "family": "Undetermined",
            "method": "insufficient_flow_evidence",
            "evidence": patterns,
            "certainty": "unknown",
            "alternative_explanations": [
                "Numerical flow observations are "
                "not sufficient for reliable "
                "attack-family identification."
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
                "flow as Attack. Further investigation is required. "
                "Numerical observations describe traffic characteristics but do "
                "not independently identify an attack "
                "family or establish compromise."
            )
        else:
            summary = (
                "The Transformer classified this network "
                "flow as Normal. This does not guarantee that the "
                "traffic is benign."
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

            # Search terms describe observations,
            # not verified adversary techniques.
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
                query_parts.extend([
                    "network flow investigation",
                ])

            context = features.get(
                "investigation_context"
            )

            if (
                isinstance(context, str)
                and context.strip()
            ):
                query_parts.append(context.strip())

            family = family_analysis["family"]

            if family not in {
                "Undetermined",
                "Not applicable",
            }:
                query_parts.append(family)

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
