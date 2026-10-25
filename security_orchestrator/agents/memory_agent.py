
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MEMORY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "incident_memory.jsonl"
)


class MemoryAgent:
    """Stores and retrieves a local history of incident records."""

    def __init__(self, memory_path=None):
        self.memory_path = Path(
            memory_path or DEFAULT_MEMORY_PATH
        )

    def store(self, detection_record, risk_assessment, threat_analysis=None):
        if not isinstance(detection_record, dict):
            raise TypeError("Detection record must be a dictionary.")

        if detection_record.get("status") != "valid":
            raise ValueError("Detection record must be validated.")

        if not isinstance(risk_assessment, dict):
            raise TypeError("Risk assessment must be a dictionary.")

        risk_level = risk_assessment.get("risk_level")
        if risk_level not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("Invalid risk level.")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": detection_record["prediction"],
            "confidence": float(detection_record["confidence"]),
            "risk_level": risk_level,
            "summary": (
                threat_analysis.get("summary", "")
                if isinstance(threat_analysis, dict)
                else ""
            ),
        }

        self.memory_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.memory_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        return record

    def get_recent(self, limit=10):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer.")

        if not self.memory_path.exists():
            return []

        with self.memory_path.open("r", encoding="utf-8") as file:
            lines = file.readlines()

        records = []
        for line in lines[-limit:]:
            if line.strip():
                records.append(json.loads(line))

        return list(reversed(records))
