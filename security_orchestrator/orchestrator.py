
from detection_engine.inference.predict import load_model, predict

from security_orchestrator.agents.detection_agent import DetectionAgent
from security_orchestrator.agents.threat_intelligence_agent import (
    ThreatIntelligenceAgent,
)
from security_orchestrator.agents.risk_assessment_agent import (
    RiskAssessmentAgent,
)
from security_orchestrator.agents.memory_agent import MemoryAgent


class SecurityOrchestrator:
    """Coordinates the NIDS detection and investigation agents."""

    def __init__(self, model=None, memory_agent=None):
        # Load the trained Transformer only when a model isn't injected.
        self.model = model if model is not None else load_model()

        self.detection_agent = DetectionAgent()
        self.threat_agent = ThreatIntelligenceAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.memory_agent = memory_agent or MemoryAgent()

    def analyze(self, features, traffic_features=None):
        # 1. Run Transformer inference.
        prediction, confidence, probabilities = predict(
            self.model,
            features,
        )

        # 2. Validate and standardize the detection result.
        detection = self.detection_agent.process(
            prediction,
            confidence,
            probabilities,
        )

        # 3. Produce preliminary threat analysis.
        threat_analysis = self.threat_agent.process(
            detection,
            traffic_features=traffic_features,
        )

        # 4. Assess preliminary risk.
        risk_assessment = self.risk_agent.process(
            detection,
            threat_analysis=threat_analysis,
        )

        # 5. Store the incident.
        memory_record = self.memory_agent.store(
            detection,
            risk_assessment,
            threat_analysis,
        )

        # 6. Return a structured result for reporting.
        return {
            "detection": detection,
            "threat_analysis": threat_analysis,
            "risk_assessment": risk_assessment,
            "memory_record": memory_record,
        }
