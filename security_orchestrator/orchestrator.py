
from detection_engine.inference.predict import load_model, predict

from security_orchestrator.agents.detection_agent import DetectionAgent
from security_orchestrator.agents.threat_intelligence_agent import (
    ThreatIntelligenceAgent,
)
from security_orchestrator.agents.risk_assessment_agent import (
    RiskAssessmentAgent,
)
from security_orchestrator.agents.memory_agent import MemoryAgent
from security_orchestrator.reporting.report_generator import ReportGenerator
from security_orchestrator.xai.feature_explainer import FeatureExplainer


class SecurityOrchestrator:
    """Coordinates detection, investigation, XAI, and reporting."""

    def __init__(
        self,
        model=None,
        memory_agent=None,
        explainer=None,
    ):
        # Preserve the existing trained-model loading behavior.
        self.model = model if model is not None else load_model()

        self.detection_agent = DetectionAgent()
        self.threat_agent = ThreatIntelligenceAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.memory_agent = memory_agent or MemoryAgent()

        self.report_generator = ReportGenerator()

        # Allow injection for testing or alternative explainers.
        self.explainer = (
            explainer if explainer is not None
            else FeatureExplainer(self.model)
        )

    def analyze(
        self,
        features,
        traffic_features=None,
        feature_names=None,
    ):
        # 1. Run the existing Transformer inference.
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

        # 3. Generate local XAI feature attributions.
        # Explanation failure must not discard a valid detection.
        try:
            xai_explanation = self.explainer.explain(
                features,
                feature_names=feature_names,
                top_k=10,
            )
            xai_status = "completed"
        except Exception as exc:
            xai_explanation = {
                "status": "unavailable",
                "reason": (
                    f"Explanation failed: {type(exc).__name__}"
                ),
            }
            xai_status = "unavailable"

        # 4. Produce preliminary threat analysis.
        threat_analysis = self.threat_agent.process(
            detection,
            traffic_features=traffic_features,
        )

        # 5. Assess preliminary risk.
        risk_assessment = self.risk_agent.process(
            detection,
            threat_analysis=threat_analysis,
        )

        # 6. Store the incident in memory.
        memory_record = self.memory_agent.store(
            detection,
            risk_assessment,
            threat_analysis,
        )

        # 7. Generate the incident report.
        report = self.report_generator.generate(
            detection=detection,
            threat_analysis=threat_analysis,
            risk_assessment=risk_assessment,
            memory_record=memory_record,
            xai_explanation=xai_explanation,
        )

        # 8. Return all outputs.
        return {
            "detection": detection,
            "xai_status": xai_status,
            "xai_explanation": xai_explanation,
            "threat_analysis": threat_analysis,
            "risk_assessment": risk_assessment,
            "memory_record": memory_record,
            "report": report,
        }
