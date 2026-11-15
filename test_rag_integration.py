import json
import pandas as pd

from security_orchestrator.orchestrator import SecurityOrchestrator
from security_orchestrator.agents.memory_agent import MemoryAgent
from security_orchestrator.rag.retriever import build_local_retriever

df = pd.read_csv("datasets/processed/test_processed.csv")

retriever = build_local_retriever()

memory = MemoryAgent(
    memory_path="data/rag_integration_test.jsonl"
)

orchestrator = SecurityOrchestrator(
    memory_agent=memory,
    retriever=retriever
)

row = df[df["target"] == 1].iloc[0]

features = row.drop(labels=["target"]).to_numpy(dtype="float32")
feature_names = row.drop(labels=["target"]).index.tolist()

traffic_context = {
    "incident_response": "cybersecurity risk management detection recovery",
    "threat_reference": "NIST incident response",
    "investigation_context": "network intrusion security incident"
}

result = orchestrator.analyze(
    features,
    traffic_features=traffic_context,
    feature_names=feature_names
)

print("\nPREDICTION:", result["detection"]["prediction"])
print("XAI STATUS:", result["xai_status"])
print("\nTHREAT ANALYSIS:")
print(json.dumps(result["threat_analysis"], indent=2))

print("\nRISK ASSESSMENT:")
print(json.dumps(result["risk_assessment"], indent=2))

print("\nREPORT STATUS:", result["report"]["report_status"])
print("\nRAG INTEGRATION TEST COMPLETED")
