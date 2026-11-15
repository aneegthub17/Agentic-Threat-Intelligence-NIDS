import json
import pandas as pd

from security_orchestrator.orchestrator import SecurityOrchestrator
from security_orchestrator.agents.memory_agent import MemoryAgent

df = pd.read_csv("datasets/processed/test_processed.csv")

memory = MemoryAgent(
    memory_path="data/demo_test_incident_memory.jsonl"
)

orchestrator = SecurityOrchestrator(
    memory_agent=memory
)

for actual in [0, 1]:
    row = df[df["target"] == actual].iloc[0]

    features = row.drop(labels=["target"]).to_numpy(dtype="float32")
    feature_names = row.drop(labels=["target"]).index.tolist()

    print("\n" + "=" * 65)
    print("ACTUAL CLASS:", "ATTACK" if actual == 1 else "NORMAL")
    print("=" * 65)

    result = orchestrator.analyze(
        features,
        feature_names=feature_names
    )

    for key, value in result.items():
        print(f"\n{key.upper()}:")
        print(json.dumps(value, indent=2, default=str)[:2500])

print("\n" + "=" * 65)
print("FULL ORCHESTRATOR TEST COMPLETED")
print("Test memory records:", len(memory.get_recent(limit=10)))
print("=" * 65)
