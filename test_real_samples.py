import pandas as pd
from detection_engine.inference.predict import load_model, predict

df = pd.read_csv("datasets/processed/test_processed.csv")
model = load_model()

for actual in [0, 1]:
    row = df[df["target"] == actual].iloc[0]
    features = row.drop(labels=["target"]).to_numpy(dtype="float32")

    prediction, confidence, probabilities = predict(model, features)

    print("\nActual:", "Attack" if actual == 1 else "Normal")
    print("Predicted:", "Attack" if prediction.item() == 1 else "Normal")
    print("Confidence:", round(confidence.item(), 4))
    print("Probabilities:", probabilities.tolist())
