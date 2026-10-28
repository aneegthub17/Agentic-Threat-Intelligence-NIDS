# Agentic Threat Intelligence Framework for Network Intrusion Detection Using Transformer-Based Traffic Analysis, Explainable AI, and Retrieval-Augmented Generation

## Problem Statement

Traditional Network Intrusion Detection System (NIDS) approaches face practical challenges in real-world network environments, including delayed packet arrival, packet loss, high volumes of network traffic, constantly evolving attack patterns, and large numbers of false-positive alerts. These challenges can increase the workload of security analysts, who must continuously monitor alerts and investigate potential threats. While existing approaches primarily focus on detecting and classifying attacks, they often provide limited contextual and explainable information to help analysts understand the nature and severity of a detected threat. The lack of integrated threat intelligence and actionable guidance further makes incident analysis and response more time-consuming and difficult.

## Objectives

### 1. Unified Feature Representation

To develop a unified feature mapping framework that transforms features from the CIC-IDS2018 and UNSW-NB15 datasets into a common feature space, enabling their integration for consistent network intrusion detection.

### 2. Transformer-Based Detection and Agentic Analysis

To develop a Transformer-based Network Intrusion Detection System (NIDS) and integrate an Agentic Artificial Intelligence (AI) framework for attack detection, validation, threat analysis, and risk assessment.

### 3. Comparative Threat Explanation and Intelligence

To integrate Explainable Artificial Intelligence (XAI) and Retrieval-Augmented Generation (RAG) for comparative evaluation of threat explanations, contextual analysis, and evidence-based incident reporting with actionable mitigation recommendations.

## Project Overview

This project is a research-oriented Network Intrusion Detection System (NIDS) that combines a Transformer-based binary traffic classifier with a Security Orchestrator, investigation agents, Explainable AI (XAI), local threat-intelligence retrieval, risk assessment, incident memory, and structured reporting.

The project aims to support security analysts by connecting intrusion detection with interpretable feature attribution, relevant security references, risk assessment, and preliminary incident reporting.

> **Current RAG scope:** The implemented RAG component uses a local keyword-based retriever to return traceable references from a local knowledge base. It does not currently use a generative LLM to generate new threat intelligence. This project is a research prototype, not a production network sensor or automated response system.

## Current Implementation Status

| Component | Current status |
|---|---|
| Unified feature mapping for CIC-IDS2018 and UNSW-NB15 | Implemented |
| Dataset merging and preprocessing | Implemented |
| Transformer-based binary intrusion detection | Implemented |
| Security Orchestrator and investigation agents | Implemented |
| Risk assessment and incident memory | Implemented |
| Explainable AI using Gradient × Input | Integrated and tested |
| Local keyword-based threat-intelligence retrieval | Integrated and tested |
| Structured preliminary report generation | Integrated and tested |
| Automated test suite | Last recorded run: 59 passed, 0 failed |
| Real-model integrated execution with XAI, local retrieval, and report generation | Successfully demonstrated on one sample |
| Broader experimental evaluation and final research documentation | In progress |

### Verified demonstration result

In the recorded end-to-end demonstration, the first row from `datasets/processed/test_processed.csv` had the true label `0` (Normal). The model predicted **Normal** with approximately **89.80% confidence**. XAI completed, local retrieval returned a NIST SP 800-61 Rev. 3 incident-response reference, and a preliminary report was generated.

This is a successful single-sample integration demonstration, not a measure of overall model performance.

## System Architecture

```text
Processed network traffic features (147 inputs)
                    |
                    v
        Transformer-based NIDS
             Normal / Attack
                    |
                    v
          Security Orchestrator
             /            \
            v              v
     XAI Explainer     Threat Intelligence Agent
   (Gradient × Input)           |
                                v
                   Local Keyword-Based Retriever
                                |
                                v
                    Traceable Source References
             \              /
              v            v
             Risk Assessment
                    |
             Incident Memory
                    |
                    v
          Structured Preliminary Report
```

The diagram presents the high-level workflow. The actual availability and output of each component should be interpreted using the status fields in the execution result.

## Main Contribution

The main contribution of this project is the integration of **Transformer-based intrusion detection, explainability, threat-intelligence retrieval, risk assessment, incident memory, and structured reporting into one orchestrated workflow**.

- **Transformer:** Classifies a processed traffic sample as Normal or Attack.
- **Security Orchestrator:** Coordinates detection and downstream analysis.
- **XAI:** Uses Gradient × Input feature attribution to indicate which input features influenced the selected-class score.
- **Local RAG:** Retrieves relevant references from a local threat-intelligence knowledge base and preserves source details.
- **Risk assessment and memory:** Support triage and incident tracking.
- **Reporting:** Consolidates available analysis into a preliminary report with references and recommendations.

The retrieval score is a keyword-overlap ranking score, not a probability that a source is correct. Gradient × Input attribution is not a causal explanation, and model confidence is not a calibrated guarantee of correctness.

## Repository Structure

```text
Agentic-Threat-Intelligence-NIDS/
├── detection_engine/
│   ├── models/
│   │   ├── tabular_transformer.py
│   │   └── best_transformer.pt
│   ├── inference/
│   │   └── predict.py
│   ├── training/
│   │   ├── dataset.py
│   │   └── train.py
│   └── evaluation/
│       └── evaluate.py
├── security_orchestrator/
│   ├── agents/
│   ├── rag/
│   │   ├── knowledge_base/
│   │   │   └── documents.json
│   │   ├── document_store.py
│   │   ├── retriever.py
│   │   ├── evidence_formatter.py
│   │   └── knowledge_base_loader.py
│   ├── reporting/
│   │   └── report_generator.py
│   ├── xai/
│   │   └── feature_explainer.py
│   ├── tests/
│   └── orchestrator.py
├── utfh/
│   ├── mappers/
│   │   ├── feature_mapper.py
│   │   └── merge_datasets.py
│   └── preprocessing/
│       └── preprocess.py
└── datasets/
    ├── raw/
    ├── merged/
    └── processed/
```

Large dataset files and trained model checkpoints may not be included in a fresh clone. Check the repository contents and ensure the required artifacts are available before running the full workflow.

## Requirements

- Git
- A Python version compatible with the project's PyTorch and scientific Python dependencies
- Project datasets and trained model checkpoint for real-model inference
- Python packages used by the code, including PyTorch, pandas, NumPy, scikit-learn, and joblib
- `pytest` for automated testing

If the repository contains a `requirements.txt`, use it to install the project's dependencies and pinned versions.

## Setup on Windows (PowerShell)

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/aneegthub17/Agentic-Threat-Intelligence-NIDS.git
cd Agentic-Threat-Intelligence-NIDS
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `requirements.txt` exists, install from it:

```powershell
python -m pip install -r requirements.txt
```

If there is no dependency manifest, install the packages required by the current code:

```powershell
python -m pip install torch pandas numpy scikit-learn joblib pytest
```

If PowerShell blocks activation, follow your environment's execution policy or use `.venv\Scripts\python.exe` directly.

## Dataset and Model Files

The mapping script expects the following raw data files:

```text
datasets/raw/CICIDS2018/cic.csv
datasets/raw/UNSW-NB15/UNSW_NB15_training-set.csv
datasets/raw/UNSW-NB15/UNSW_NB15_testing-set.csv
```

Generated data and model artifacts used by the current pipeline include:

```text
datasets/merged/unified_train.csv
datasets/merged/unified_test.csv
datasets/processed/train_processed.csv
datasets/processed/test_processed.csv
datasets/processed/standard_scaler.joblib
detection_engine/models/best_transformer.pt
```

If files are missing from a fresh clone, obtain them from the project owner or approved dataset sources. Respect dataset terms and avoid committing large raw datasets or private data without checking repository policy.

### Rebuild the Merged and Processed Datasets

After confirming the raw CSV files are available, the intended command sequence is:

```powershell
python utfh/mappers/merge_datasets.py
python utfh/preprocessing/preprocess.py
```

These scripts are intended to create the unified and processed datasets.

**Before regenerating existing data, verify that the mapper import path in `utfh/mappers/merge_datasets.py` points to the actual `utfh/mappers` directory.** Do not overwrite working data artifacts until the script path and expected outputs have been checked.

## Running the Automated Tests

From the repository root, with the virtual environment activated:

```powershell
python -m pytest -q
```

The last recorded test run returned:

```text
59 passed, 2 warnings
```

This is the last recorded development result; run the suite on the current checkout to verify it. The previously observed warnings were non-fatal PyTorch Transformer encoder warnings.

Useful focused commands:

```powershell
python -m pytest security_orchestrator/tests/test_real_inference.py -q -s
python -m pytest security_orchestrator/tests/test_report_generator.py -q
python -m pytest security_orchestrator/rag -q
```

## Running a Real End-to-End Demonstration

Run this from the repository root after the model checkpoint, processed dataset, and dependencies are available. It loads one processed test sample, runs the existing model-loading and orchestration path, enables local retrieval, and prints the detection, XAI, retrieved references, and report.

```powershell
@'
import pandas as pd
from security_orchestrator.orchestrator import SecurityOrchestrator
from security_orchestrator.rag.retriever import build_local_retriever

df = pd.read_csv("datasets/processed/test_processed.csv", nrows=1)
X = df.drop(columns=["target"])
features = X.iloc[0].to_numpy(dtype="float32")

orchestrator = SecurityOrchestrator(
    retriever=build_local_retriever()
)

result = orchestrator.analyze(
    features,
    feature_names=list(X.columns),
    traffic_features={
        "investigation_context":
            "network intrusion incident response cybersecurity risk management"
    },
)

report = result.get("report", {})
threat = report.get("threat_analysis", {})

print("\n=== DETECTION ===")
print(result.get("detection"))

print("\n=== XAI ===")
print("Status:", result.get("xai_status"))
print("Explanation:", result.get("xai_explanation"))

print("\n=== THREAT INTELLIGENCE ===")
print("Retrieval performed:", threat.get("retrieval_performed"))
print("Sources:", threat.get("sources"))

print("\n=== REPORT ===")
print("Report ID:", report.get("report_id"))
print("Status:", report.get("report_status"))
print("Risk assessment:", report.get("risk_assessment"))
print("Recommendations:", report.get("recommendations"))
'@ | python -
```

The exact prediction and confidence depend on the input sample and checkpoint. This single-sample demonstration does not replace comprehensive model evaluation.

## Training and Evaluation

Retraining can take significant time and may overwrite the current checkpoint. Run it only when you intentionally want to reproduce or change the training experiment.

```powershell
python detection_engine/training/train.py
```

Evaluate the saved model using the project evaluation script:

```powershell
python detection_engine/evaluation/evaluate.py
```

Check the evaluation script's expected checkpoint and dataset paths before running it. For rigorous final research results, checkpoint selection should use a separate validation set and the test set should be reserved for final evaluation. The current training implementation selects the best checkpoint using test F1, so existing test metrics should be interpreted with that limitation in mind.

## What Is This Project Used For?

This research prototype is designed to support a security-analysis workflow:

- **Intrusion detection:** Classify processed network traffic records as Normal or Attack.
- **Explanation:** Inspect feature attributions associated with the model's selected-class score.
- **Threat context:** Retrieve relevant incident-response and threat-intelligence references from a local knowledge base.
- **Triage support:** Combine the detection output with risk assessment and follow-up recommendations.
- **Incident reporting:** Consolidate available results into a structured preliminary report.

The current project does not claim to capture live packets, block attacks, execute mitigation actions, or provide production-grade Security Operations Center monitoring. Outputs should be reviewed by a security analyst.

## Remaining Work

- Evaluate additional Normal and Attack samples and document representative results.
- Consolidate accuracy, precision, recall, F1-score, and confusion-matrix results.
- Review dataset split and checkpoint-selection methodology to avoid test-set selection bias.
- Assess explanation quality, retrieval relevance, source traceability, and recommendation usefulness.
- Document the experimental setup, limitations, and reproducible results.
- Finalize the research paper and presentation.

## Threat-Intelligence References

The local knowledge base currently contains project-authored summaries referencing:

- [NIST SP 800-61 Rev. 3 — Incident Response](https://csrc.nist.gov/pubs/sp/800/61/r3/final)
- [MITRE ATT&CK — Resources](https://attack.mitre.org/resources/)

Retrieved references provide supporting context; they are not proof that an event belongs to a specific attack technique.

## Disclaimer

This is an academic/research prototype for defensive cybersecurity analysis. Use only datasets and network traffic you are authorized to analyze. Do not use its outputs as the sole basis for consequential security decisions.

## License

Add the license and dataset attribution information applicable to this repository before redistributing the project.
