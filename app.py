
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from security_orchestrator.orchestrator import SecurityOrchestrator
from security_orchestrator.agents.memory_agent import MemoryAgent
from security_orchestrator.rag.retriever import build_local_retriever


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "processed" / "test_processed.csv"
DEMO_MEMORY_PATH = ROOT / "data" / "dashboard_demo_memory.jsonl"

st.set_page_config(
    page_title="Agentic Threat Intelligence NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. DASHBOARD STYLING
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background-color: #111c2e;
    border: 1px solid #26364d;
    padding: 14px;
    border-radius: 12px;
}

.hero {
    background: linear-gradient(120deg, #14243b, #182c46);
    border: 1px solid #2b4564;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1 style="color:#f8fafc;margin:0;">
        🛡️ Agentic Threat Intelligence NIDS
    </h1>
    <p style="color:#cbd5e1;margin-top:10px;">
        Transformer Detection · Explainable AI ·
        Threat Intelligence Retrieval · Risk Assessment
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Demonstration mode: Sequential replay of processed "
    "CIC-IDS2018 / UNSW-NB15 samples. "
    "This is not live network packet capture."
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

@st.cache_data(show_spinner="Loading processed network traffic...")
def load_data(path):
    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a target column.")

    feature_cols = [
        column for column in df.columns
        if column != "target"
    ]

    if len(feature_cols) != 147:
        raise ValueError(
            f"Expected 147 features, found {len(feature_cols)}."
        )

    if df[feature_cols].isnull().any().any():
        raise ValueError("Dataset contains missing feature values.")

    return df, feature_cols


# ============================================================
# 4. LOAD TRANSFORMER AND AGENTS
# ============================================================

@st.cache_resource(show_spinner="Loading Transformer and agents...")
def load_system(root):
    root = Path(root)

    memory = MemoryAgent(
        memory_path=root / "data" / "dashboard_demo_memory.jsonl"
    )

    retriever = build_local_retriever()

    orchestrator = SecurityOrchestrator(
        memory_agent=memory,
        retriever=retriever
    )

    return orchestrator


# ============================================================
# 5. INITIALIZATION
# ============================================================

try:
    data, feature_cols = load_data(str(DATA_PATH))
    orchestrator = load_system(str(ROOT))

except Exception as exc:
    st.error(
        f"Initialization failed: {type(exc).__name__}: {exc}"
    )
    st.info(
        "Check the project files and ensure .venv-cpu is activated."
    )
    st.stop()


# ============================================================
# 6. SESSION STATE
# ============================================================

if "replay" not in st.session_state:
    st.session_state.replay = (
        data.sample(
            n=min(1000, len(data)),
            random_state=42
        ).reset_index(drop=True)
    )

    st.session_state.position = 0
    st.session_state.events = []
    st.session_state.latest = None
    st.session_state.error = None
    st.session_state.auto_replay = False

if "auto_replay" not in st.session_state:
    st.session_state.auto_replay = False


# ============================================================
# 7. SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("⚙️ Demo Controls")
st.sidebar.caption("Dataset replay configuration")

batch_size = st.sidebar.select_slider(
    "Samples per batch",
    options=[1, 2, 5, 10],
    value=2
)

delay = st.sidebar.slider(
    "Replay interval (seconds)",
    min_value=0.0,
    max_value=1.5,
    value=0.5,
    step=0.1
)

analyst_context = st.sidebar.text_area(
    "Optional analyst investigation context (RAG)",
    placeholder=(
        "Example: network intrusion incident response, "
        "cybersecurity risk management, detection and recovery"
    ),
    height=110
)

st.sidebar.caption(
    "This context is supplied by the analyst. "
    "It is not automatically extracted from network packets."
)

remaining = (
    len(st.session_state.replay)
    - st.session_state.position
)

st.sidebar.metric("Replay samples remaining", remaining)

st.sidebar.divider()
st.sidebar.markdown("**Automatic Replay Controls**")

# Start or resume automatic replay.
if not st.session_state.auto_replay:
    if st.sidebar.button(
        "▶ Start / Resume Auto Replay",
        use_container_width=True,
        disabled=remaining <= 0
    ):
        st.session_state.auto_replay = True
        st.rerun()

# Pause automatic replay.
else:
    if st.sidebar.button(
        "⏸ Pause Auto Replay",
        use_container_width=True
    ):
        st.session_state.auto_replay = False
        st.rerun()

    st.sidebar.success("Automatic replay is running")

# Reset replay session.
if st.sidebar.button(
    "↻ Reset replay session",
    use_container_width=True
):
    st.session_state.position = 0
    st.session_state.events = []
    st.session_state.latest = None
    st.session_state.error = None
    st.session_state.auto_replay = False
    st.rerun()

# Clear session history.
if st.sidebar.button(
    "Clear dashboard session history",
    use_container_width=True
):
    st.session_state.events = []
    st.session_state.latest = None
    st.session_state.error = None
    st.rerun()

st.sidebar.divider()

st.sidebar.caption(
    "Dashboard records are stored separately in "
    "data/dashboard_demo_memory.jsonl. "
    "Production incident_memory.jsonl remains untouched."
)


# ============================================================
# 8. AUTOMATIC REFRESH
# ============================================================

if st.session_state.auto_replay:
    st_autorefresh(
        interval=max(500, int(delay * 1000)),
        limit=None,
        key="nids_auto_replay_refresh"
    )


# ============================================================
# 9. DATASET REPLAY CONTROLS
# ============================================================

st.subheader("▶ Traffic Replay")

left, right = st.columns([1, 2])

with left:
    run_batch = st.button(
        "▶ Process next batch",
        type="primary",
        use_container_width=True,
        disabled=(
            remaining <= 0
            or st.session_state.auto_replay
        )
    )

with right:
    replay_status = st.empty()

if st.session_state.auto_replay:
    st.info(
        "Automatic replay is active. "
        "Use Pause Auto Replay in the sidebar to pause it."
    )


# ============================================================
# 10. PROCESS TRAFFIC SAMPLES
# ============================================================

should_process = (
    run_batch or st.session_state.auto_replay
)

if should_process and remaining > 0:

    count = min(batch_size, remaining)

    progress = st.progress(
        0,
        text="Starting traffic replay..."
    )

    processed = 0

    for i in range(count):

        idx = st.session_state.position

        row = st.session_state.replay.iloc[idx]

        # Only 147 features are sent to the model.
        # The target column is never used for prediction.

        features = row[feature_cols].to_numpy(
            dtype=np.float32
        )

        # Ground-truth label is used only for evaluation.

        actual_label = (
            "Attack"
            if int(row["target"]) == 1
            else "Normal"
        )

        # Optional analyst-provided RAG context.

        traffic_context = None

        if analyst_context.strip():
            traffic_context = {
                "investigation_context": analyst_context.strip()
            }

        try:
            result = orchestrator.analyze(
                features,
                traffic_features=traffic_context,
                feature_names=feature_cols
            )

            detection = result.get("detection", {})
            risk = result.get("risk_assessment", {})
            threat = result.get("threat_analysis", {})

            prediction = detection.get(
                "prediction",
                "Unknown"
            )

            memory_record = result.get(
                "memory_record",
                {}
            )

            event = {
                "sample": idx + 1,

                "timestamp": memory_record.get(
                    "timestamp",
                    pd.Timestamp.now(tz="UTC").isoformat()
                ),

                "prediction": prediction,

                "actual_label": actual_label,

                "correct": prediction == actual_label,

                "confidence": float(
                    detection.get("confidence", 0.0)
                ),

                "risk_level": risk.get(
                    "risk_level",
                    "UNKNOWN"
                ),

                "xai_status": result.get(
                    "xai_status",
                    "unknown"
                ),

                "retrieved_sources": len(
                    threat.get("sources", [])
                )
            }

            st.session_state.events.append(event)
            st.session_state.latest = result
            st.session_state.error = None
            st.session_state.position += 1

            processed += 1

        except Exception as exc:
            st.session_state.error = (
                f"Sample {idx + 1}: "
                f"{type(exc).__name__}: {exc}"
            )

            # Stop automatic replay if processing fails.
            st.session_state.auto_replay = False
            break

        progress.progress(
            (i + 1) / count,
            text=f"Analysed {i + 1} of {count} samples"
        )

    progress.empty()

    if processed:
        st.success(
            f"Successfully processed {processed} traffic sample(s)."
        )

    if st.session_state.error:
        st.error(st.session_state.error)

    # Stop automatic replay when all samples are processed.
    if st.session_state.position >= len(st.session_state.replay):
        st.session_state.auto_replay = False
        st.info(
            "Replay completed. Reset the session to start again."
        )


# Display updated replay position.

replay_status.caption(
    f"Replay position: "
    f"{st.session_state.position} / "
    f"{len(st.session_state.replay)}"
)


# ============================================================
# 11. MONITORING OVERVIEW
# ============================================================

st.divider()
st.subheader("📊 Monitoring Overview")

events = st.session_state.events
total = len(events)

attack_count = sum(
    event["prediction"] == "Attack"
    for event in events
)

normal_count = sum(
    event["prediction"] == "Normal"
    for event in events
)

high_count = sum(
    event["risk_level"] == "HIGH"
    for event in events
)

agreement = (
    sum(event["correct"] for event in events)
    / total * 100
    if total else 0
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Samples analysed", total)
m2.metric("Attack predictions", attack_count)
m3.metric("Normal predictions", normal_count)
m4.metric("High-risk events", high_count)
m5.metric(
    "Replay agreement",
    f"{agreement:.1f}%" if total else "—"
)


# ============================================================
# 12. DETECTION AND RISK DISTRIBUTION
# ============================================================

if events:

    history_df = pd.DataFrame(events)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Detection distribution**")

        for label in ["Normal", "Attack"]:
            count = int(
                (history_df["prediction"] == label).sum()
            )

            percentage = count / total

            st.write(
                f"{label}: {count} ({percentage:.1%})"
            )

            st.progress(percentage)

    with c2:
        st.markdown("**Preliminary risk distribution**")

        for label in ["LOW", "MEDIUM", "HIGH"]:
            count = int(
                (history_df["risk_level"] == label).sum()
            )

            percentage = count / total

            st.write(
                f"{label}: {count} ({percentage:.1%})"
            )

            st.progress(percentage)

    st.caption(
        "Replay agreement compares model predictions with "
        "dataset target labels. This is a session-level "
        "demonstration metric, not the full model evaluation."
    )

else:
    st.info(
        "No samples analysed yet. "
        "Click Process next batch to begin."
    )


# ============================================================
# 13. LATEST TRAFFIC ANALYSIS
# ============================================================

st.divider()
st.subheader("🔎 Latest Traffic Analysis")

latest = st.session_state.latest

if latest is None:

    st.info(
        "Latest detection, XAI, threat intelligence, "
        "risk assessment and report outputs will appear here."
    )

else:

    detection = latest.get("detection", {})
    risk = latest.get("risk_assessment", {})
    threat = latest.get("threat_analysis", {})

    prediction = detection.get("prediction", "Unknown")
    confidence = float(detection.get("confidence", 0.0))
    risk_level = risk.get("risk_level", "UNKNOWN")

    # --------------------------------------------------------
    # Detection summary
    # --------------------------------------------------------

    st.markdown("### Transformer Detection")

    a, b, c = st.columns(3)

    a.metric("Prediction", prediction)
    b.metric("Confidence", f"{confidence * 100:.2f}%")
    c.metric("Preliminary risk", risk_level)

    with st.expander(
        "Detection details and class probabilities",
        expanded=True
    ):
        st.json(detection)

    # --------------------------------------------------------
    # Explainable AI
    # --------------------------------------------------------

    with st.expander(
        "🧠 Explainable AI — Gradient × Input",
        expanded=True
    ):

        if latest.get("xai_status") == "completed":

            explanation = latest.get(
                "xai_explanation",
                {}
            )

            top_features = explanation.get(
                "top_features",
                []
            )

            if top_features:

                feature_df = pd.DataFrame(top_features)

                st.markdown("**Top contributing traffic features**")

                st.markdown(
                    feature_df.to_html(
                        index=False,
                        escape=True,
                        border=0
                    ),
                    unsafe_allow_html=True
                )

                st.markdown("**Feature attribution scores**")

                for _, item in feature_df.iterrows():
                    st.write(
                        f"{item['feature']}: "
                        f"{item['attribution']:.6f}"
                    )

            st.caption(
                explanation.get(
                    "interpretation_note",
                    ""
                )
            )

            st.caption(
                "These are local Gradient × Input attributions. "
                "They are not causal explanations or calibrated risk scores."
            )

        else:
            st.warning(
                latest.get(
                    "xai_explanation",
                    {}
                ).get(
                    "reason",
                    "XAI explanation unavailable."
                )
            )

    # --------------------------------------------------------
    # Threat intelligence and RAG
    # --------------------------------------------------------

    with st.expander(
        "🔍 Threat Intelligence and RAG Evidence",
        expanded=True
    ):

        st.write(
            threat.get(
                "summary",
                "No threat summary available."
            )
        )

        sources = threat.get("sources", [])

        if sources:

            st.success(
                f"{len(sources)} reference(s) retrieved "
                "from the local knowledge base."
            )

            for i, source in enumerate(sources, start=1):

                title = source.get(
                    "title",
                    "Untitled reference"
                )

                source_name = source.get(
                    "source",
                    "Not specified"
                )

                source_url = source.get(
                    "source_url",
                    source.get("url")
                )

                score = source.get(
                    "retrieval_score",
                    source.get("score", "N/A")
                )

                st.markdown(f"**Reference {i}: {title}**")
                st.write(f"Source: {source_name}")

                if source_url:
                    st.markdown(
                        f"[Open reference]({source_url})"
                    )

                st.caption(
                    f"Matched terms: "
                    f"{source.get('matched_terms', 'N/A')} "
                    f"· Retrieval score: {score}"
                )

                content = source.get("content", "")

                if content:
                    st.write(content)

                st.divider()

        else:
            st.info(
                "No reference retrieved for this sample. "
                "You can enter relevant investigation context "
                "in the sidebar and process another sample."
            )

        observed = threat.get("observed_features", {})

        if observed:
            st.markdown(
                "**Analyst-supplied investigation context**"
            )
            st.json(observed)

    # --------------------------------------------------------
    # Risk assessment
    # --------------------------------------------------------

    with st.expander(
        "⚠️ Preliminary Risk Assessment",
        expanded=True
    ):

        st.write(
            risk.get(
                "rationale",
                ""
            )
        )

        st.caption(
            f"Assessment type: "
            f"{risk.get('assessment_type', 'Not specified')} "
            f"· Requires review: "
            f"{risk.get('requires_review', False)}"
        )

    # --------------------------------------------------------
    # Incident report
    # --------------------------------------------------------

    with st.expander("📄 Preliminary Incident Report"):

        report = latest.get("report", {})

        st.json(report)

        st.download_button(
            "⬇ Download incident report (JSON)",

            data=json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                default=str
            ),

            file_name=(
                f"incident_report_"
                f"{report.get('report_id', 'latest')}.json"
            ),

            mime="application/json"
        )


# ============================================================
# 14. INCIDENT HISTORY
# ============================================================

st.divider()
st.subheader("🧾 Dashboard Incident History")

if events:

    history = pd.DataFrame(events).sort_values(
        "sample",
        ascending=False
    )

    st.markdown(
        history.to_html(
            index=False,
            escape=True,
            border=0
        ),
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇ Download session history (CSV)",

        data=history.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="nids_dashboard_session_history.csv",

        mime="text/csv"
    )

else:

    st.caption(
        "Incident history will appear here as samples are analysed."
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype | Binary Normal/Attack classification | "
    "Preliminary rule-based risk assessment | "
    "Local keyword-based RAG | "
    "Dataset replay rather than live network packet capture."
)
