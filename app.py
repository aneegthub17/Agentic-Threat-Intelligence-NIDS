
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from streamlit_autorefresh import st_autorefresh

from security_orchestrator.orchestrator import SecurityOrchestrator
from security_orchestrator.agents.memory_agent import MemoryAgent
from security_orchestrator.rag.retriever import build_local_retriever


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "datasets" / "processed" / "test_processed.csv"
METADATA_PATH = ROOT / "datasets" / "merged" / "unified_test.csv"
MEMORY_PATH = ROOT / "data" / "dashboard_demo_memory.jsonl"

EXPECTED_FEATURES = 147
REPLAY_SIZE = 1000

RAW_FLOW_COLUMNS = [
    "protocol",
    "duration",
    "src_packets",
    "dst_packets",
    "src_bytes",
    "dst_bytes",
    "src_pkt_size_mean",
    "dst_pkt_size_mean",
    "src_iat_mean",
    "dst_iat_mean",
    "src_tcp_window",
    "dst_tcp_window",
    "src_packet_rate",
    "dst_packet_rate",
    "src_bytes_rate",
    "dst_bytes_rate",
]

GROUND_TRUTH_COLUMNS = [
    "dataset_source",
    "target",
    "attack_category",
]

st.set_page_config(
    page_title="Agentic Threat Intelligence NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. STYLING
# ============================================================

st.markdown(
    """
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
        border-radius: 12px;
        padding: 14px;
    }

    .hero {
        background: linear-gradient(120deg, #14243b, #182c46);
        border: 1px solid #2b4564;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="color:#f8fafc;margin:0">
            🛡️ Agentic Threat Intelligence NIDS
        </h1>
        <p style="color:#cbd5e1;margin-top:10px">
            Transformer Detection · Explainable AI ·
            Threat Intelligence Retrieval · Risk Assessment
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Research demonstration using recorded network-flow datasets. "
    "This is not live packet capture."
)


# ============================================================
# 3. DATA LOADING AND VALIDATION
# ============================================================

@st.cache_data(
    show_spinner="Loading processed and raw network traffic..."
)
def load_data(processed_path, metadata_path):

    processed = pd.read_csv(processed_path)

    raw = pd.read_csv(
        metadata_path,
        usecols=GROUND_TRUTH_COLUMNS + RAW_FLOW_COLUMNS,
    )

    if "target" not in processed.columns:
        raise ValueError(
            "Processed dataset is missing target."
        )

    if len(processed) != len(raw):
        raise ValueError(
            "Processed and raw datasets have different row counts."
        )

    processed_targets = pd.to_numeric(
        processed["target"],
        errors="raise",
    ).to_numpy()

    raw_targets = pd.to_numeric(
        raw["target"],
        errors="raise",
    ).to_numpy()

    if not np.array_equal(
        processed_targets,
        raw_targets,
    ):
        raise ValueError(
            "Processed and raw target ordering does not match."
        )

    if raw[
        ["dataset_source", "attack_category"]
    ].isna().any().any():
        raise ValueError(
            "Ground-truth metadata contains missing values."
        )

    # Matching target sequences is a partial alignment check.
    # This assumes preprocessing preserved original row order.
    #
    # Raw features are investigation-only.
    # They must never be included in the 147 model features.

    invalid_counts = {}

    for column in RAW_FLOW_COLUMNS:

        if column == "protocol":

            values = (
                raw[column]
                .fillna("unknown")
                .astype(str)
                .str.strip()
            )

            values = values.replace(
                "",
                "unknown",
            )

        else:

            values = pd.to_numeric(
                raw[column],
                errors="coerce",
            )

            # Convert infinite measurements to missing.
            values = values.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            invalid_counts[column] = int(
                values.isna().sum()
            )

            # Preserve missingness rather than inventing zero.
            # Missing values become None during extraction.

        processed[f"raw_{column}"] = (
            values.to_numpy()
        )

    processed["dataset_source"] = (
        raw["dataset_source"]
        .astype(str)
        .to_numpy()
    )

    processed["attack_category"] = (
        raw["attack_category"]
        .astype(str)
        .to_numpy()
    )

    excluded = {
        "target",
        "dataset_source",
        "attack_category",
    }

    excluded.update(
        f"raw_{column}"
        for column in RAW_FLOW_COLUMNS
    )

    feature_columns = [
        column
        for column in processed.columns
        if column not in excluded
    ]

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} model features, "
            f"found {len(feature_columns)}."
        )

    model_values = processed[
        feature_columns
    ].to_numpy(dtype=np.float32)

    if not np.isfinite(model_values).all():
        raise ValueError(
            "Processed Transformer features contain "
            "missing or infinite values."
        )

    return processed, feature_columns, invalid_counts


# ============================================================
# 4. INITIALIZE SECURITY ORCHESTRATOR
# ============================================================

@st.cache_resource(
    show_spinner="Loading Transformer and security agents..."
)
def load_system():

    MEMORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory = MemoryAgent(
        memory_path=MEMORY_PATH
    )

    retriever = build_local_retriever()

    return SecurityOrchestrator(
        memory_agent=memory,
        retriever=retriever,
    )


try:

    data, feature_cols, invalid_counts = load_data(
        str(DATA_PATH),
        str(METADATA_PATH),
    )

    orchestrator = load_system()

except Exception as exc:

    st.error(
        f"Initialization failed: "
        f"{type(exc).__name__}: {exc}"
    )

    st.stop()


# ============================================================
# 5. DEVICE STATUS
# ============================================================

st.sidebar.title("⚙️ Demo Controls")

st.sidebar.caption(
    f"PyTorch version: {torch.__version__}"
)

if torch.cuda.is_available():

    st.sidebar.success(
        f"CUDA available: {torch.cuda.get_device_name(0)}"
    )

else:

    st.sidebar.warning(
        "CUDA unavailable; CPU fallback expected."
    )

st.sidebar.caption(
    "CUDA availability does not independently confirm "
    "the loaded model's device."
)


# ============================================================
# 6. RAW DATA QUALITY
# ============================================================

total_invalid = sum(
    invalid_counts.values()
)

with st.sidebar.expander(
    "Raw traffic data quality"
):

    st.write(
        f"Unavailable numeric measurements: "
        f"{total_invalid:,}"
    )

    affected = {
        column: count
        for column, count in invalid_counts.items()
        if count > 0
    }

    if affected:

        st.json(affected)

    else:

        st.success(
            "No missing numeric raw measurements."
        )

    st.caption(
        "Unavailable raw measurements are excluded "
        "from numerical interpretation, not replaced "
        "with invented traffic values."
    )


# ============================================================
# 7. SESSION STATE
# ============================================================

if "replay" not in st.session_state:

    st.session_state.replay = (
        data.sample(
            n=min(REPLAY_SIZE, len(data)),
            random_state=42,
        ).reset_index(drop=True)
    )

    st.session_state.position = 0
    st.session_state.events = []
    st.session_state.latest = None
    st.session_state.latest_metadata = None
    st.session_state.auto_replay = False
    st.session_state.error = None

if "auto_replay" not in st.session_state:
    st.session_state.auto_replay = False

if "latest_metadata" not in st.session_state:
    st.session_state.latest_metadata = None


# ============================================================
# 8. SIDEBAR CONTROLS
# ============================================================

batch_size = st.sidebar.select_slider(
    "Samples per batch",
    options=[1, 2, 5, 10],
    value=2,
)

delay = st.sidebar.slider(
    "Replay interval (seconds)",
    min_value=0.0,
    max_value=1.5,
    value=0.5,
    step=0.1,
)

analyst_context = st.sidebar.text_area(
    "Optional analyst investigation context",
    placeholder=(
        "Example: repeated login failures "
        "or suspicious network behavior"
    ),
    height=100,
)

st.sidebar.caption(
    "Raw network-flow measurements are automatically "
    "extracted. Analyst context is optional."
)

remaining = (
    len(st.session_state.replay)
    - st.session_state.position
)

st.sidebar.metric(
    "Replay samples remaining",
    remaining,
)

st.sidebar.divider()

if not st.session_state.auto_replay:

    if st.sidebar.button(
        "▶ Start / Resume Auto Replay",
        use_container_width=True,
        disabled=remaining <= 0,
    ):

        st.session_state.auto_replay = True
        st.rerun()

else:

    if st.sidebar.button(
        "⏸ Pause Auto Replay",
        use_container_width=True,
    ):

        st.session_state.auto_replay = False
        st.rerun()

    st.sidebar.success(
        "Automatic replay is running"
    )


if st.sidebar.button(
    "↻ Reset replay session",
    use_container_width=True,
):

    st.session_state.position = 0
    st.session_state.events = []
    st.session_state.latest = None
    st.session_state.latest_metadata = None
    st.session_state.error = None
    st.session_state.auto_replay = False

    st.rerun()


if st.sidebar.button(
    "Clear dashboard session history",
    use_container_width=True,
):

    st.session_state.events = []
    st.session_state.latest = None
    st.session_state.latest_metadata = None
    st.session_state.error = None

    st.rerun()


st.sidebar.caption(
    "Incident memory: data/dashboard_demo_memory.jsonl"
)


# ============================================================
# 9. AUTO REFRESH
# ============================================================

if st.session_state.auto_replay:

    st_autorefresh(
        interval=max(
            500,
            int(delay * 1000),
        ),
        limit=None,
        key="nids_auto_replay_refresh",
    )


# ============================================================
# 10. TRAFFIC REPLAY
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
        ),
    )

with right:

    st.caption(
        f"Replay position: "
        f"{st.session_state.position} / "
        f"{len(st.session_state.replay)}"
    )


# ============================================================
# 11. EXTRACT ORIGINAL NETWORK FLOW
# ============================================================

def extract_raw_flow(row):
    """
    Extract raw traffic measurements for investigation.

    Never pass target, attack_category, dataset_source,
    or other ground-truth labels to the agents.
    """

    context = {}

    for column in RAW_FLOW_COLUMNS:

        value = row[f"raw_{column}"]

        if column == "protocol":

            context[column] = str(value)

        elif pd.isna(value):

            # Missing/invalid measurement.
            # Do not pretend it is zero.
            context[column] = None

        else:

            numeric_value = float(value)

            if np.isfinite(numeric_value):
                context[column] = numeric_value
            else:
                context[column] = None

    return context


# ============================================================
# 12. PROCESS TRAFFIC
# ============================================================

should_process = (
    run_batch
    or st.session_state.auto_replay
)

if should_process and remaining > 0:

    count = min(
        batch_size,
        remaining,
    )

    progress = st.progress(
        0,
        text="Starting traffic analysis...",
    )

    processed_count = 0

    for i in range(count):

        idx = st.session_state.position

        row = st.session_state.replay.iloc[idx]

        # Transformer receives ONLY 147 processed features.
        features = row[feature_cols].to_numpy(
            dtype=np.float32
        )

        # Investigation receives raw flow measurements.
        traffic_context = extract_raw_flow(row)

        if analyst_context.strip():

            traffic_context["investigation_context"] = (
                analyst_context.strip()
            )

        # Ground truth remains evaluation-only.
        actual_label = (
            "Attack"
            if int(row["target"]) == 1
            else "Normal"
        )

        actual_category = str(
            row["attack_category"]
        )

        dataset_source = str(
            row["dataset_source"]
        )

        try:

            result = orchestrator.analyze(
                features,
                traffic_features=traffic_context,
                feature_names=feature_cols,
            )

            detection = result.get(
                "detection", {}
            )

            threat = result.get(
                "threat_analysis", {}
            )

            risk = result.get(
                "risk_assessment", {}
            )

            memory_record = result.get(
                "memory_record", {}
            )

            prediction = detection.get(
                "prediction", "Unknown"
            )

            family_analysis = threat.get(
                "attack_family_analysis", {}
            )

            event = {
                "sample": idx + 1,
                "timestamp": memory_record.get(
                    "timestamp",
                    pd.Timestamp.now(
                        tz="UTC"
                    ).isoformat(),
                ),
                "dataset_source": dataset_source,
                "prediction": prediction,
                "actual_label": actual_label,
                "ground_truth_attack_category": (
                    actual_category
                ),
                "agent_attack_family": (
                    family_analysis.get(
                        "family", "Undetermined"
                    )
                ),
                "correct": (
                    prediction == actual_label
                ),
                "confidence": float(
                    detection.get(
                        "confidence", 0.0
                    )
                ),
                "risk_level": risk.get(
                    "risk_level", "UNKNOWN"
                ),
                "xai_status": result.get(
                    "xai_status", "unknown"
                ),
                "retrieved_sources": len(
                    threat.get("sources", [])
                ),
            }

            st.session_state.events.append(
                event
            )

            st.session_state.latest = result

            st.session_state.latest_metadata = {
                "dataset_source": dataset_source,
                "actual_label": actual_label,
                "attack_category": actual_category,
                "raw_flow": traffic_context,
            }

            st.session_state.position += 1
            st.session_state.error = None

            processed_count += 1

        except Exception as exc:

            st.session_state.error = (
                f"Sample {idx + 1}: "
                f"{type(exc).__name__}: {exc}"
            )

            st.session_state.auto_replay = False
            break

        progress.progress(
            (i + 1) / count,
            text=(
                f"Analysed {i + 1} "
                f"of {count} samples"
            ),
        )

    progress.empty()

    if processed_count:

        st.success(
            f"Successfully processed "
            f"{processed_count} traffic sample(s)."
        )

    if st.session_state.error:

        st.error(
            st.session_state.error
        )

    if (
        st.session_state.position
        >= len(st.session_state.replay)
    ):

        st.session_state.auto_replay = False

        st.info(
            "Replay completed. Reset the session "
            "to start again."
        )


# ============================================================
# 13. MONITORING OVERVIEW
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
    100 * sum(
        event["correct"]
        for event in events
    ) / total
    if total
    else 0
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Samples analysed",
    total,
)

m2.metric(
    "Attack predictions",
    attack_count,
)

m3.metric(
    "Normal predictions",
    normal_count,
)

m4.metric(
    "High-risk events",
    high_count,
)

m5.metric(
    "Replay agreement",
    f"{agreement:.1f}%" if total else "—",
)


# ============================================================
# 14. DETECTION AND RISK DISTRIBUTION
# ============================================================

if events:

    history_df = pd.DataFrame(events)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "**Detection distribution**"
        )

        for label in ["Normal", "Attack"]:

            count = int(
                (
                    history_df["prediction"]
                    == label
                ).sum()
            )

            st.write(
                f"{label}: {count} "
                f"({count / total:.1%})"
            )

            st.progress(
                count / total
            )

    with c2:

        st.markdown(
            "**Preliminary risk distribution**"
        )

        for label in ["LOW", "MEDIUM", "HIGH"]:

            count = int(
                (
                    history_df["risk_level"]
                    == label
                ).sum()
            )

            st.write(
                f"{label}: {count} "
                f"({count / total:.1%})"
            )

            st.progress(
                count / total
            )

    st.caption(
        "Replay agreement is a session-level metric, "
        "not a full model evaluation."
    )

else:

    st.info(
        "No samples analysed yet. "
        "Process a batch to begin."
    )


# ============================================================
# 15. LATEST TRAFFIC ANALYSIS
# ============================================================

st.divider()
st.subheader("🔎 Latest Traffic Analysis")

latest = st.session_state.latest

if latest is None:

    st.info(
        "Detection, XAI, threat intelligence, "
        "risk assessment and incident reporting "
        "will appear here."
    )

else:

    detection = latest.get(
        "detection", {}
    )

    threat = latest.get(
        "threat_analysis", {}
    )

    risk = latest.get(
        "risk_assessment", {}
    )

    prediction = detection.get(
        "prediction", "Unknown"
    )

    confidence = float(
        detection.get(
            "confidence", 0.0
        )
    )

    risk_level = risk.get(
        "risk_level", "UNKNOWN"
    )

    st.markdown(
        "### Transformer Detection"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Prediction",
        prediction,
    )

    b.metric(
        "Confidence",
        f"{confidence * 100:.2f}%",
    )

    c.metric(
        "Preliminary risk",
        risk_level,
    )

    with st.expander(
        "Detection details and class probabilities",
        expanded=True,
    ):

        st.json(detection)


    # ========================================================
    # DATASET GROUND TRUTH
    # ========================================================

    st.markdown(
        "### Dataset Ground Truth"
    )

    metadata = (
        st.session_state.latest_metadata
        or {}
    )

    a, b, c = st.columns(3)

    a.metric(
        "Dataset source",
        metadata.get(
            "dataset_source", "Unavailable"
        ),
    )

    b.metric(
        "Actual binary label",
        metadata.get(
            "actual_label", "Unavailable"
        ),
    )

    c.metric(
        "Original attack category",
        metadata.get(
            "attack_category", "Unavailable"
        ),
    )

    st.caption(
        "Ground-truth labels are displayed only "
        "for evaluation. They are never passed "
        "to the Transformer or investigation agents."
    )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    with st.expander(
        "🧠 Explainable AI — Gradient × Input",
        expanded=True,
    ):

        if latest.get(
            "xai_status"
        ) == "completed":

            explanation = latest.get(
                "xai_explanation", {}
            )

            top_features = explanation.get(
                "top_features", []
            )

            if top_features:

                feature_df = pd.DataFrame(
                    top_features
                )

                st.markdown(
                    "**Top contributing features**"
                )

                st.markdown(
                    feature_df.to_html(
                        index=False,
                        escape=True,
                        border=0,
                    ),
                    unsafe_allow_html=True,
                )

                for _, item in feature_df.iterrows():

                    st.write(
                        f"{item['feature']}: "
                        f"{item['attribution']:.6f}"
                    )

            st.caption(
                explanation.get(
                    "interpretation_note", ""
                )
            )

            st.caption(
                "Gradient × Input provides local "
                "feature attributions, not causal explanations."
            )

        else:

            st.warning(
                latest.get(
                    "xai_explanation", {}
                ).get(
                    "reason",
                    "XAI explanation unavailable.",
                )
            )


    # ========================================================
    # AUTOMATIC RAW TRAFFIC EVIDENCE
    # ========================================================

    with st.expander(
        "📡 Automatically Extracted Network Flow",
        expanded=True,
    ):

        raw_flow = metadata.get(
            "raw_flow", {}
        )

        if raw_flow:

            st.json(raw_flow)

            unavailable = [
                key
                for key, value in raw_flow.items()
                if value is None
            ]

            if unavailable:

                st.warning(
                    "Unavailable measurements: "
                    + ", ".join(unavailable)
                )

            st.caption(
                "These are original, unscaled traffic "
                "measurements. Missing values are shown "
                "as null and must not be interpreted as zero."
            )

        else:

            st.info(
                "No raw flow measurements available."
            )


    # ========================================================
    # THREAT INTELLIGENCE AND RAG
    # ========================================================

    with st.expander(
        "🔍 Threat Intelligence and RAG Evidence",
        expanded=True,
    ):

        st.write(
            threat.get(
                "summary",
                "No threat summary available.",
            )
        )

        family_analysis = threat.get(
            "attack_family_analysis", {}
        )

        if family_analysis:

            st.markdown(
                "**Agent attack-family hypothesis**"
            )

            st.write(
                family_analysis.get(
                    "family", "Undetermined"
                )
            )

            st.caption(
                f"Method: "
                f"{family_analysis.get('method', 'unknown')} "
                f"· Certainty: "
                f"{family_analysis.get('certainty', 'unknown')}"
            )

            if family_analysis.get("evidence"):

                st.write(
                    "Supporting investigation evidence:"
                )

                st.json(
                    family_analysis["evidence"]
                )

            st.caption(
                "This is a heuristic investigation result, "
                "not a trained multiclass prediction."
            )

        sources = threat.get(
            "sources", []
        )

        if sources:

            st.success(
                f"{len(sources)} local reference(s) retrieved."
            )

            for i, source in enumerate(
                sources,
                start=1,
            ):

                st.markdown(
                    f"**Reference {i}: "
                    f"{source.get('title', 'Untitled')}**"
                )

                st.write(
                    f"Source: "
                    f"{source.get('source', 'Unknown')}"
                )

                url = source.get(
                    "source_url",
                    source.get("url"),
                )

                if url:

                    st.markdown(
                        f"[Open reference]({url})"
                    )

                st.caption(
                    f"Matched terms: "
                    f"{source.get('matched_terms', [])} "
                    f"· Retrieval score: "
                    f"{source.get('retrieval_score', source.get('score', 'N/A'))}"
                )

                if source.get("content"):

                    st.write(
                        source["content"]
                    )

                st.divider()

        else:

            st.info(
                "No matching local threat reference "
                "was retrieved."
            )

        observed = threat.get(
            "observed_features", {}
        )

        if observed:

            st.markdown(
                "**Investigation evidence received by agent**"
            )

            st.json(observed)


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    with st.expander(
        "⚠️ Preliminary Risk Assessment",
        expanded=True,
    ):

        st.write(
            risk.get(
                "rationale", ""
            )
        )

        indicators = risk.get(
            "critical_indicators", []
        )

        if indicators:

            st.markdown(
                "**Explicit critical indicators**"
            )

            st.json(indicators)

        st.caption(
            f"Assessment type: "
            f"{risk.get('assessment_type', 'unknown')} "
            f"· Requires review: "
            f"{risk.get('requires_review', False)}"
        )

        st.caption(
            "Model confidence and generic RAG references "
            "are not independent proof of attack severity."
        )


    # ========================================================
    # INCIDENT REPORT
    # ========================================================

    with st.expander(
        "📄 Preliminary Incident Report"
    ):

        report = latest.get(
            "report", {}
        )

        st.json(report)

        st.download_button(
            "⬇ Download incident report (JSON)",
            data=json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            file_name=(
                f"incident_report_"
                f"{report.get('report_id', 'latest')}.json"
            ),
            mime="application/json",
        )


# ============================================================
# 16. INCIDENT HISTORY
# ============================================================

st.divider()
st.subheader("🧾 Dashboard Incident History")

if events:

    history = (
        pd.DataFrame(events)
        .sort_values(
            "sample",
            ascending=False,
        )
    )

    st.markdown(
        history.to_html(
            index=False,
            escape=True,
            border=0,
        ),
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇ Download session history (CSV)",
        data=history.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="nids_dashboard_session_history.csv",
        mime="text/csv",
    )

else:

    st.caption(
        "Incident history will appear here "
        "as samples are analysed."
    )


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype | Binary Normal/Attack Transformer | "
    "Automatic raw-flow extraction | "
    "Missing-value-aware investigation data | "
    "Explainable AI | Preliminary rule-based risk | "
    "Local keyword-based RAG | Dataset replay, "
    "not live packet capture."
)
