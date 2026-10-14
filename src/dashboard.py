"""Streamlit dashboard shell for RAGAS evaluation results."""

import streamlit as st

st.set_page_config(
    page_title="RAGAS Eval Pipeline",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #020617; }
        h1, h2, h3 { color: #f8fafc !important; font-family: Inter, sans-serif; }
        .metric-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 600;
            color: #22d3ee;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #64748b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAGAS Eval Pipeline")
st.caption("Evaluating RAG pipelines with faithfulness, relevancy, precision, and recall.")

columns = st.columns(4)
for column, label in zip(
    columns,
    ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">-</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.info("Run an evaluation first: python -m src.evaluate")