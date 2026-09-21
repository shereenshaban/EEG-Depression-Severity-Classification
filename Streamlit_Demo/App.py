# Design philosophy: Quiet Neural Lab — clear research UI, warm paper background,
# neural teal accents, generous spacing, and transparent non-diagnostic language.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Neural Lab | EEG Severity Demo",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROUP_LABELS = {1: "Minimal", 2: "Mild", 3: "Moderate"}
LABEL_COLORS = {"Minimal": "#0F5B66", "Mild": "#D9A441", "Moderate": "#C96B57"}
META_COLUMNS = {"Group & ID", "Group", "Label", "Gender", "BDI_II", "PHQ9"}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
    :root { --ink:#17313A; --teal:#0F5B66; --paper:#F7F5EF; --mist:#E7EFEC; --gold:#D9A441; --coral:#C96B57; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #EEF3F0; border-right: 1px solid #D9E4DF; }
    h1,h2,h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: var(--ink); letter-spacing:-0.03em; }
    p, label, .stMarkdown, .stCaption, div { font-family: 'DM Sans', sans-serif; }
    .hero { background: linear-gradient(135deg, #0F5B66 0%, #17313A 100%); border-radius: 24px; padding: 2.4rem 2.8rem; color:white; margin-bottom:1.2rem; position:relative; overflow:hidden; }
    .hero:after { content:'〰 〰 〰'; position:absolute; right:2rem; bottom:.7rem; font-size:4rem; color:rgba(255,255,255,.12); letter-spacing:.15rem; transform:rotate(-3deg); }
    .eyebrow { text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:700; color:#B8E2DD; margin-bottom:.7rem; }
    .hero h1 { color:white !important; font-size:clamp(2rem,4vw,3.35rem); margin:0; max-width:850px; }
    .hero p { color:#D9EEEA; font-size:1.05rem; max-width:720px; line-height:1.55; margin:.9rem 0 0; }
    .section-label { color:var(--teal); text-transform:uppercase; letter-spacing:.16em; font-weight:700; font-size:.72rem; margin-top:1.4rem; }
    .result-card { background:#FFFFFF; border:1px solid #D9E4DF; border-left:7px solid var(--teal); border-radius:18px; padding:1.2rem 1.35rem; box-shadow:0 12px 30px rgba(23,49,58,.08); }
    .result-card h2 { margin:.1rem 0 .25rem; font-size:2rem; }
    .metric { background:#FFFFFF; border:1px solid #D9E4DF; border-radius:15px; padding:1rem; min-height:105px; }
    .metric-label { color:#60747A; font-size:.73rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
    .metric-value { color:var(--teal); font-size:1.7rem; font-weight:800; margin-top:.4rem; }
    .notice { background:#FFF8E8; border:1px solid #E8CD8A; color:#614A16; border-radius:14px; padding:1rem 1.1rem; line-height:1.5; }
    .small-note { color:#60747A; font-size:.86rem; line-height:1.45; }
    .stButton > button { background:var(--teal); color:white; border:0; border-radius:11px; padding:.65rem 1.2rem; font-weight:700; }
    .stButton > button:hover { background:#0B4B54; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_table(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def prepare_features(df):
    feature_columns = [c for c in df.columns if c not in META_COLUMNS]
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    return X, feature_columns


def train_demo_model(df, feature_columns):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(score_func=f_classif, k=min(100, len(feature_columns)))),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_features="log2",
                    min_samples_leaf=1,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(df[feature_columns].apply(pd.to_numeric, errors="coerce"), df["Group"].astype(int))
    return model


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Quiet Neural Lab · Research Prototype</div>
      <h1>From EEG features to a severity signal.</h1>
      <p>Explore how a participant-level, eyes-closed EEG feature profile is mapped to Minimal, Mild, or Moderate depression severity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Demo setup")
    st.write("Upload the participant-level `final_df.csv` or Excel file created by the EDA notebook.")
    uploaded = st.file_uploader("Upload final_df", type=["csv", "xlsx"])
    st.markdown("---")
    st.markdown("**Expected structure**")
    st.caption("One row per participant, with 600 numeric EEG features and a `Group` column for the research demo.")
    st.markdown("---")
    st.markdown("**Research notice**")
    st.caption("This prototype is for research demonstration only. It is not a clinical diagnostic tool.")

if uploaded is None:
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown('<div class="section-label">Start here</div>', unsafe_allow_html=True)
        st.header("Load a participant-level EEG table")
        st.write("Use the `final_df.csv` exported from your EDA notebook. The app will validate the schema, train the documented demo pipeline, and let you inspect one participant at a time.")
        st.markdown('<div class="notice">The first version of this demo intentionally uses the processed EEG feature table, not raw EEG. Raw-signal processing requires the same channel order, sampling frequency, segmentation, and feature extraction pipeline used during training.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-label">Model snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric"><div class="metric-label">Pipeline</div><div class="metric-value">Extra Trees</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric"><div class="metric-label">Feature selection</div><div class="metric-value">Top 100</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric"><div class="metric-label">Classes</div><div class="metric-value">3 levels</div></div>', unsafe_allow_html=True)
    st.stop()

try:
    data = load_table(uploaded)
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

data.columns = data.columns.astype(str).str.strip()
required = {"Group", "Group & ID"}
missing_required = required - set(data.columns)
if missing_required:
    st.error(f"Missing required columns: {', '.join(sorted(missing_required))}")
    st.stop()

X, feature_columns = prepare_features(data)
if len(feature_columns) < 100:
    st.error(f"Only {len(feature_columns)} numeric feature columns were found. The demo expects the processed EEG table with approximately 600 EEG features.")
    st.stop()

with st.spinner("Validating features and fitting the demo pipeline…"):
    model = train_demo_model(data, feature_columns)

st.success(f"Schema validated: {len(data)} participants and {len(feature_columns)} numeric EEG features found.")

st.markdown('<div class="section-label">Participant explorer</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1.2, 1], gap="large")
with col_a:
    ids = data["Group & ID"].astype(str).tolist()
    selected_id = st.selectbox("Choose a participant for the demonstration", ids)
with col_b:
    st.markdown('<div class="small-note">For a competition presentation, select a participant, click Analyze, then explain the probabilities as model outputs—not clinical certainty.</div>', unsafe_allow_html=True)

selected_row = data.loc[data["Group & ID"].astype(str) == selected_id].iloc[[0]]

if st.button("Analyze EEG features", type="primary"):
    selected_X = selected_row[feature_columns].apply(pd.to_numeric, errors="coerce")
    prediction = int(model.predict(selected_X)[0])
    probabilities = model.predict_proba(selected_X)[0]
    classes = model.named_steps["model"].classes_
    prob_map = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    predicted_label = GROUP_LABELS[prediction]

    st.markdown('<div class="section-label">Prediction ready</div>', unsafe_allow_html=True)
    result_col, chart_col = st.columns([0.9, 1.1], gap="large")
    with result_col:
        st.markdown(
            f'<div class="result-card"><div class="metric-label">Predicted severity</div><h2 style="color:{LABEL_COLORS[predicted_label]};">{predicted_label}</h2><div class="small-note">Participant: {selected_id}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown('<div class="notice">Research prototype only — this output does not constitute a clinical diagnosis or treatment recommendation.</div>', unsafe_allow_html=True)
    with chart_col:
        probability_df = pd.DataFrame(
            {"Severity": [GROUP_LABELS.get(int(cls), str(cls)) for cls in classes], "Probability": probabilities}
        ).set_index("Severity")
        st.bar_chart(probability_df, y="Probability", color="#0F5B66")

    with st.expander("Show validation details"):
        true_group = int(selected_row["Group"].iloc[0])
        st.write({
            "Predicted group": prediction,
            "Reference group in uploaded table": true_group,
            "Match": prediction == true_group,
            "Feature count used": len(feature_columns),
        })
        st.caption("The reference group is shown only for demonstration validation. It is not used as an input feature.")

st.markdown("---")
st.markdown('<div class="small-note">The demo retrains the documented Extra Trees configuration on the uploaded labeled table for presentation purposes. The reported held-out test and repeated cross-validation metrics belong to the separate Modeling notebook and should not be inferred from this interactive display.</div>', unsafe_allow_html=True)

