"""
app.py — Streamlit UI for the Indian Spam Detector (professional design).
Run with: streamlit run app.py
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from features import extract_signals, signal_messages

# ------------------- Config -------------------
MODEL_PATH = Path("model") / "model.joblib"
REPORT_PATH = Path("model") / "report.json"
LABELS = ["normal", "spam", "scam"]
LABEL_DISPLAY = {"normal": "Normal", "spam": "Spam", "scam": "Scam"}
LABEL_ICONS   = {"normal": "✅", "spam": "⚠️", "scam": "🚫"}
LABEL_COLORS  = {"normal": "#16a34a", "spam": "#d97706", "scam": "#dc2626"}
LABEL_BG      = {"normal": "#f0fdf4", "spam": "#fffbeb", "scam": "#fef2f2"}

EXAMPLES = {
    "normal": "Papa, main late aunga aaj office se. Dinner ready rakho.",
    "spam":   "50% off on all shoes! Limited time offer. Shop now at www.shopzone.com",
    "scam":   "Dear customer, your KYC is pending. Account will be blocked in 24 hours. Update at http://sbi-kyc-update.xyz",
}

st.set_page_config(
    page_title="Indian Spam Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------- Custom CSS -------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container {
        max-width: 780px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Hero header */
    .hero {
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #1e293b 0%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Section labels */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 1.25rem 0 0.5rem 0;
    }

    /* Example buttons */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        color: #334155 !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04) !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #94a3b8 !important;
        background: #f8fafc !important;
        transform: translateY(-1px);
    }

    /* Textarea */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 14px 16px !important;
        background: #ffffff !important;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    /* Primary predict button */
    .stButton > button[kind="primary"] {
        width: 100%;
        border-radius: 10px !important;
        background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.7rem 1rem !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79,70,229,0.25) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(79,70,229,0.35) !important;
    }

    /* Result card */
    .result-card {
        border-radius: 14px;
        padding: 1.5rem 1.5rem 1.25rem 1.5rem;
        margin: 1.5rem 0 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    }
    .result-label {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }
    .result-conf {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    .result-conf b { color: #1e293b; }

    /* Probability bar */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.4rem 0;
    }
    .prob-label {
        width: 80px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
    }
    .prob-track {
        flex: 1;
        height: 10px;
        background: #f1f5f9;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
    .prob-value {
        width: 52px;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* Why section */
    .why-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid #e2e8f0;
        margin-top: 0.5rem;
    }
    .why-item {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        padding: 0.35rem 0;
        font-size: 0.9rem;
        color: #334155;
    }
    .why-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #6366f1;
        margin-top: 0.5rem;
        flex-shrink: 0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* Footer caption */
    .disclaimer {
        margin-top: 2rem;
        padding: 0.9rem 1.1rem;
        background: #fefce8;
        border-left: 3px solid #eab308;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #713f12;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------- Cached loaders -------------------
@st.cache_resource
def load_model():
    """Load the model file only once."""
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_report():
    """Load report.json only once."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------- UI helpers -------------------
def _prob_bar(label, value, color):
    """Return an HTML row for one probability bar."""
    pct = max(0.0, min(1.0, float(value))) * 100
    return f"""
    <div class="prob-row">
        <div class="prob-label">{label}</div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{pct:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-value">{pct:.1f}%</div>
    </div>
    """


def show_result(text):
    """Show prediction, confidence, styled probability bars, and 'Why?' section."""
    model = load_model()
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    proba_map = {c: float(p) for c, p in zip(classes, proba)}

    pred = max(proba_map, key=proba_map.get)
    confidence = proba_map[pred] * 100.0

    color = LABEL_COLORS[pred]
    bg    = LABEL_BG[pred]
    icon  = LABEL_ICONS[pred]

    # Result card
    st.markdown(
        f"""
        <div class="result-card" style="background:{bg}; border-left:5px solid {color};">
            <div class="result-label" style="color:{color};">
                <span>{icon}</span>
                <span>{LABEL_DISPLAY[pred]}</span>
            </div>
            <div class="result-conf">Confidence: <b>{confidence:.1f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability bars
    st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
    bars_html = "".join(
        _prob_bar(LABEL_DISPLAY[c], proba_map.get(c, 0.0), LABEL_COLORS[c])
        for c in LABELS
    )
    st.markdown(bars_html, unsafe_allow_html=True)

    # Why section
    st.markdown('<div class="section-label">Why?</div>', unsafe_allow_html=True)
    signals = extract_signals(text)
    items = signal_messages(signals)
    why_html = '<div class="why-box">' + "".join(
        f'<div class="why-item"><span class="why-dot"></span><span>{m}</span></div>'
        for m in items
    ) + "</div>"
    st.markdown(why_html, unsafe_allow_html=True)


def render_model_info():
    """Content for the 'About the model' expander."""
    try:
        report = load_report()
    except FileNotFoundError:
        st.info("Report file not found. Please run `python train.py` first.")
        return

    best = report.get("best_model", "unknown")
    st.markdown(f"**Best model:** `{best}`")

    for name, metrics in report.get("models", {}).items():
        st.markdown(f"#### {name.replace('_', ' ').title()}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        c2.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        c3.metric("CV F1 (5-fold)",
                  f"{metrics['cv_macro_f1_mean']:.3f}")

        f1_df = pd.DataFrame([
            {"Label": LABEL_DISPLAY[lbl],
             "Precision": metrics["precision"][lbl],
             "Recall": metrics["recall"][lbl],
             "F1": metrics["f1"][lbl]}
            for lbl in LABELS
        ])
        st.table(f1_df)

        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABELS], fontsize=9)
        ax.set_yticklabels([LABEL_DISPLAY[l] for l in LABELS], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(
                    j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=10, fontweight="600",
                )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


def _load_example(text):
    """Example button callback — put text into the text area and set auto-predict flag."""
    st.session_state.input_textarea = text
    st.session_state.auto_predict = True


# ------------------- Main app -------------------
def main():
    # Hero header
    st.markdown(
        """
        <div class="hero">
            <h1>📩 Indian Spam Detector</h1>
            <p>Paste any SMS or message — instantly find out if it's Normal, Spam, or Scam.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model check
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please run `python train.py` first.")
        st.stop()

    # Session state init
    if "input_textarea" not in st.session_state:
        st.session_state.input_textarea = ""
    if "auto_predict" not in st.session_state:
        st.session_state.auto_predict = False

    # Example buttons
    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.button("✅  Normal", on_click=_load_example, args=(EXAMPLES["normal"],), use_container_width=True)
    c2.button("⚠️  Spam",   on_click=_load_example, args=(EXAMPLES["spam"],),   use_container_width=True)
    c3.button("🚫  Scam",   on_click=_load_example, args=(EXAMPLES["scam"],),   use_container_width=True)

    # Input box
    st.markdown('<div class="section-label">Your message</div>', unsafe_allow_html=True)
    text = st.text_area(
        "Message",
        height=150,
        key="input_textarea",
        placeholder="Type or paste the message you want to check...",
        label_visibility="collapsed",
    )

    # Predict
    predict_clicked = st.button("Analyze Message", type="primary", use_container_width=True)

    if predict_clicked or st.session_state.auto_predict:
        st.session_state.auto_predict = False
        if not text.strip():
            st.warning("Please paste a message first.")
        else:
            show_result(text)

    # Model info expander
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("📊 About the model"):
        render_model_info()

    # Disclaimer
    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ This model is not 100% accurate. Always be careful before clicking any link.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()