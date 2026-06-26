"""
TNL6323 Natural Language Processing — Project
Malaysia-Focused Sentiment Analysis System (E-commerce / Skincare Domain)

Streamlit deployment app with two pages:
  1. Predict Sentiment  — enter a new review, get DistilBERT sentiment prediction
  2. Dashboard          — summarizes the Shopee skincare dataset + business insights

Run locally with:
    streamlit run app.py
"""

import streamlit as st

from utils.data_loader import load_dataset
from utils.model_loader import load_model_and_tokenizer, MODEL_DIR
import page_predict
import page_dashboard

st.set_page_config(
    page_title="Malaysia Skincare Sentiment Analysis",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧴 Sentiment Analysis")
    st.caption("Shopee Malaysia Skincare Reviews — Cushion & Moisturizer")

    page = st.radio(
        "Navigate",
        options=["🔮 Predict Sentiment", "📊 Dashboard"],
        index=0,
    )

    st.divider()
    st.caption(
        "TNL6323 Natural Language Processing\n\n"
        "Malaysia-Focused Sentiment Analysis System\n\n"
        "Domain: Product / E-commerce Reviews (Malaysia Context)"
    )

# ---------------------------------------------------------------------------
# Shared cached resources
# ---------------------------------------------------------------------------
df = load_dataset()

# ---------------------------------------------------------------------------
# Route to the selected page
# ---------------------------------------------------------------------------
if page == "🔮 Predict Sentiment":
    model, tokenizer, model_load_error = load_model_and_tokenizer()
    page_predict.render(model, tokenizer, model_load_error, MODEL_DIR)
else:
    page_dashboard.render(df)
