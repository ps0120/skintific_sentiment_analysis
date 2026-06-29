
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

#sidebar
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
        "Shopee Skincare Sentiment Analysis System\n\n"
        "Domain: Product / E-commerce Reviews (Malaysia Context)"
    )


df = load_dataset()

# Route to the selected page
if page == "🔮 Predict Sentiment":
    model, tokenizer, model_load_error = load_model_and_tokenizer()
    page_predict.render(model, tokenizer, model_load_error, MODEL_DIR)
else:
    page_dashboard.render(df)
