"""
Page 1: Predict Sentiment
Lets a user type/paste a new review and see the DistilBERT model's sentiment prediction.
"""

import streamlit as st
import plotly.graph_objects as go

from utils.model_loader import predict_sentiment

SENTIMENT_COLORS = {"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}
SENTIMENT_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😞"}

SAMPLE_REVIEWS = [
    "this product is amazing, my skin feels so smooth and hydrated after just one week!",
    "make skin get ance and oil.",
    "terrible, the cushion broke me out badly and the customer service never replied to my refund request",
]


def render(model, tokenizer, model_load_error, model_dir):
    st.header("🔮 Predict Review Sentiment")
    st.caption(
        "Enter a new skincare product review below. The fine-tuned DistilBERT model "
        "will classify it as **positive**, **neutral**, or **negative**."
    )

    if model_load_error:
        st.warning(model_load_error, icon="⚠️")
        st.info(
            "You can still explore the **Dashboard** page while the model is unavailable.",
            icon="ℹ️",
        )
        return

    with st.expander("💡 Try a sample review"):
        cols = st.columns(3)
        for i, sample in enumerate(SAMPLE_REVIEWS):
            if cols[i].button(f"Sample {i + 1}", use_container_width=True, key=f"sample_{i}"):
                st.session_state["review_text"] = sample

    review_text = st.text_area(
        "Review text",
        key="review_text",
        height=140,
        placeholder="e.g. 'love this moisturizer, my skin feels so much softer and the packaging is super cute!'",
    )

    predict_clicked = st.button("Predict Sentiment", type="primary", use_container_width=True)

    if predict_clicked:
        if not review_text or not review_text.strip():
            st.error("Please enter a review first.")
            return

        with st.spinner("Running inference..."):
            label, probs = predict_sentiment(review_text, model, tokenizer)

        color = SENTIMENT_COLORS.get(label, "#3498db")
        emoji = SENTIMENT_EMOJI.get(label, "")

        st.markdown(
            f"""
            <div style="padding:1.2rem;border-radius:12px;background-color:{color}22;
                        border:2px solid {color};text-align:center;margin-bottom:1rem;">
                <span style="font-size:2.2rem;">{emoji}</span><br>
                <span style="font-size:1.6rem;font-weight:700;color:{color};
                             text-transform:uppercase;letter-spacing:1px;">
                    {label}
                </span><br>
                <span style="font-size:0.95rem;color:#555;">
                    Confidence: {probs[label] * 100:.1f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        order = ["negative", "neutral", "positive"]
        fig = go.Figure(
            go.Bar(
                x=[probs.get(k, 0) * 100 for k in order],
                y=[k.title() for k in order],
                orientation="h",
                marker_color=[SENTIMENT_COLORS[k] for k in order],
                text=[f"{probs.get(k, 0) * 100:.1f}%" for k in order],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Prediction Confidence by Class",
            xaxis_title="Probability (%)",
            xaxis_range=[0, 105],
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
