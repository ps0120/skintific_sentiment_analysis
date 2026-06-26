"""
Loads the fine-tuned DistilBERT sentiment model + tokenizer.

Model is hosted on Hugging Face Hub and downloaded automatically on first run.
Set the environment variable SENTIMENT_HF_REPO to your HuggingFace repo ID, e.g.:
    SENTIMENT_HF_REPO=your-username/distilbert-skintific-sentiment

Local override: set SENTIMENT_MODEL_DIR to a local folder path to skip HF download
(useful for local development if you still have the folder).
"""

import os
import streamlit as st
import torch

# ── Configuration ──────────────────────────────────────────────────────────────
# Priority 1: explicit local folder (for local dev)
LOCAL_MODEL_DIR = os.environ.get("SENTIMENT_MODEL_DIR", "")

# Priority 2: HuggingFace repo ID  ← set this in Streamlit Cloud secrets
HF_REPO = os.environ.get("SENTIMENT_HF_REPO", "")

ID2LABEL_FALLBACK = {0: "negative", 1: "neutral", 2: "positive"}

# used only for display in page_predict.py
MODEL_DIR = LOCAL_MODEL_DIR or HF_REPO or "distilbert_skintific_sentiment_model"


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_model_and_tokenizer():
    """
    Returns (model, tokenizer, error_message).
    Tries local folder first, then Hugging Face Hub.
    error_message is None on success.
    """
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    # ── Option A: local folder ────────────────────────────────────────────────
    if LOCAL_MODEL_DIR and os.path.isdir(LOCAL_MODEL_DIR):
        source = LOCAL_MODEL_DIR
    # ── Option B: Hugging Face Hub ────────────────────────────────────────────
    elif HF_REPO:
        source = HF_REPO   # from_pretrained accepts "username/repo-name" directly
    else:
        return None, None, (
            "No model source configured.\n\n"
            "**Option A — Hugging Face Hub (recommended for Streamlit Cloud):**\n"
            "1. Upload your model folder to https://huggingface.co/new (create a new model repo).\n"
            "2. In Streamlit Cloud → App settings → Secrets, add:\n"
            "   ```\n"
            "   SENTIMENT_HF_REPO = \"your-username/your-repo-name\"\n"
            "   ```\n\n"
            "**Option B — local folder:**\n"
            "Set the environment variable `SENTIMENT_MODEL_DIR` to your local model folder path."
        )

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(source)
        model = DistilBertForSequenceClassification.from_pretrained(source)
        model.eval()
        return model, tokenizer, None
    except Exception as exc:  # noqa: BLE001
        return None, None, f"Failed to load model from `{source}`:\n\n{exc}"


def predict_sentiment(text, model, tokenizer, max_len=96):
    """
    Runs inference on a single review string.
    Returns (predicted_label, probability_dict).
    """
    id2label = getattr(model.config, "id2label", None) or ID2LABEL_FALLBACK
    id2label = {int(k): v for k, v in id2label.items()}

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    pred_id = int(probs.argmax())
    label = id2label[pred_id]
    prob_dict = {id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}
    return label, prob_dict
