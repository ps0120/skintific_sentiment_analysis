

import os
import streamlit as st
import torch


# 1: explicit local folder 
LOCAL_MODEL_DIR = os.environ.get("SENTIMENT_MODEL_DIR", "distilbert_skintific_sentiment_model")

# 2: HuggingFace repo ID 
HF_REPO = os.environ.get("SENTIMENT_HF_REPO", "")

ID2LABEL_FALLBACK = {0: "negative", 1: "neutral", 2: "positive"}

MODEL_DIR = LOCAL_MODEL_DIR or HF_REPO or "distilbert_skintific_sentiment_model"


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_model_and_tokenizer():

    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )


    if LOCAL_MODEL_DIR and os.path.isdir(LOCAL_MODEL_DIR):
        source = LOCAL_MODEL_DIR

    elif HF_REPO:
        source = HF_REPO   
    else:
        return None, None, (
            "No model source configured.\n\n"
        )

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(source)
        model = DistilBertForSequenceClassification.from_pretrained(source)
        model.eval()
        return model, tokenizer, None
    except Exception as exc:  
        return None, None, f"Failed to load model from `{source}`:\n\n{exc}"


def predict_sentiment(text, model, tokenizer, max_len=96):

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
