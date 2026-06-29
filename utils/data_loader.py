

import os
import streamlit as st
import pandas as pd

DATA_PATH = os.environ.get("SENTIMENT_DATA_PATH", "skintific_preprocessed.csv")


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(
            f"Dataset file '{path}' was not found. Place `skintific_preprocessed.csv` "
            "in the app directory, or set the `SENTIMENT_DATA_PATH` environment variable."
        )
        return pd.DataFrame(
            columns=["product", "author", "stars", "sentiment", "tags",
                     "free_text", "clean_text", "processed_text", "tokens_lemma"]
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=["clean_text"]).copy()
    df = df[df["clean_text"].astype(str).str.strip() != ""]
    df["processed_text"] = df["processed_text"].fillna("")
    df["sentiment"] = df["sentiment"].str.lower()
    return df.reset_index(drop=True)
