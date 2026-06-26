# User Manual — Malaysia Skincare Sentiment Analysis System

This is the deployment component of the TNL6323 Sentiment Analysis project
(Product / E-commerce Reviews — Malaysia Context, Shopee skincare reviews).

It is a Streamlit web app with two pages:

1. **🔮 Predict Sentiment** — type or paste a new review and get an instant
   positive / neutral / negative prediction from the fine-tuned DistilBERT model.
2. **📊 Dashboard** — visual summary of the full Shopee cushion + moisturizer
   dataset, with auto-generated business insights.

---

## 1. Folder structure

```
deployment_app/
├── app.py                                # main entry point
├── page_predict.py                       # Predict Sentiment page
├── page_dashboard.py                     # Dashboard page
├── utils/
│   ├── model_loader.py                   # loads DistilBERT model + tokenizer
│   ├── data_loader.py                    # loads the dataset
│   └── insights.py                       # keyword extraction + business insights
├── requirements.txt
├── skintific_preprocessed.csv            # dataset used by the Dashboard
└── distilbert_skintific_sentiment_model/ # << YOU add this (see Step 3 below)
```

## 2. Install requirements

Requires Python 3.9+.

```bash
cd deployment_app
pip install -r requirements.txt
```

(If you only want to explore the Dashboard without the model, you can skip
installing `torch`/`transformers` — the app will still run, the Predict page
will just show a setup message instead of crashing.)

## 3. Add the trained model

Train the model using the **Knowledge Representation + Model Training**
notebook on Google Colab (Step 9 of that notebook saves and zips the model).

After training, download `distilbert_skintific_sentiment_model.zip` from
Colab and unzip it **inside this `deployment_app` folder**, so you end up with:

```
deployment_app/distilbert_skintific_sentiment_model/
    config.json
    model.safetensors  (or pytorch_model.bin)
    tokenizer_config.json
    vocab.txt
    ...
```

If you saved the model elsewhere, set an environment variable instead of moving it:

```bash
export SENTIMENT_MODEL_DIR="/path/to/distilbert_skintific_sentiment_model"
```

## 4. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser automatically (usually `http://localhost:8501`).
If it doesn't open automatically, copy that URL into your browser manually.

## 5. Using the app

### Predict Sentiment page
- Type a review in the text box, or click one of the **Sample** buttons to
  auto-fill an example.
- Click **Predict Sentiment**.
- The predicted label is shown with a confidence percentage, plus a bar chart
  showing the model's probability for all three classes.

### Dashboard page
- **KPI row** — total reviews, % positive/neutral/negative, average star rating.
- **Sentiment charts** — overall split (donut chart) and a breakdown by product
  (cushion vs moisturizer).
- **Product comparison** — side-by-side positive-rate comparison.
- **Keyword themes** — most frequent words in positive reviews vs
  negative/neutral reviews (computed live from the dataset).
- **Business Insights** — plain-language insights generated automatically from
  the current data (e.g. which product has the stronger sentiment profile,
  what share of complaints are service/fulfillment related rather than about
  the product itself).
- **Explore raw reviews** — filterable table to inspect individual reviews by
  product and sentiment.

## 6. Using a different dataset

To point the Dashboard at a different CSV (e.g. an updated/larger dataset),
either replace `skintific_preprocessed.csv` in this folder with a file that has
the same column structure (`product`, `stars`, `sentiment`, `clean_text`,
`processed_text`, ...), or set:

```bash
export SENTIMENT_DATA_PATH="/path/to/your_dataset.csv"
```

## 7. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| "Model folder not found" warning on Predict page | Model not trained/copied yet | Follow Step 3 above |
| Dashboard shows "Dataset file not found" | CSV missing from app folder | Copy `skintific_preprocessed.csv` into `deployment_app/`, or set `SENTIMENT_DATA_PATH` |
| `streamlit: command not found` | Streamlit not installed / not on PATH | `pip install -r requirements.txt`, then retry, or run `python -m streamlit run app.py` |
| App is slow on first prediction | Model is loading into memory (cached after first run) | Normal — subsequent predictions are fast |
