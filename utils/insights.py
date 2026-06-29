
from collections import Counter
import pandas as pd

# Words that are frequent but carry no business meaning for a skincare seller
GENERIC_FILLER = {
    "try", "well", "make", "like", "look", "take", "thing", "get",
    "really", "also", "quite", "little", "use", "buy", "order",
    "skin", "good", "face", "condition",
}


def top_keywords(texts: pd.Series, top_n: int = 12, exclude: set | None = None) -> list[tuple[str, int]]:
    #Word-frequency count over a series of already-lemmatized, stopword-removed text."""
    exclude = exclude or set()
    words = " ".join(texts.dropna().astype(str)).split()
    words = [w for w in words if w not in exclude]
    return Counter(words).most_common(top_n)


def compute_insights(df: pd.DataFrame) -> dict:

    total = len(df)
    sentiment_counts = df["sentiment"].value_counts()
    sentiment_pct = (sentiment_counts / total * 100).round(1)

    products = sorted(df["product"].dropna().unique().tolist())

    per_product = {}
    for p in products:
        sub = df[df["product"] == p]
        counts = sub["sentiment"].value_counts()
        pct = (counts / len(sub) * 100).round(1)
        per_product[p] = {
            "n": len(sub),
            "pct_positive": float(pct.get("positive", 0.0)),
            "pct_neutral": float(pct.get("neutral", 0.0)),
            "pct_negative": float(pct.get("negative", 0.0)),
            "avg_stars": round(float(sub["stars"].mean()), 2) if "stars" in sub else None,
            "top_positive_keywords": top_keywords(
                sub.loc[sub["sentiment"] == "positive", "processed_text"],
                top_n=10, exclude=GENERIC_FILLER,
            ),
            "top_negative_keywords": top_keywords(
                sub.loc[sub["sentiment"].isin(["negative", "neutral"]), "processed_text"],
                top_n=10, exclude=GENERIC_FILLER,
            ),
        }

    overall_top_positive = top_keywords(
        df.loc[df["sentiment"] == "positive", "processed_text"], top_n=12, exclude=GENERIC_FILLER
    )
    overall_top_negative = top_keywords(
        df.loc[df["sentiment"].isin(["negative", "neutral"]), "processed_text"],
        top_n=12, exclude=GENERIC_FILLER,
    )


    service_terms = {"service", "customer", "refund", "refill", "missing", "box",
                      "sticker", "wrong", "shade", "delivery", "courier", "seller"}
    neg_neu = df[df["sentiment"].isin(["negative", "neutral"])]
    neg_words = " ".join(neg_neu["processed_text"].dropna().astype(str)).split()
    service_mentions = sum(1 for w in neg_words if w in service_terms)
    service_review_count = neg_neu["processed_text"].apply(
        lambda t: any(w in service_terms for w in str(t).split())
    ).sum()
    pct_service_related = round(
        service_review_count / len(neg_neu) * 100, 1
    ) if len(neg_neu) else 0.0

    # Auto-generated insight bullets (plain text, computed — not hardcoded)
    insights = []

    insights.append(
        f"Out of {total:,} reviews analyzed, "
        f"{sentiment_pct.get('positive', 0):.1f}% are positive, "
        f"{sentiment_pct.get('neutral', 0):.1f}% neutral, and "
        f"{sentiment_pct.get('negative', 0):.1f}% negative — overall customer "

    )

    if len(products) >= 2:
        best = max(per_product.items(), key=lambda kv: kv[1]["pct_positive"])
        worst = min(per_product.items(), key=lambda kv: kv[1]["pct_positive"])
        if best[0] != worst[0]:
            insights.append(
                f"'{best[0].title()}' has the stronger sentiment profile "
                f"({best[1]['pct_positive']:.1f}% positive) compared to "
                f"'{worst[0].title()}' ({worst[1]['pct_positive']:.1f}% positive) — "
                f"if marketing budget is limited, '{best[0].title()}' is the safer product to push."
            )

    if overall_top_positive:
        kw_str = ", ".join(w for w, _ in overall_top_positive[:4])
        insights.append(
            f"The most common themes in positive reviews are: {kw_str}. "
            f"These are the selling points which can be highlighting in product listings and ads."
        )

    if pct_service_related > 0:
        insights.append(
            f"Among negative/neutral reviews, {pct_service_related:.1f}% reviews mention customer-service issues (refunds, wrong shade, missing items packaging)"
            f" rather than the product itself. "
            f"Company need to fix these issues (order accuracy, packaging) to reduce some complaints "
            f"without changing the product."
        )

    if overall_top_negative:
        kw_str = ", ".join(w for w, _ in overall_top_negative[:4])
        insights.append(
            f"The most common themes in negative/neutral reviews are: {kw_str}. "
            f"These are the first areas to investigate for product or service improvement."
        )

    return {
        "total": total,
        "sentiment_counts": sentiment_counts,
        "sentiment_pct": sentiment_pct,
        "products": products,
        "per_product": per_product,
        "overall_top_positive": overall_top_positive,
        "overall_top_negative": overall_top_negative,
        "pct_service_related": pct_service_related,
        "insights": insights,
    }
