
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.insights import compute_insights

SENTIMENT_COLORS = {"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def render(df: pd.DataFrame):
    st.header("📊 Shopee Skincare Reviews Dashboard")

    if df.empty:
        st.error("No data loaded — check that `skintific_preprocessed.csv` is available to the app.")
        return

    stats = compute_insights(df)

#Dataset reviews
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Reviews", f"{stats['total']:,}")
    k2.metric("% Positive", f"{stats['sentiment_pct'].get('positive', 0):.1f}%")
    k3.metric("% Neutral", f"{stats['sentiment_pct'].get('neutral', 0):.1f}%")
    k4.metric("% Negative", f"{stats['sentiment_pct'].get('negative', 0):.1f}%")
    avg_stars = round(df["stars"].mean(), 2) if "stars" in df else None
    k5.metric("Avg. Star Rating", f"{avg_stars} ⭐" if avg_stars else "N/A")

    st.divider()

    # Sentiment distribution: overall + by product
    c1, c2 = st.columns([1, 1.4])

    with c1:
        st.subheader("Overall Sentiment Split")
        counts = stats["sentiment_counts"].reindex(SENTIMENT_ORDER).fillna(0)
        fig = go.Figure(
            go.Pie(
                labels=[s.title() for s in SENTIMENT_ORDER],
                values=counts.values,
                marker_colors=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
                hole=0.45,
            )
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Sentiment by Product")
        prod_sent = (
            df.groupby(["product", "sentiment"]).size().reset_index(name="count")
        )
        prod_sent["sentiment"] = prod_sent["sentiment"].astype(str)
        fig = px.bar(
            prod_sent,
            x="product",
            y="count",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            category_orders={"sentiment": SENTIMENT_ORDER},
            barmode="stack",
            labels={"product": "Product", "count": "Number of Reviews", "sentiment": "Sentiment"},
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()


    st.subheader("Product Comparison")
    prod_cols = st.columns(len(stats["per_product"]))
    for col, (product, info) in zip(prod_cols, stats["per_product"].items()):
        with col:
            st.markdown(f"**{product.title()}** ({info['n']:,} reviews)")
            st.progress(info["pct_positive"] / 100, text=f"{info['pct_positive']:.1f}% positive")
            st.caption(
                f"Neutral: {info['pct_neutral']:.1f}% · Negative: {info['pct_negative']:.1f}% "
                f"· Avg stars: {info['avg_stars']}⭐"
            )

    st.divider()


    st.subheader("What Customers Talk About")
    kw1, kw2 = st.columns(2)

    with kw1:
        st.markdown("**🟢 Top themes in positive reviews**")
        if stats["overall_top_positive"]:
            kw_df = pd.DataFrame(stats["overall_top_positive"], columns=["word", "count"])
            fig = px.bar(
                kw_df.sort_values("count"), x="count", y="word", orientation="h",
                color_discrete_sequence=[SENTIMENT_COLORS["positive"]],
            )
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Mentions", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with kw2:
        st.markdown("**🔴 Top themes in negative / neutral reviews**")
        if stats["overall_top_negative"]:
            kw_df = pd.DataFrame(stats["overall_top_negative"], columns=["word", "count"])
            fig = px.bar(
                kw_df.sort_values("count"), x="count", y="word", orientation="h",
                color_discrete_sequence=[SENTIMENT_COLORS["negative"]],
            )
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Mentions", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()


    st.subheader("💼 Business Insights")
    st.caption("Automatically generated from the current dataset.")
    for insight in stats["insights"]:
        st.info(insight, icon="💡")

    st.divider()

    
    with st.expander("🔍 Explore raw reviews"):
        f1, f2 = st.columns(2)
        product_filter = f1.multiselect(
            "Product", options=stats["products"], default=stats["products"]
        )
        sentiment_filter = f2.multiselect(
            "Sentiment", options=SENTIMENT_ORDER, default=SENTIMENT_ORDER
        )
        filtered = df[
            df["product"].isin(product_filter) & df["sentiment"].isin(sentiment_filter)
        ]
        st.caption(f"Showing {len(filtered):,} of {len(df):,} reviews")
        st.dataframe(
            filtered[["product", "stars", "sentiment", "clean_text"]].rename(
                columns={"clean_text": "review_text"}
            ),
            use_container_width=True,
            height=320,
        )
