"""
Customer Sentiment Analysis Dashboard
Live companion to Sentiment_Analysis_Airlines.ipynb

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import re

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer Sentiment Dashboard", page_icon="💬", layout="wide")

NEG, NEU, POS = "#DC2626", "#94A3B8", "#059669"
COLOR_MAP = {"negative": NEG, "neutral": NEU, "positive": POS}

nltk.download("vader_lexicon", quiet=True)


@st.cache_data
def load_default():
    df = pd.read_csv("data/Tweets.csv")
    return df[["airline_sentiment", "airline", "negativereason", "text"]].copy()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data
def prepare(df: pd.DataFrame):
    df = df.dropna(subset=["text", "airline_sentiment"]).copy()
    df = df.drop_duplicates(subset="text")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["airline_sentiment"], test_size=0.25,
        random_state=42, stratify=df["airline_sentiment"],
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Macro_F1": f1_score(y_test, preds, average="macro"),
    }
    return model, vectorizer, metrics, X_test_text, y_test, preds, probs


# ---------------------------------------------------------------------------
st.sidebar.title("💬 Sentiment Analysis")
st.sidebar.caption("Customer feedback classification — live model")

data_mode = st.sidebar.radio("Data source", ["Demo dataset (Airline Tweets)", "Upload my own"])
if data_mode == "Upload my own":
    uploaded = st.sidebar.file_uploader(
        "CSV with a 'text' column (+ optional 'airline_sentiment' labels)", type="csv"
    )
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        if "text" not in raw_df.columns:
            st.sidebar.error("Need a 'text' column. Showing demo data instead.")
            raw_df = load_default()
        elif "airline_sentiment" not in raw_df.columns:
            st.sidebar.warning("No sentiment labels found — can't train/evaluate without labels. Showing demo data.")
            raw_df = load_default()
        else:
            if "airline" not in raw_df.columns:
                raw_df["airline"] = "N/A"
            if "negativereason" not in raw_df.columns:
                raw_df["negativereason"] = np.nan
            st.sidebar.success("Your data is loaded ✅")
    else:
        st.sidebar.info("Upload a CSV to use your own data. Showing demo data meanwhile.")
        raw_df = load_default()
else:
    raw_df = load_default()
    st.sidebar.info("Showing the public Twitter US Airline Sentiment demo dataset.")

with st.spinner("Cleaning text and training model..."):
    df = prepare(raw_df)
    model, vectorizer, metrics, X_test_text, y_test, preds, probs = train_model(df)

st.sidebar.markdown("---")
st.sidebar.caption("[View the full notebook & methodology on GitHub](https://github.com/NurMithu)")

# ---------------------------------------------------------------------------
sentiment_counts = df["airline_sentiment"].value_counts()

st.title("Customer Sentiment Analysis Dashboard")
st.subheader("Twitter US Airline Sentiment — Live Model")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews Analyzed", f"{len(df):,}")
k2.metric("% Negative", f"{sentiment_counts.get('negative', 0) / len(df):.1%}")
k3.metric("Model Accuracy", f"{metrics['Accuracy']:.1%}")
k4.metric("Model Macro F1", f"{metrics['Macro_F1']:.3f}")

st.markdown("---")

tab_overview, tab_brands, tab_words, tab_try, tab_reco = st.tabs(
    ["📊 Overview", "🏢 Brand Comparison", "🔤 What Drives Sentiment", "✍️ Try It Live", "📋 Recommendations"]
)

# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            x=sentiment_counts.index, y=sentiment_counts.values,
            title="Overall Sentiment Distribution", color=sentiment_counts.index,
            color_discrete_map=COLOR_MAP,
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if df["negativereason"].notna().any():
            top_reasons = df["negativereason"].value_counts().head(8)
            fig = px.bar(
                y=top_reasons.index, x=top_reasons.values, orientation="h",
                title="Top Reasons for Negative Sentiment", color_discrete_sequence=[NEG],
            )
            fig.update_layout(yaxis_title="", xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No negative-reason labels in this dataset.")

    cm_labels = sorted(df["airline_sentiment"].unique())
    cm = confusion_matrix(y_test, preds, labels=cm_labels)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=[f"Predicted {l}" for l in cm_labels], y=[f"Actual {l}" for l in cm_labels],
        title="Confusion Matrix (holdout test set)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_brands:
    if "airline" in df.columns and df["airline"].nunique() > 1:
        sentiment_by_brand = pd.crosstab(df["airline"], df["airline_sentiment"], normalize="index")
        cols = [c for c in ["negative", "neutral", "positive"] if c in sentiment_by_brand.columns]
        sentiment_by_brand = sentiment_by_brand[cols].sort_values(cols[0] if "negative" in cols else cols[0])

        fig = px.bar(
            sentiment_by_brand, orientation="h", title="Sentiment Share by Brand/Airline",
            color_discrete_map=COLOR_MAP, barmode="stack",
        )
        fig.update_layout(xaxis_title="Share of Reviews", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        if "negative" in sentiment_by_brand.columns:
            worst = sentiment_by_brand["negative"].idxmax()
            best = sentiment_by_brand["negative"].idxmin()
            c1, c2 = st.columns(2)
            c1.metric("Needs the most attention", worst, f"{sentiment_by_brand.loc[worst, 'negative']:.0%} negative")
            c2.metric("Best performer", best, f"{sentiment_by_brand.loc[best, 'negative']:.0%} negative")
    else:
        st.info("Upload data with a brand/category column (e.g. 'airline') to compare brands side by side.")

# ---------------------------------------------------------------------------
with tab_words:
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(model.classes_)

    c1, c2 = st.columns(2)
    if "negative" in classes:
        neg_idx = classes.index("negative")
        top_neg = feature_names[np.argsort(model.coef_[neg_idx])[-15:]][::-1]
        with c1:
            st.markdown("**Words driving NEGATIVE sentiment**")
            fig = px.bar(y=top_neg[::-1], x=list(range(len(top_neg), 0, -1)), orientation="h",
                         color_discrete_sequence=[NEG])
            fig.update_layout(showlegend=False, xaxis_title="Relative strength", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    if "positive" in classes:
        pos_idx = classes.index("positive")
        top_pos = feature_names[np.argsort(model.coef_[pos_idx])[-15:]][::-1]
        with c2:
            st.markdown("**Words driving POSITIVE sentiment**")
            fig = px.bar(y=top_pos[::-1], x=list(range(len(top_pos), 0, -1)), orientation="h",
                         color_discrete_sequence=[POS])
            fig.update_layout(showlegend=False, xaxis_title="Relative strength", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_try:
    st.markdown("#### Type a review or tweet and see the live prediction")
    user_text = st.text_area(
        "Enter text", value="The flight was delayed for 3 hours and nobody told us why. Terrible service.",
        height=100,
    )
    if st.button("Analyze sentiment"):
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = dict(zip(model.classes_, model.predict_proba(vec)[0]))

        color = COLOR_MAP.get(pred, "#000000")
        st.markdown(f"### Prediction: <span style='color:{color}'>**{pred.upper()}**</span>", unsafe_allow_html=True)

        proba_df = pd.DataFrame({"Sentiment": list(proba.keys()), "Probability": list(proba.values())})
        fig = px.bar(proba_df, x="Sentiment", y="Probability", color="Sentiment",
                     color_discrete_map=COLOR_MAP, title="Confidence by Class")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        sia = SentimentIntensityAnalyzer()
        vader_score = sia.polarity_scores(user_text)["compound"]
        st.caption(f"For comparison, the rule-based VADER compound score is {vader_score:.2f} (range -1 to +1).")

# ---------------------------------------------------------------------------
with tab_reco:
    st.markdown("#### Business recommendations")
    neg_pct = sentiment_counts.get("negative", 0) / len(df)
    top_complaint = df["negativereason"].value_counts().idxmax() if df["negativereason"].notna().any() else None

    reco_text = f"""
- **Deploy this model for ongoing social listening** — it classifies sentiment
  at **{metrics['Accuracy']:.0%} accuracy** ({metrics['Macro_F1']:.2f} macro F1),
  meaningfully ahead of a zero-training rule-based baseline (see the
  notebook for the VADER comparison).
- **{neg_pct:.0%} of reviews in this dataset are negative** — worth treating
  as a baseline to track against over time, not just a one-time snapshot.
"""
    if top_complaint:
        reco_text += f"- **'{top_complaint}' is the top driver of negative sentiment** — the highest-leverage single area for improvement, ahead of other operational factors.\n"
    reco_text += """- **Use the extracted top words (previous tab) as an ongoing "what's changing" signal** — a new word entering the negative list often flags an emerging issue before it shows up in aggregate scores.
- **Benchmark against competitors/peers**, not just your own trend line — the "Brand Comparison" tab shows this is often more actionable than viewing your own sentiment in isolation.
"""
    st.markdown(reco_text)
    st.link_button("📓 View the full notebook & methodology on GitHub", "https://github.com/NurMithu")

st.markdown("---")
st.caption(
    "Live model trained on the public Twitter US Airline Sentiment dataset. "
    "Upload your own review/tweet data (with 'text' and 'airline_sentiment' columns) to see results on your own data."
)
