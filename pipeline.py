"""
Customer Sentiment Analysis — Twitter US Airline Sentiment Case Study
Pipeline script — produces the metrics and figures referenced in the README
and mirrors the notebook exactly (deterministic, random_state=42 throughout).
"""
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
nltk.download("vader_lexicon", quiet=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "Tweets.csv")
df = df[["airline_sentiment", "airline", "negativereason", "text"]].copy()

# ---------------------------------------------------------------------------
# 2. Data quality
# ---------------------------------------------------------------------------
df = df.dropna(subset=["text", "airline_sentiment"]).copy()
df = df.drop_duplicates(subset="text")

# ---------------------------------------------------------------------------
# 3. Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    # Expand contractions BEFORE stripping punctuation, so negation survives.
    # The old approach ("wouldn't" -> "wouldn t") silently destroyed negation,
    # which is a well-known failure mode for bag-of-words sentiment models.
    contractions = {
        "won't": "will not", "can't": "can not", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am",
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Merge a negation word with the word right after it into one token:
    # "not good" -> "not_good". This lets the model learn negated phrases as
    # their own feature instead of seeing "good" alone and predicting positive.
    text = re.sub(r"\b(not|no|never)\s+(\w+)", r"\1_\2", text)
    return text


df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 0]

# ---------------------------------------------------------------------------
# 4. Business EDA
# ---------------------------------------------------------------------------
sentiment_counts = df["airline_sentiment"].value_counts()

sentiment_by_airline = pd.crosstab(df["airline"], df["airline_sentiment"], normalize="index")
worst_airline = sentiment_by_airline["negative"].idxmax()
best_airline = sentiment_by_airline["negative"].idxmin()

top_negative_reasons = df["negativereason"].value_counts().head(8)
top_complaint = top_negative_reasons.index[0]
top_pct = top_negative_reasons.iloc[0] / (df["airline_sentiment"] == "negative").sum() * 100

# ---------------------------------------------------------------------------
# 5. VADER baseline (rule-based, no training)
# ---------------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()


def vader_label(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


df["vader_pred"] = df["text"].apply(vader_label)

# ---------------------------------------------------------------------------
# 6. Train / test split + TF-IDF
# ---------------------------------------------------------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"], df["airline_sentiment"], test_size=0.25,
    random_state=RANDOM_STATE, stratify=df["airline_sentiment"],
)
vader_test = df.loc[X_test_text.index, "vader_pred"]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, stop_words="english")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------------------------------------------------------------------------
# 7. Models
# ---------------------------------------------------------------------------
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)

log_reg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
log_reg.fit(X_train, y_train)
lr_preds = log_reg.predict(X_test)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=30, class_weight="balanced_subsample",
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

label_map = {"negative": 0, "neutral": 1, "positive": 2}
inv_label_map = {v: k for k, v in label_map.items()}
y_train_enc = y_train.map(label_map)
y_test_enc = y_test.map(label_map)

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.85, colsample_bytree=0.85, objective="multi:softmax",
    num_class=3, random_state=RANDOM_STATE, n_jobs=-1,
)
xgb.fit(X_train, y_train_enc)
xgb_preds = pd.Series(xgb.predict(X_test)).map(inv_label_map).values

# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Macro_F1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "Macro_Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Macro_Recall": round(recall_score(y_true, y_pred, average="macro"), 4),
    }


results = [
    evaluate(y_test, vader_test, "VADER (rule-based, no training)"),
    evaluate(y_test, nb_preds, "Naive Bayes (TF-IDF)"),
    evaluate(y_test, lr_preds, "Logistic Regression (TF-IDF, class-weighted)"),
    evaluate(y_test, rf_preds, "Random Forest (TF-IDF, class-weighted)"),
    evaluate(y_test, xgb_preds, "XGBoost (TF-IDF)"),
]
best_model_name = max(results, key=lambda r: r["Macro_F1"])["model"]

best_preds_map = {
    "VADER (rule-based, no training)": vader_test,
    "Naive Bayes (TF-IDF)": nb_preds,
    "Logistic Regression (TF-IDF, class-weighted)": lr_preds,
    "Random Forest (TF-IDF, class-weighted)": rf_preds,
    "XGBoost (TF-IDF)": xgb_preds,
}
best_preds = best_preds_map[best_model_name]

# ---------------------------------------------------------------------------
# 9. Top words
# ---------------------------------------------------------------------------
feature_names = np.array(vectorizer.get_feature_names_out())
neg_idx = list(log_reg.classes_).index("negative")
pos_idx = list(log_reg.classes_).index("positive")
top_neg_words = feature_names[np.argsort(log_reg.coef_[neg_idx])[-15:]][::-1].tolist()
top_pos_words = feature_names[np.argsort(log_reg.coef_[pos_idx])[-15:]][::-1].tolist()

# ---------------------------------------------------------------------------
# 10. Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "n_tweets": int(len(df)),
    "sentiment_distribution_pct": (sentiment_counts / len(df) * 100).round(1).to_dict(),
    "results": results,
    "best_model": best_model_name,
    "top_negative_words": top_neg_words,
    "top_positive_words": top_pos_words,
    "worst_airline": worst_airline,
    "worst_airline_negative_pct": round(sentiment_by_airline.loc[worst_airline, "negative"] * 100, 1),
    "best_airline": best_airline,
    "best_airline_negative_pct": round(sentiment_by_airline.loc[best_airline, "negative"] * 100, 1),
    "top_complaint_reason": top_complaint,
    "top_complaint_pct_of_negative": round(top_pct, 1),
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2, default=str))

# ---------------------------------------------------------------------------
# 11. Figures
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
sentiment_counts.plot(kind="bar", ax=ax, color=["#DC2626", "#94A3B8", "#059669"])
ax.set_title("Overall Sentiment Distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "sentiment_distribution.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
sentiment_by_airline[["negative", "neutral", "positive"]].sort_values("negative").plot(
    kind="barh", stacked=True, ax=ax, color=["#DC2626", "#94A3B8", "#059669"]
)
ax.set_title("Sentiment Share by Airline")
ax.set_xlabel("Share of Tweets")
plt.tight_layout()
plt.savefig(OUT_DIR / "sentiment_by_airline.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
top_negative_reasons.sort_values().plot(kind="barh", ax=ax, color="#DC2626")
ax.set_title("Top Reasons for Negative Sentiment")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_negative_reasons.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
results_sorted = sorted(results, key=lambda r: r["Macro_F1"])
colors = ["#94A3B8" if r["model"] != best_model_name else "#059669" for r in results_sorted]
ax.barh([r["model"] for r in results_sorted], [r["Macro_F1"] for r in results_sorted], color=colors)
ax.set_xlabel("Macro F1")
ax.set_title("Model Comparison — Macro F1 (3-class: neg/neu/pos)")
plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison.png", dpi=140)
plt.close()

labels_order = ["negative", "neutral", "positive"]
cm = confusion_matrix(y_test, best_preds, labels=labels_order)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=labels_order, yticklabels=labels_order)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=140)
plt.close()

print("\nSaved figures to", OUT_DIR)
