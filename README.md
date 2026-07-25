# Customer Sentiment Analysis — Twitter US Airline Sentiment

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-VADER-8A2BE2)](https://www.nltk.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](#live-demo)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end sentiment classification case study — five modeling approaches compared on real customer feedback, with brand benchmarking, root-cause complaint analysis, and a live dashboard where anyone can type a review and get an instant prediction.

## Live Demo

🔗 **[Try the live dashboard](https://sentiment-analysis-ytcvdn7yhftrexed7yoczj.streamlit.app/)** 
## Overview

Businesses with any public-facing customer channel generate a constant stream of feedback they rarely have time to systematically read. This project builds a pipeline that automatically classifies sentiment, identifies *why* customers are unhappy (not just that they are), and benchmarks brands against each other — the kind of analysis a business would otherwise pay a social-listening tool a monthly subscription for.

The notebook answers five business questions:

1. What's the overall sentiment split, and does a simple rule-based approach (no training data needed) get close to a trained model?
2. Which machine learning approach classifies sentiment most accurately?
3. What specific words drive negative vs. positive sentiment?
4. Which brand is underperforming its peers, and why?
5. What's the single biggest driver of customer complaints?

## Repository Structure

```
sentiment-analysis-customer-feedback/
├── app.py                              # Live Streamlit dashboard
├── pipeline.py                         # Standalone script (metrics/figures)
├── Sentiment_Analysis_Airlines.ipynb   # Full analysis notebook (15 modules, executed)
├── requirements.txt
├── runtime.txt                         # Pinned Python version for clean cloud deploys
├── data/
│   └── Tweets.csv                      # 14,640 labeled tweets, 6 airlines
├── outputs/                            # Saved figures & metrics.json
├── LICENSE
└── README.md
```

## Dataset

Public **Twitter US Airline Sentiment** dataset — 14,640 tweets about six major US airlines (United, US Airways, American, Southwest, Delta, Virgin America), collected February 2015, each hand-labeled positive, neutral, or negative, with negative tweets further labeled by *reason* (customer service, late flight, lost luggage, etc.). No sampling required — the full dataset is 3.4MB, well within GitHub's limits.

| Column | Description |
|---|---|
| `text` | Raw tweet text |
| `airline` | Which airline the tweet is about |
| `airline_sentiment` | Target: positive / neutral / negative |
| `negativereason` | Category of complaint (negative tweets only) |

## Notebook Structure (15 Modules)

| Module | Contents |
|---|---|
| 1 | Business Understanding |
| 2 | Data Understanding |
| 3 | Data Quality Assessment |
| 4 | Text Cleaning — including negation-aware tokenization |
| 5 | Business EDA — sentiment split, by-airline breakdown, complaint reasons |
| 6 | VADER — rule-based baseline (zero training) |
| 7 | Train/Test Split & TF-IDF Vectorization |
| 8 | Naive Bayes |
| 9 | Logistic Regression (class-weighted) |
| 10 | Random Forest (class-weighted) |
| 11 | XGBoost |
| 12 | Model Evaluation — Macro F1, confusion matrix |
| 13 | What Words Actually Drive Sentiment |
| 14 | Business Insights Summary |
| 15 | Business Recommendations & Limitations |

## Key Results

**Model comparison (3-class: negative / neutral / positive, stratified holdout)**

| Model | Accuracy | Macro F1 |
|---|---|---|
| VADER (rule-based, zero training) | 54.7% | 0.513 |
| Naive Bayes (TF-IDF) | 73.1% | 0.579 |
| **Logistic Regression (TF-IDF, class-weighted)** | 75.5% | **0.714** |
| Random Forest (TF-IDF, class-weighted) | 67.7% | 0.648 |
| XGBoost (TF-IDF) | 75.0% | 0.647 |

> **Macro F1 (not accuracy) is the primary metric** — it weighs all three classes equally, so a model that's great at spotting negative tweets but weak on neutral/positive ones (the majority class here is negative, ~63%) can't hide behind a high accuracy score. Logistic Regression wins clearly on Macro F1 despite XGBoost posting a similar raw accuracy — evidence the LR model handles the harder minority classes (neutral, positive) meaningfully better.

**Does a free, untrained tool get you most of the way there?** No — VADER's rule-based approach reaches only 0.513 Macro F1 versus 0.714 for the trained Logistic Regression model, a genuine ~39% relative improvement that justifies the cost of building a trained pipeline rather than defaulting to an off-the-shelf lexicon tool.

**Brand benchmarking:** US Airways had the highest negative-sentiment share (77.8%) versus Virgin America's 36.0% — a 42-point gap between the best and worst-performing airline in the same dataset, illustrating why sentiment is most actionable compared against peers, not viewed in isolation.

**Root cause:** "Customer Service Issue" was the single largest driver of negative sentiment, responsible for 31.8% of all negative tweets — ahead of operational factors like delays or cancellations, meaning it's a controllable, trainable factor rather than a purely operational one.

**Top words driving sentiment:** negative — *hours, worst, hold, delayed, rude, cancelled*; positive — *great, thank, awesome, love, amazing* — exactly the kind of concrete, quotable detail a client wants in a report.


### A note on negation handling

An early version of the text-cleaning step silently destroyed negation — `"wouldn't"` became two meaningless fragments (`"wouldn"`, `"t"`) after stripping punctuation, so the model sometimes couldn't distinguish `"not good"` from `"good"`. This is now fixed: contractions are expanded first (`"wouldn't"` → `"would not"`), then a negation word is merged with the word right after it into a single token (`"not good"` → `"not_good"`), letting the model learn negated phrases as their own feature. This measurably fixes common patterns like *"not bad"* and *"no issues"* — but rare negated phrases that appear only a handful of times in the training data (e.g. *"wouldn't recommend"*) still don't get enough signal to be learned reliably with this small a training set. That's an honest limitation of TF-IDF on ~14K examples, not a bug — a larger dataset or a sequence model (LSTM/transformer) would close this gap further.

## How to Run

### Notebook
```bash
git clone https://github.com/NurMithu/sentiment-analysis-customer-feedback.git
cd sentiment-analysis-customer-feedback
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Sentiment_Analysis_Airlines.ipynb
```

### Live Dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```


## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `nltk` (VADER) · `matplotlib` · `seaborn` · `plotly` · `streamlit` · `Jupyter`

## Business Recommendations (Summary)

- **Deploy the trained Logistic Regression model for ongoing social listening** — it meaningfully outperforms the zero-training-cost VADER baseline, which is worth confirming empirically before investing in a trained pipeline, not assuming.
- **Customer service response quality is the single largest lever** — the top driver of negative sentiment by a wide margin, ahead of operational issues, meaning it's a controllable, trainable factor.
- **Benchmark competitors, not just yourself** — the gap between the best- and worst-performing airline here is enormous; sentiment is far more actionable compared against peers.
- **Use the extracted top words as a lightweight, ongoing "what's changing" signal** — a sudden new negative word entering the top list is often an early warning of an emerging issue.

Full detail in Module 15 of the notebook.

## Limitations & Future Work

- Neutral-class detection is the hardest of the three classes (visible in the confusion matrix) — neutral tweets are often genuinely ambiguous even to a human reader, a known challenge in sentiment analysis broadly.
- This is 2015 airline-specific data; sentiment vocabulary and slang shift over time and across industries — a production system should be retrained periodically on fresh, in-domain data.
- TF-IDF treats words independently of order/context beyond bigrams; a transformer-based embedding model (e.g., DistilBERT) would likely improve accuracy further, at higher compute cost — a natural next iteration.
- No hyperparameter search was run — parameters used are reasonable defaults, not tuned.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Built as an end-to-end analytics case study — from raw social media text to a validated, business-ready sentiment classification system with a live interactive demo.*
