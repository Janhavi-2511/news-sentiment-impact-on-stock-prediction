# News Sentiment Impact on Stock Market Prediction

A hybrid NLP + machine learning pipeline that analyzes how financial news sentiment
influences next-day S&P 500 market direction, combining VADER sentiment analysis,
technical indicators, and an ARIMA-XGBoost model.

---

## What It Does

This project investigates whether the **sentiment of financial news headlines**
can predict whether the S&P 500 will go **up or down the next trading day**.

It goes beyond simple sentiment scoring — the pipeline integrates:
- NLP-based sentiment features from 50,000+ real financial headlines
- Classical technical indicators used by quantitative traders (RSI, MACD, Bollinger Bands)
- An ARIMA time-series model to capture the portion of market movement
  *not* explained by historical patterns, leaving the residual for XGBoost to learn from

---

## Dataset

| Source   | Type                      | Date Range  |
|----------|---------------------------|-------------|
| CNBC     | Financial news headlines  | 2016–2023   |
| Guardian | Financial news headlines  | 2016–2023   |
| Reuters  | Financial news headlines  | 2016–2023   |
| yfinance | S&P 500 (^GSPC) OHLCV     | 2016–2023   |

- ~50,000+ headlines collected across 3 sources
- Aligned to ~1,800 common trading days (weekends and market holidays excluded)

---

## Pipeline Overview

### 1. Data Collection & Cleaning
- Loaded and merged headlines from CNBC, Guardian, and Reuters
- Parsed mixed date formats, dropped unparseable rows
- Downloaded S&P 500 price data via `yfinance`
- Aligned news dates to market trading days only (intersection of both datasets)

### 2. Exploratory Data Analysis
- S&P 500 price trend with 50-day and 200-day moving averages
- Distribution of daily returns (approx. normal, centered near 0%)
- 30-day rolling volatility — spike visible during COVID-19 crash (March 2020)
- Daily headline volume over time with annotated news spikes
- Top 30 most frequent financial keywords + word cloud
- Average returns by day of week (Monday Effect confirmed)
- Source-wise headline count comparison

### 3. Sentiment Analysis (VADER)
- Applied VADER `SentimentIntensityAnalyzer` to every headline
- Computed daily aggregates: mean sentiment, std, headline count
- Engineered rolling sentiment features: 5-day MA, 10-day MA, 5-day std
- Visualized daily sentiment trend with 30-day smoothing

### 4. Feature Engineering (16 Features)

| Category              | Features                                                  |
|-----------------------|-----------------------------------------------------------|
| Sentiment (5)         | `sentiment_score`, `sentiment_ma5`, `sentiment_ma10`, `sentiment_std5`, `sentiment_std` |
| Lagged Returns (3)    | `return_lag1`, `return_lag2`, `return_lag3`               |
| Market Rolling (3)    | `return_ma10`, `volatility_10`, `log_headline_count`      |
| ARIMA Residual (1)    | `arima_residual`                                          |
| Technical (4)         | `rsi_14`, `macd_hist`, `bb_width`, `bb_position`          |

### 5. ARIMA Hybrid Component
- Fitted `auto_arima` on training returns to find best (p,d,q) order
- Generated walk-forward residuals for the full series (no data leakage)
- Residuals represent market movement *unexplained* by historical return patterns
- Model refitted every 60 days on the test set to stay current

### 6. Model Training (XGBoost Classifier)
- **Target**: binary — market Up (1) or Down (0) next day
- **Split**: 80% train / 20% test, time-ordered (no random shuffle)
- **Validation**: 5-fold `TimeSeriesSplit` with `GridSearchCV` (405 fits)
- **Hyperparameters tuned**: `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`

### 7. Evaluation
- Confusion matrix with per-class precision, recall, and F1
- Feature importance ranking (top 3 highlighted)
- PCA 2D projection of the 16-feature space (variance explained per component)
- Fold-by-fold cross-validation accuracy vs 50% baseline

---

## Tech Stack

| Tool              | Purpose                            |
|-------------------|------------------------------------|
| Python            | Core language                      |
| Pandas, NumPy     | Data manipulation                  |
| yfinance          | S&P 500 price data                 |
| vaderSentiment    | NLP sentiment scoring              |
| NLTK              | Stopword removal, tokenization     |
| pmdarima          | Auto ARIMA model selection         |
| XGBoost           | Gradient boosted classifier        |
| Scikit-learn      | Preprocessing, CV, metrics         |
| Matplotlib, Seaborn | Visualizations                   |
| WordCloud         | Keyword frequency visualization    |
| Google Colab      | Development environment            |

---

## Key Visualizations

- S&P 500 price with MA50/MA200 and Death Cross annotation
- Daily return distribution (near-normal with fat tails)
- 30-day rolling volatility (COVID spike clearly visible)
- Daily headline volume with annotated spikes
- VADER sentiment trend over time
- Sentiment vs next-day return scatterplot (Pearson + Spearman correlation)
- XGBoost confusion matrix with precision/recall
- Feature importance ranking
- PCA explained variance + 2D scatter (Up vs Down)

---

## How to Run

1. Clone the repo and open `news_sentiment_stock_prediction.ipynb` in Google Colab
2. Download the datasets from `https://www.kaggle.com/datasets/notlucasp/financial-news-headlines`
3. Upload the three headline CSVs (`cnbc_headlines.csv`, `guardian_headlines.csv`, `reuters_headlines.csv`) to your Google Drive under `dataset/`
4. Run all cells in order

> Note: The ARIMA fitting step takes ~30 seconds. GridSearchCV (405 fits) may take several minutes depending on hardware.

---
