# Financial News Sentiment Analysis

Stack: Python, pandas, NumPy, matplotlib, scikit-learn, TensorFlow.

## Setup
1. `pip install -r requirements.txt`
2. Download Financial PhraseBank (`Sentences_AllAgree.txt`, or `Sentences_75Agree.txt`) into `data/`.
3. Smoke test: `python sentiment_market_signal.py --phrasebank data/Sentences_AllAgree.txt --demo`
   (synthetic prices and random dates, so correlations should be ~0; this only verifies the pipeline).
4. Real run: supply a dated news CSV (`date, headline[, ticker]`) and prices
   (CSV with date index + one close column per ticker, or let yfinance download them).

## Important data note
PhraseBank has **no dates or tickers**. It is used to train/evaluate the sentiment classifier.
For the returns analysis you need a separate dated news source (e.g. a Kaggle financial-headlines
dataset or an API such as Alpha Vantage News / Finnhub).

## Methodology
- Models: TF-IDF (1-2 grams) + Logistic Regression, a TensorFlow BiLSTM, and their ensemble; best macro-F1 is used.
- Sentiment score = P(positive) - P(negative); daily score = mean per company per trading day.
- Weekend/holiday news is mapped to the next trading day.
- Analysis: lagged Spearman correlations, tercile return buckets with Welch t-test, directional agreement.
- Lags < 0 act as a reverse-causality check (sentiment merely reacting to past returns).

## Limitations to state on your resume / in interviews
- Association, not prediction or causation; no transaction costs; not trading advice.
- Domain shift: PhraseBank sentences are cleaner than real headlines.
- Intraday timing: news after the close should map to the next day (handle with timestamps if available).
- Next steps: FinBERT embeddings, spaCy NER, market-adjusted (abnormal) returns, Granger tests, walk-forward validation.

## Resume line
**Financial News Sentiment & Market Signal Analytics using NLP** - Built a TF-IDF/BiLSTM sentiment
pipeline on Financial PhraseBank, extracted company entities, aggregated daily sentiment and analysed its
lagged association with stock returns (Spearman, tercile tests) in a matplotlib analytics dashboard.
