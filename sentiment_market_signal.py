"""
Financial News Sentiment & Market Signal Analytics using NLP
=============================================================
Pipeline:
  Financial News -> Text Cleaning -> TF-IDF / BiLSTM embeddings
  -> Sentiment Classification -> Company Extraction
  -> Daily Sentiment Aggregation -> Comparison with Stock Returns -> Dashboard

Research framing: this investigates the *association* between news sentiment
and subsequent returns. It does NOT claim to predict prices.

Usage
-----
  # Pipeline smoke test (synthetic prices + random dates; NO real signal expected)
  python sentiment_market_signal.py --phrasebank data/Sentences_AllAgree.txt --demo

  # Real analysis: dated news CSV (columns: date, headline[, ticker]) + prices
  python sentiment_market_signal.py --phrasebank data/Sentences_AllAgree.txt \
      --news data/news.csv --prices data/prices.csv
  # (omit --prices to download via yfinance, if installed)
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import tensorflow as tf

SEED = 42
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Simple alias table for rule-based company extraction (extend as needed).
COMPANIES = {
    "NVDA": ["nvidia"], "AAPL": ["apple"], "MSFT": ["microsoft"],
    "TSLA": ["tesla"], "AMZN": ["amazon"], "GOOGL": ["google", "alphabet"],
    "META": ["meta platforms", "facebook"], "JPM": ["jpmorgan", "jp morgan"],
    "GS": ["goldman sachs"], "NFLX": ["netflix"],
}
_PATTERNS = {
    t: re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, n)), re.I)
    for t, n in COMPANIES.items()
}
_SYMBOLS = {t: re.compile(r"(?<![A-Za-z])\$?%s(?![A-Za-z])" % t) for t in COMPANIES}


# ----------------------------------------------------------------- 1. data
def _read_text(path):
    """Financial PhraseBank / Kaggle files are often NOT valid UTF-8 -> fall back to latin-1."""
    if not os.path.exists(path):
        raise SystemExit(f"Dataset not found: {path}\n(run from the project folder and check the data/ path)")
    raw = open(path, "rb").read()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue


def load_phrasebank(path):
    """Supports the original 'sentence@label' .txt and the Kaggle all-data.csv."""
    text = _read_text(path)
    if path.lower().endswith(".csv"):
        import io
        d = pd.read_csv(io.StringIO(text), header=None, names=["label", "text"])
        d["label"] = d.label.astype(str).str.strip().str.lower().map(L2I)
        d = d.dropna(subset=["label"]).astype({"label": int})
        return d[["text", "label"]].reset_index(drop=True)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if "@" in line:
            s, lab = line.rsplit("@", 1)
            if lab in L2I:
                rows.append((s, L2I[lab]))
    return pd.DataFrame(rows, columns=["text", "label"])


def clean_text(s):
    s = re.sub(r"https?://\S+", " ", str(s).lower())
    s = re.sub(r"[^a-z0-9%$.,'\- ]", " ", s)  # keep numbers/%/$ - they carry signal
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------- 2. models
def train_tfidf(X_tr, y_tr):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    return pipe.fit(X_tr, y_tr)


def build_bilstm(train_texts, max_tokens=20000, seq_len=64):
    vec = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=seq_len)
    vec.adapt(tf.constant(list(train_texts)))
    inp = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vec(inp)
    x = tf.keras.layers.Embedding(max_tokens, 96, mask_zero=True)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def keras_proba(model, texts):
    return model.predict(tf.constant(list(texts))[:, None], batch_size=256, verbose=0)


# ---------------------------------------------------- 3. company extraction
def extract_companies(text):
    return [t for t in COMPANIES
            if _PATTERNS[t].search(text) or _SYMBOLS[t].search(text)]


def sentiment_card(company, text, proba):
    i = int(np.argmax(proba))
    signal = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral / No clear signal"}[LABELS[i]]
    return (f"Company: {company}\nNews: \"{text}\"\n"
            f"Sentiment: {LABELS[i].capitalize()}\nConfidence: {proba[i]:.0%}\n"
            f"Potential Market Signal: {signal}")


# -------------------------------------------------------- 4. market data
def get_prices(tickers, start, end, csv=None, demo=False):
    if demo:  # synthetic random walks - purely for testing the pipeline
        idx = pd.bdate_range(start, end)
        rng = np.random.default_rng(SEED)
        return pd.DataFrame(
            {t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, len(idx)))) for t in tickers},
            index=idx)
    if csv:
        return pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()
    import yfinance as yf  # optional dependency
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all")


def to_trading_day(dates, trading_index):
    """Map each news date to the first trading day on/after it (weekend news -> Monday).
    Returns a DatetimeIndex (NaT where no later trading day exists)."""
    trading_index = pd.DatetimeIndex(trading_index)
    pos = trading_index.searchsorted(pd.DatetimeIndex(dates))
    ok = pos < len(trading_index)
    mapped = trading_index.take(np.minimum(pos, len(trading_index) - 1))
    return mapped.where(ok)


# ------------------------------------------------------------ 5. analytics
def lag_correlations(sent_w, ret_w, lags=range(-3, 4)):
    """Spearman corr(sentiment_t, return_{t+k}). k=1 is the 'subsequent return' test;
    k<0 is a reverse-causality check (does sentiment merely follow past returns?)."""
    rows = []
    for k in lags:
        a = pd.concat([sent_w.stack(), ret_w.shift(-k).stack()], axis=1, keys=["s", "r"]).dropna()
        if len(a) > 10:
            rho, p = stats.spearmanr(a.s, a.r)
            rows.append((k, rho, p, len(a)))
    return pd.DataFrame(rows, columns=["lag", "spearman", "p_value", "n"])


def bucket_analysis(panel):
    p = panel.copy()
    p["bucket"] = pd.qcut(p.sent.rank(method="first"), 3, labels=["Negative", "Neutral", "Positive"])
    means = p.groupby("bucket", observed=True).ret_next.mean() * 1e4  # basis points
    hi, lo = p[p.bucket == "Positive"].ret_next, p[p.bucket == "Negative"].ret_next
    t, pv = stats.ttest_ind(hi, lo, equal_var=False)
    return means, t, pv


# ------------------------------------------------------------- 6. dashboard
def dashboard(path, cm, perf, scored, focus, daily_focus, px_focus, lagdf, bucket_means, demo):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Financial News Sentiment & Market Signal Analytics"
                 + ("  [DEMO: synthetic prices]" if demo else ""), fontsize=15, weight="bold")

    a = ax[0, 0]
    a.imshow(cm, cmap="Blues")
    a.set_xticks(range(3)); a.set_xticklabels(LABELS); a.set_yticks(range(3)); a.set_yticklabels(LABELS)
    for i in range(3):
        for j in range(3):
            a.text(j, i, cm[i, j], ha="center", va="center")
    a.set_title("Confusion matrix (best model, PhraseBank test)")
    a.set_xlabel("Predicted"); a.set_ylabel("True")

    a = ax[0, 1]
    perf.plot.bar(ax=a, rot=0); a.set_ylim(0, 1); a.set_title("Model comparison (test set)")

    a = ax[0, 2]
    scored.sent_label.value_counts().reindex(LABELS).fillna(0).plot.bar(
        ax=a, color=["#d9534f", "#999", "#5cb85c"], rot=0)
    a.set_title("Scored headline sentiment mix")

    a = ax[1, 0]
    a.plot(daily_focus.rolling(5, min_periods=1).mean(), color="tab:blue", label="5d avg sentiment")
    a.set_ylabel("Sentiment (P(pos) - P(neg))", color="tab:blue")
    b = a.twinx()
    b.plot(px_focus / px_focus.iloc[0], color="tab:orange", alpha=.8, label="Price (rebased)")
    b.set_ylabel("Price (rebased)", color="tab:orange")
    a.set_title(f"{focus}: sentiment vs price")

    a = ax[1, 1]
    a.bar(lagdf.lag, lagdf.spearman, color=np.where(lagdf.p_value < 0.05, "tab:red", "tab:gray"))
    a.axhline(0, color="k", lw=.5)
    a.set_title("Spearman corr: sentiment(t) vs return(t+lag)\n(red = p<0.05; lag 1 = next trading day)")
    a.set_xlabel("lag (days)")

    a = ax[1, 2]
    bucket_means.plot.bar(ax=a, color=["#d9534f", "#999", "#5cb85c"], rot=0)
    a.axhline(0, color="k", lw=.5)
    a.set_title("Mean next-day return by sentiment tercile (bps)")

    plt.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrasebank", required=True)
    ap.add_argument("--news", help="CSV with columns: date, headline[, ticker]")
    ap.add_argument("--prices", help="CSV: first col = date, other cols = tickers (close)")
    ap.add_argument("--demo", action="store_true", help="synthetic dates/prices for pipeline testing")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- train/evaluate sentiment models on Financial PhraseBank
    df = load_phrasebank(args.phrasebank)
    df["clean"] = df.text.map(clean_text)
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    print(f"PhraseBank: {len(df)} sentences | class mix:\n{df.label.map(dict(enumerate(LABELS))).value_counts()}\n")

    tfidf = train_tfidf(tr.clean, tr.label)
    p_tfidf = tfidf.predict_proba(te.clean)

    bilstm = build_bilstm(tr.clean)
    bilstm.fit(tf.constant(list(tr.clean))[:, None], tr.label.values, validation_split=0.1,
               epochs=args.epochs, batch_size=32, verbose=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    p_lstm = keras_proba(bilstm, te.clean)
    p_ens = (p_tfidf + p_lstm) / 2

    candidates = {"TF-IDF+LogReg": p_tfidf, "BiLSTM (TF)": p_lstm, "Ensemble": p_ens}
    perf = pd.DataFrame({
        n: {"accuracy": accuracy_score(te.label, p.argmax(1)),
            "macro_F1": f1_score(te.label, p.argmax(1), average="macro")}
        for n, p in candidates.items()}).T
    print(perf.round(3))
    best = perf.macro_F1.idxmax()
    print(f"\nBest model: {best}")
    cm = confusion_matrix(te.label, candidates[best].argmax(1))

    def score_texts(texts):
        c = [clean_text(t) for t in texts]
        return {"TF-IDF+LogReg": lambda: tfidf.predict_proba(c),
                "BiLSTM (TF)": lambda: keras_proba(bilstm, c),
                "Ensemble": lambda: (tfidf.predict_proba(c) + keras_proba(bilstm, c)) / 2}[best]()

    example = "NVIDIA reports stronger-than-expected quarterly revenue, raising full-year guidance."
    print("\n" + sentiment_card("NVIDIA", example, score_texts([example])[0]) + "\n")

    # --- news to analyse
    if args.demo:
        s = te.sample(min(3000, len(te)), random_state=SEED).text.reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        tick = rng.choice(list(COMPANIES), len(s))
        news = pd.DataFrame({
            "date": rng.choice(pd.bdate_range("2023-01-02", "2024-12-31"), len(s)),
            "headline": [f"{COMPANIES[t][0].title()}: {x}" for t, x in zip(tick, s)]})
    elif args.news:
        news = pd.read_csv(args.news, parse_dates=["date"])
    else:
        raise SystemExit("Provide --news (dated headlines) or use --demo.")

    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    if "ticker" in news:
        news["ticker"] = news.ticker.map(lambda t: [t])
    else:
        news["ticker"] = news.headline.map(extract_companies)
    news = news.explode("ticker").dropna(subset=["ticker"]).reset_index(drop=True)
    print(f"News items linked to a company: {len(news)}")

    proba = score_texts(news.headline.tolist())
    news["p_neg"], news["p_neu"], news["p_pos"] = proba.T
    news["sent"] = news.p_pos - news.p_neg            # continuous score in [-1, 1]
    news["confidence"] = proba.max(1)
    news["sent_label"] = [LABELS[i] for i in proba.argmax(1)]

    # --- prices, alignment, daily aggregation
    tickers = sorted(news.ticker.unique())
    px = get_prices(tickers, news.date.min() - pd.Timedelta(days=5),
                    news.date.max() + pd.Timedelta(days=10), args.prices, args.demo)
    px = px[[t for t in tickers if t in px.columns]]
    if px.empty:
        raise SystemExit("No price data matched the tickers found in the news.")
    news = news[news.ticker.isin(px.columns)]
    rets = px.pct_change()

    news["tday"] = to_trading_day(news.date, px.index).values
    news = news.dropna(subset=["tday"])
    daily = news.groupby(["tday", "ticker"]).sent.mean().unstack().reindex(px.index)

    # --- association analysis
    lagdf = lag_correlations(daily, rets)
    print("\nLagged Spearman correlation (lag 1 = next trading day):\n", lagdf.round(4).to_string(index=False))

    panel = pd.concat([daily.stack(), rets.shift(-1).stack()], axis=1, keys=["sent", "ret_next"]).dropna()
    bucket_means, t, pv = bucket_analysis(panel)
    dir_acc = (np.sign(panel.sent) == np.sign(panel.ret_next)).mean()
    print(f"\nMean next-day return by sentiment tercile (bps):\n{bucket_means.round(2)}")
    print(f"Positive vs Negative tercile Welch t={t:.2f}, p={pv:.3f}")
    print(f"Directional agreement (sign sentiment vs sign next-day return): {dir_acc:.1%} (n={len(panel)})")

    # --- dashboard + exports
    focus = news.ticker.value_counts().index[0]
    dashboard(os.path.join(args.out, "dashboard.png"), cm, perf, news, focus,
              daily[focus].dropna(), px[focus].dropna(), lagdf, bucket_means, args.demo)
    news.to_csv(os.path.join(args.out, "scored_news.csv"), index=False)
    panel.to_csv(os.path.join(args.out, "daily_panel.csv"))
    lagdf.to_csv(os.path.join(args.out, "lag_correlations.csv"), index=False)
    print(f"\nSaved dashboard + CSVs to {args.out}/")


if __name__ == "__main__":
    main()