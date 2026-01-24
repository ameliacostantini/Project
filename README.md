# Project
#1.0 Notebook

#cell 1
# --- Environment + NLTK Setup (Run once per kernel) ---
# Purpose: ensure VADER sentiment + tokenizers are available locally

import os
import nltk

NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

def download_if_missing(resource_path: str, package_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading NLTK package: {package_name}")
        nltk.download(package_name, download_dir=NLTK_DATA_DIR, quiet=False)

download_if_missing("sentiment/vader_lexicon", "vader_lexicon")
download_if_missing("tokenizers/punkt", "punkt")
download_if_missing("tokenizers/punkt_tab", "punkt_tab")

print("NLTK data path:", nltk.data.path[0])

#cell 2
# --- Imports + Project Universe (EM Asia) ---

import time, random
import requests
import pandas as pd
from datetime import datetime, timezone

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# EM Asia focus universe (edit to match your project scope)
ASIAN_COUNTRIES = {
    "China": "CN",
    "India": "IN",
    "Indonesia": "ID",
    "Malaysia": "MY",
    "Philippines": "PH",
    "Thailand": "TH",
    "Vietnam": "VN",
    "Pakistan": "PK",
}

# Tradable ETF proxies (best-effort; adjust as needed for your write-up)
COUNTRY_TO_ETF = {
    "China": "MCHI",
    "India": "INDA",
    "Indonesia": "EIDO",
    "Malaysia": "EWM",
    "Philippines": "EPHE",
    "Thailand": "THD",
    "Vietnam": "VNM",
    "Pakistan": "PAK",
}

#cell 3
# --- Query design: Fiscal vs Monetary policy sentiment ---
# We keep terms policy-focused and not too broad (avoids GDELT "too common" issues)

MONETARY_QUERY = (
    '("central bank" OR "policy rate" OR "interest rate" OR inflation OR '
    '"rate hike" OR "rate cut" OR tightening OR easing)'
)

FISCAL_QUERY = (
    '("budget" OR "fiscal stimulus" OR "government spending" OR deficit OR debt OR '
    '"tax cut" OR "tax hike" OR subsidy OR subsidies OR capex OR "public investment")'
)

#cell 4
# --- GDELT fetch helpers (stable, rate-limit aware) ---

def looks_like_json(text: str) -> bool:
    t = (text or "").lstrip()
    return t.startswith("{") or t.startswith("[")

def fetch_gdelt_articles(country_code: str,
                         query: str,
                         timespan: str = "7d",
                         maxrecords: int = 25,
                         retries: int = 2):
    """
    Pull GDELT DOC 2.0 articles.
    - Uses timespan (simple + works well)
    - Filters English AFTER pulling (do NOT put language:english in query)
    - Small retries to avoid long "hangs" in notebooks
    """
    params = {
        "query": f"{query} sourcecountry:{country_code}",
        "mode": "artlist",
        "format": "json",
        "timespan": timespan,
        "maxrecords": maxrecords,
        "sort": "datedesc",
    }

    for attempt in range(retries):
        time.sleep(0.25 + random.random() * 0.25)
        r = requests.get(GDELT_DOC_ENDPOINT, params=params, timeout=20)

        if r.status_code == 429:
            wait = 1.2 * (attempt + 1)
            print(f"[{country_code}] 429 rate limit. sleeping {wait:.1f}s")
            time.sleep(wait)
            continue

        if r.status_code != 200:
            print(f"[{country_code}] status={r.status_code} head={(r.text or '')[:120]!r}")
            return []

        if not looks_like_json(r.text):
            print(f"[{country_code}] non-JSON head={(r.text or '')[:120]!r}")
            return []

        data = r.json()
        articles = data.get("articles", [])
        articles = [a for a in articles if (a.get("language") or "").lower() == "english"]
        return articles

    return []

#cell 5
# --- Pull GDELT Asia policy news into a single dataframe ---

def pull_asia_policy_news(timespan="7d", maxrecords=25):
    rows = []
    for country, code in ASIAN_COUNTRIES.items():
        for topic, q in [("monetary", MONETARY_QUERY), ("fiscal", FISCAL_QUERY)]:
            articles = fetch_gdelt_articles(code, q, timespan=timespan, maxrecords=maxrecords, retries=2)

            for a in articles:
                rows.append({
                    "country": country,
                    "country_code": code,
                    "topic": topic,
                    "seendate": a.get("seendate"),
                    "headline": a.get("title"),
                    "url": a.get("url"),
                    "language": a.get("language"),
                })

    df = pd.DataFrame(rows).dropna(subset=["headline"])
    if df.empty:
        return df

    df["datetime_utc"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    df["date"] = df["datetime_utc"].dt.date

    df = df.sort_values(["country","date","topic","datetime_utc"],
                        ascending=[True, True, True, False]).reset_index(drop=True)
    return df

gdelt_df = pull_asia_policy_news(timespan="7d", maxrecords=25)

print("Rows pulled:", len(gdelt_df))
print(gdelt_df[["country","topic","date","headline"]].head(15))

#cell 6
# --- Sentiment scoring + daily aggregation ---

if gdelt_df.empty:
    print("No articles returned. Try timespan='14d' or maxrecords=50.")
else:
    gdelt_df["vader_compound"] = gdelt_df["headline"].apply(lambda t: sia.polarity_scores(t)["compound"])

    daily_sentiment = (
        gdelt_df.groupby(["date","country","topic"], as_index=False)
        .agg(
            sentiment=("vader_compound","mean"),
            n_articles=("headline","count")
        )
        .sort_values(["country","date","topic"], ascending=[True, True, True])
        .reset_index(drop=True)
    )

    print("Daily sentiment (top 25):")
    print(daily_sentiment.head(25))

#cell 7
# =========================
# CELL 7 — RAW HEADLINES (GDELT, EM Asia)
# Prints a sample of real headlines pulled from GDELT (not mock data)
# =========================

# Guards
if "gdelt_df" not in globals():
    raise NameError("gdelt_df not found. Run the GDELT pull cell (Cell 5) first.")

if gdelt_df.empty:
    print("gdelt_df is empty. Increase timespan/maxrecords in the GDELT pull cell and rerun.")
else:
    # Optional: filter to one topic or one country if you want
    # gdelt_sample = gdelt_df[gdelt_df["topic"] == "monetary"].copy()
    gdelt_sample = gdelt_df.copy()

    # Take a stable sample (most recent first)
    gdelt_sample = gdelt_sample.sort_values("datetime_utc", ascending=False)

    N = 10  # how many headlines to print
    headlines = gdelt_sample["headline"].dropna().head(N).tolist()

    print("Raw Headlines:\n")
    for hl in headlines:
        print("-", hl)

    # store for next cell
    raw_headlines = headlines

#cell 8
# =========================
# CELL 8 — FILTERED TOKENS + SENTIMENT (GDELT headlines)
# Matches the exact print style from your example
# =========================

from pprint import pprint
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Guards
if "raw_headlines" not in globals():
    raise NameError("raw_headlines not found. Run Cell 7 first.")

if len(raw_headlines) == 0:
    print("raw_headlines is empty. Nothing to analyze.")
else:
    stop_words = set(stopwords.words("english"))
    sia = SentimentIntensityAnalyzer()

    print("\nFiltered and Analyzed Headlines:\n")

    for hl in raw_headlines:
        tokens = word_tokenize(hl)

        # Keep alphabetic tokens only; drop punctuation/numbers
        filtered = [
            t.lower()
            for t in tokens
            if t.isalpha() and t.lower() not in stop_words
        ]

        scores = sia.polarity_scores(hl)

        print("Headline:", hl)
        print("Filtered Tokens:", filtered)
        print("Sentiment:")
        pprint(scores)
        print("-" * 50)

#2.0 Notebook

#cell 1
# Notebook setup: versions + NLTK data downloads (run once)
import sys
from pathlib import Path
import nltk

# Keep NLTK data local to the project for reproducibility
NLTK_DATA_DIR = Path("nltk_data").resolve()
NLTK_DATA_DIR.mkdir(exist_ok=True)

# Ensure NLTK searches this directory first
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

def _download_if_missing(resource_path: str, download_name: str | None = None) -> bool:
    try:
        nltk.data.find(resource_path)
        return False
    except LookupError:
        nltk.download(download_name or resource_path.split("/")[-1], download_dir=str(NLTK_DATA_DIR), quiet=True)
        return True

resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("sentiment/vader_lexicon", "vader_lexicon"),
]

attempted = []
for resource_path, download_name in resources:
    if _download_if_missing(resource_path, download_name):
        attempted.append(download_name)

print(f"Python: {sys.version.split()[0]}")
print(f"NLTK:   {nltk.__version__}")
print(f"NLTK data dir: {NLTK_DATA_DIR}")
print("Downloaded:" if attempted else "NLTK resources already present ✅", ", ".join(attempted) if attempted else "")

#cell 2
from collections import Counter
import re
import time, random
import requests
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

ASIAN_COUNTRIES = {
    "China": "CN",
    "India": "IN",
    "Indonesia": "ID",
    "Malaysia": "MY",
    "Philippines": "PH",
    "Thailand": "TH",
    "Vietnam": "VN",
    "Pakistan": "PK",
}

MONETARY_QUERY = (
    '("central bank" OR "policy rate" OR "interest rate" OR inflation OR '
    '"rate hike" OR "rate cut" OR tightening OR easing)'
)

FISCAL_QUERY = (
    '("budget" OR "fiscal stimulus" OR "government spending" OR deficit OR debt OR '
    '"tax cut" OR "tax hike" OR subsidy OR subsidies OR capex OR "public investment")'
)

def _looks_like_json(text: str) -> bool:
    t = (text or "").lstrip()
    return t.startswith("{") or t.startswith("[")

def fetch_gdelt(country_code: str, query: str, timespan: str = "7d", maxrecords: int = 25, retries: int = 2):
    params = {
        "query": f"{query} sourcecountry:{country_code}",
        "mode": "artlist",
        "format": "json",
        "timespan": timespan,
        "maxrecords": maxrecords,
        "sort": "datedesc",
    }
    for attempt in range(retries):
        time.sleep(0.25 + random.random() * 0.25)
        r = requests.get(GDELT_DOC_ENDPOINT, params=params, timeout=20)
        if r.status_code == 429:
            time.sleep(1.2 * (attempt + 1))
            continue
        if r.status_code != 200 or not _looks_like_json(r.text):
            return []
        data = r.json()
        arts = data.get("articles", [])
        # filter English AFTER pull (more stable than language:english in query)
        return [a for a in arts if (a.get("language") or "").lower() == "english"]
    return []

def pull_asia_headlines(timespan="7d", maxrecords=25):
    rows = []
    for country, code in ASIAN_COUNTRIES.items():
        for topic, q in [("monetary", MONETARY_QUERY), ("fiscal", FISCAL_QUERY)]:
            arts = fetch_gdelt(code, q, timespan=timespan, maxrecords=maxrecords, retries=2)
            for a in arts:
                rows.append({
                    "country": country,
                    "country_code": code,
                    "topic": topic,
                    "seendate": a.get("seendate"),
                    "headline": a.get("title"),
                    "url": a.get("url"),
                    "language": a.get("language"),
                })
    df = pd.DataFrame(rows).dropna(subset=["headline"])
    if df.empty:
        return df
    df["datetime_utc"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    df["date"] = df["datetime_utc"].dt.date
    return df.sort_values(["date","country","topic","datetime_utc"], ascending=[True, True, True, False]).reset_index(drop=True)

gdelt_df = pull_asia_headlines(timespan="7d", maxrecords=25)
print("Rows pulled:", len(gdelt_df))
print(gdelt_df[["country","topic","date","headline"]].head(8))

# Tokenization demo using REAL GDELT headlines (combine a small sample into mini_text)
sample_text = "\n".join(gdelt_df["headline"].head(30).tolist()) if not gdelt_df.empty else ""
regex_tokens = re.findall(r"\w+", sample_text.lower())
nltk_tokens = [t for t in word_tokenize(sample_text.lower()) if t.isalnum()]

print("\nRegex tokens (first 30):", regex_tokens[:30])
print("\nNLTK tokens (first 30):", nltk_tokens[:30])

fd = FreqDist(nltk_tokens)
print("\nTop unigrams (from GDELT headlines sample):")
for word, freq in fd.most_common(12):
    print(f"{word:>12} : {freq}")

#cell 3
import random
import re
import nltk

random.seed(7)

if gdelt_df.empty:
    print("No GDELT data yet — rerun Cell 2 with larger timespan/maxrecords.")
else:
    text_corpus = " ".join(gdelt_df["headline"].head(80).tolist())
    tokens = re.findall(r"\w+", text_corpus.lower())
    bigrams_list = list(nltk.bigrams(tokens))

    markov_chain: dict[str, list[str]] = {}
    for w1, w2 in bigrams_list:
        markov_chain.setdefault(w1, []).append(w2)

    print("Naive Markov Chain Structure (sample 12 keys):\n")
    for i, (key, followers) in enumerate(markov_chain.items()):
        if i >= 12:
            break
        print(f"{key} -> {followers[:8]}{'...' if len(followers) > 8 else ''}")

    start_word = random.choice(tokens) if tokens else "policy"
    chain_output = [start_word]
    for _ in range(12):
        last = chain_output[-1]
        if last in markov_chain:
            chain_output.append(random.choice(markov_chain[last]))
        else:
            break

    print("\nRandomly Generated Sequence:", " ".join(chain_output))

#cell 4
import re

if gdelt_df.empty:
    print("No GDELT data yet — rerun Cell 2.")
else:
    finance_text = "\n".join(gdelt_df["headline"].head(25).tolist())

    # Token pattern: tickers, $ amounts, % amounts, numbers, words
    pattern = r"(?:\b[A-Z]{2,5}\b|\$?\d+(?:\.\d+)?(?:B|M|K)?|\d+%|\w+)"
    custom_tokens = re.findall(pattern, finance_text)
    custom_tokens = [t for t in custom_tokens if t.strip()]

    print("Custom-Tokenized Output (first 80 tokens):")
    print(custom_tokens[:80])

#cell 5
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.corpus import stopwords
from collections import Counter

stop_words = set(stopwords.words("english"))

def tokenize_clean(text: str) -> list[str]:
    toks = word_tokenize(text.lower())
    return [t for t in toks if t.isalnum() and t not in stop_words]

if gdelt_df.empty:
    print("No GDELT data yet — rerun Cell 2.")
else:
    # Compare bigrams by topic
    for topic in ["monetary", "fiscal"]:
        subset = gdelt_df[gdelt_df["topic"] == topic]["headline"].dropna().head(120).tolist()
        all_text = " ".join(subset)

        tokens_clean = tokenize_clean(all_text)
        bg = list(bigrams(tokens_clean))
        bg_freq = Counter(bg)

        print(f"\nTop 10 Bigrams — topic={topic}:")
        for pair, freq in bg_freq.most_common(10):
            print(pair, freq)

    # store for later cells (use monetary by default)
    tokens_clean_monetary = tokenize_clean(" ".join(gdelt_df[gdelt_df["topic"]=="monetary"]["headline"].dropna().head(200).tolist()))

#cell 6
import math
from collections import Counter
import nltk
from nltk.util import bigrams

if gdelt_df.empty:
    print("No GDELT data yet — rerun Cell 2.")
else:
    # Use "documents" as batches of headlines by (date, country, topic)
    grouped = (
        gdelt_df.dropna(subset=["headline"])
        .groupby(["date","country","topic"])["headline"]
        .apply(lambda x: " ".join(x.tolist()))
        .reset_index(name="doc_text")
    )

    stop_words = set(stopwords.words("english"))
    processed_docs = []

    for doc in grouped["doc_text"].head(60):  # cap for speed
        tokens = word_tokenize(doc.lower())
        filtered = [t for t in tokens if t.isalnum() and t not in stop_words]
        processed_docs.append(filtered)

    all_bigrams = []
    for tokens_doc in processed_docs:
        all_bigrams.extend(list(nltk.bigrams(tokens_doc)))

    bigram_freq = Counter(all_bigrams)
    total_token_count = sum(len(doc) for doc in processed_docs)

    print("Top 10 Bigrams with raw + normalized frequency:\n")
    for pair, freq in bigram_freq.most_common(10):
        norm_freq = freq / total_token_count if total_token_count else 0
        print(f"{pair} -> raw: {freq}, normalized: {norm_freq:.4f}")

#cell 7
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

if "processed_docs" not in globals() or len(processed_docs) == 0:
    print("processed_docs not found/empty. Run Cell 6 first (needs GDELT data).")
else:
    N = 2
    train_data, padded_sents = padded_everygram_pipeline(N, processed_docs)

    lm = Laplace(N)
    lm.fit(train_data, padded_sents)

    # Try a few conditional probabilities
    print("P('bank' | 'central') =", lm.score("bank", ["central"]))
    print("P('rate' | 'interest') =", lm.score("rate", ["interest"]))

    test_sentence = "central bank policy rate".split()
    test_ngrams, _ = padded_everygram_pipeline(N, [test_sentence])
    perp = lm.perplexity(next(test_ngrams))

    print("\nTest sentence:", " ".join(test_sentence))
    print("Perplexity:", round(perp, 3))

    generated = lm.generate(12, text_seed=["central"])
    print("\nGenerated (12 tokens):", " ".join(generated))

#cell 8
if "bigram_freq" not in globals():
    print("bigram_freq not found. Run Cell 6 first.")
else:
    cost_related = [
        (pair, freq) for pair, freq in bigram_freq.items()
        if ("cost" in pair) or ("expense" in pair) or ("savings" in pair)
    ]

    cost_sorted = sorted(cost_related, key=lambda x: x[1], reverse=True)

    print("Bigrams referencing 'cost' or 'expense' or 'savings':")
    if not cost_sorted:
        print("(none found in this sample)")
    else:
        for pair, freq in cost_sorted[:30]:
            print(pair, "-", freq)

#cell 9
from collections import Counter
import matplotlib.pyplot as plt
from nltk.util import bigrams

stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in stop_words]

if gdelt_df.empty:
    print("No GDELT data yet — rerun Cell 2.")
else:
    # Make "documents" keyed by DATE_COUNTRY_TOPIC (like your quarterly docs, but real)
    grouped = (
        gdelt_df.dropna(subset=["headline"])
        .groupby(["date","country","topic"])["headline"]
        .apply(lambda x: " ".join(x.tolist()))
        .reset_index(name="doc_text")
    )

    # Build dict of docs
    docs = {}
    for _, row in grouped.iterrows():
        doc_id = f"{row['date']}_{row['country']}_{row['topic']}"
        docs[doc_id] = row["doc_text"]

    # Extract bigrams and accumulate counts (global + per-doc)
    bigram_freq_all = Counter()
    bigram_freq_by_doc: dict[str, Counter] = {}

    # Cap for speed in class notebooks
    for doc_id, text in list(docs.items())[:40]:
        tokens = preprocess(text)
        bg = list(bigrams(tokens))
        doc_counter = Counter(bg)
        bigram_freq_by_doc[doc_id] = doc_counter
        bigram_freq_all.update(doc_counter)

    # Top 10 bigrams globally
    top_10_bigrams = bigram_freq_all.most_common(10)
    print("Top 10 global bigrams across GDELT docs:\n")
    for (bg, freq) in top_10_bigrams:
        print(bg, "-", freq)

    # Top 3 bigrams per document (first 10 docs)
    print("\nTop 3 bigrams per document (first 10 docs):\n")
    for doc_id, ctr in list(bigram_freq_by_doc.items())[:10]:
        top3 = ctr.most_common(3)
        pretty = ", ".join([f"{a} {b} ({n})" for ((a, b), n) in top3]) if top3 else "(none)"
        print(f"{doc_id}: {pretty}")

    # Bar chart (global top 10)
    labels = [f"{x[0][0]} {x[0][1]}" for x in top_10_bigrams]
    counts = [x[1] for x in top_10_bigrams]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Bigrams Across GDELT EM Asia Policy Docs")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Filter for bigrams referencing 'cost' or 'savings'
    cost_related = [(bg, count) for (bg, count) in bigram_freq_all.items()
                    if ("cost" in bg) or ("savings" in bg)]
    print("\nBigrams referencing 'cost' or 'savings':")
    if not cost_related:
        print("(none found)")
    else:
        for pair, freq in sorted(cost_related, key=lambda x: x[1], reverse=True)[:30]:
            print(pair, "-", freq)



