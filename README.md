# Project
#1.0 Notebook

#1.
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

#2
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

#3
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

#4
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

#5
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

#6
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

#7
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

#8
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


