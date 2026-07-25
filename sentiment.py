"""
Sentiment scoring for Finviz headlines using FinBERT.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned specifically on
financial text, so it understands finance-specific language ("beats
estimates", "misses guidance", "downgrades") much better than a generic
sentiment model would.

First run will download the model (~400MB) from HuggingFace and cache it
locally (usually under ~/.cache/huggingface). After that, loading is fast.

Usage:
    python sentiment.py TSLA
    python sentiment.py          # defaults to overall market sentiment
"""

from __future__ import annotations

import sys
import dataclasses
from typing import Union

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from finviz_news_scraper import (
    NewsItem,
    StocksNewsItem,
    get_news,
    get_stocks_news,
    filter_by_ticker,
)


MODEL_NAME = "ProsusAI/finbert"

# Lazy-loaded globals so importing this module doesn't immediately trigger
# a model download -- only the first actual scoring call does.
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Loading {MODEL_NAME} (first run downloads ~400MB, cached after)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


@dataclasses.dataclass
class SentimentResult:
    label: str          # "positive" | "negative" | "neutral"
    score: float         # confidence 0.0-1.0 for that label
    positive: float       # raw probability for each class, useful for aggregation
    negative: float
    neutral: float


def score_headline(headline: str) -> SentimentResult:
    """Run a single headline through FinBERT and return its sentiment."""
    tokenizer, model = _load_model()

    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # FinBERT's label order: 0=positive, 1=negative, 2=neutral
    labels = ["positive", "negative", "neutral"]
    probs_dict = {label: probs[i].item() for i, label in enumerate(labels)}
    top_label = max(probs_dict, key=probs_dict.get)

    return SentimentResult(
        label=top_label,
        score=probs_dict[top_label],
        positive=probs_dict["positive"],
        negative=probs_dict["negative"],
        neutral=probs_dict["neutral"],
    )


def score_headlines(items: list[Union[NewsItem, StocksNewsItem]]) -> list[tuple]:
    """Score a list of news items. Returns list of (item, SentimentResult) pairs."""
    return [(item, score_headline(item.headline)) for item in items]


@dataclasses.dataclass
class AggregateSentiment:
    overall_label: str      # "Bullish" | "Bearish" | "Neutral"
    avg_positive: float
    avg_negative: float
    avg_neutral: float
    n_articles: int
    n_positive: int
    n_negative: int
    n_neutral: int


def aggregate_sentiment(scored: list[tuple]) -> AggregateSentiment:
    """
    Combine per-headline sentiment scores into one overall reading.

    This is a simple average -- no recency weighting, no source-credibility
    weighting. Good enough for an MVP; worth revisiting once you have a
    sense of whether recent headlines should count more than older ones.
    """
    if not scored:
        return AggregateSentiment("Neutral", 0.0, 0.0, 0.0, 0, 0, 0, 0)

    n = len(scored)
    avg_pos = sum(r.positive for _, r in scored) / n
    avg_neg = sum(r.negative for _, r in scored) / n
    avg_neu = sum(r.neutral for _, r in scored) / n

    n_pos = sum(1 for _, r in scored if r.label == "positive")
    n_neg = sum(1 for _, r in scored if r.label == "negative")
    n_neu = sum(1 for _, r in scored if r.label == "neutral")

    # Simple net-sentiment bucket. Tune these thresholds as you see more data.
    net = avg_pos - avg_neg
    if net > 0.15:
        overall = "Bullish"
    elif net < -0.15:
        overall = "Bearish"
    else:
        overall = "Neutral"

    return AggregateSentiment(
        overall_label=overall,
        avg_positive=avg_pos,
        avg_negative=avg_neg,
        avg_neutral=avg_neu,
        n_articles=n,
        n_positive=n_pos,
        n_negative=n_neg,
        n_neutral=n_neu,
    )


def get_ticker_sentiment(ticker: str) -> tuple[AggregateSentiment, list[tuple]]:
    """
    Full pipeline for one ticker: scrape Stocks News, filter by ticker,
    score each headline, return the aggregate + the per-headline detail.
    """
    all_stock_news = get_stocks_news()
    ticker_news = filter_by_ticker(all_stock_news, ticker)
    scored = score_headlines(ticker_news)
    aggregate = aggregate_sentiment(scored)
    return aggregate, scored


def get_market_sentiment() -> tuple[AggregateSentiment, list[tuple]]:
    """Full pipeline for overall market mood using the general news feed."""
    news = get_news("market_news")
    scored = score_headlines(news)
    aggregate = aggregate_sentiment(scored)
    return aggregate, scored


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else None

    if ticker:
        print(f"\nFetching and scoring news for {ticker}...\n")
        aggregate, scored = get_ticker_sentiment(ticker)

        if aggregate.n_articles == 0:
            print(f"No recent headlines found for {ticker} in the current Stocks News feed.")
            print("Try a more actively-traded/newsy ticker, or check back later.")
            sys.exit(0)
    else:
        print("\nNo ticker given -- scoring overall market news...\n")
        aggregate, scored = get_market_sentiment()

    print(f"{'='*60}")
    print(f"Overall sentiment: {aggregate.overall_label}")
    print(f"  Based on {aggregate.n_articles} headlines")
    print(f"  Positive: {aggregate.n_positive}  Negative: {aggregate.n_negative}  Neutral: {aggregate.n_neutral}")
    print(f"  Avg scores -- pos: {aggregate.avg_positive:.3f}  neg: {aggregate.avg_negative:.3f}  neu: {aggregate.avg_neutral:.3f}")
    print(f"{'='*60}\n")

    print("Per-headline breakdown:\n")
    for item, result in scored:
        print(f"[{result.label.upper():8}] ({result.score:.2f}) {item.headline}")
    print()
