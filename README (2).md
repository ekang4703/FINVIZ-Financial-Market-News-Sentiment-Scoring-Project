# 📈 Ticker Sentiment

Real-time financial news sentiment analysis using FinBERT. Search any stock ticker and see what recent financial headlines say about it—powered by a transformer model trained specifically on financial language.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Ticker-specific sentiment**: Search any stock ticker and get sentiment on recent news about it
- **Overall market mood**: See general market sentiment from broad financial news
- **FinBERT scoring**: Uses a BERT model fine-tuned on financial text for context-aware analysis
- **Live headlines**: View individual headlines with sentiment scores and source attribution
- **Cached results**: Quick re-searches (5-minute cache for the same ticker)

## Tech Stack

- **Web Scraping**: [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) + [Requests](https://requests.readthedocs.io/)
- **Sentiment Model**: [FinBERT](https://huggingface.co/ProsusAI/finbert) (via Hugging Face Transformers)
- **ML Framework**: [PyTorch](https://pytorch.org/)
- **Web UI**: [Streamlit](https://streamlit.io/)

## Installation

### Requirements
- Python 3.8+
- ~2GB disk space (for the FinBERT model on first run)

### Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/ticker-sentiment.git
   cd ticker-sentiment
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   On first run, the FinBERT model (~400MB) will be downloaded from Hugging Face and cached locally (usually `~/.cache/huggingface`). This is automatic—just wait for it.

## Usage

### Run the Streamlit app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Enter a ticker symbol (e.g., `TSLA`, `AAPL`, `NVDA`) and hit "Search" to see recent sentiment.

### Use as a Python library

```python
from sentiment import get_ticker_sentiment, get_market_sentiment

# Single ticker
aggregate, scored = get_ticker_sentiment("TSLA")
print(f"Overall: {aggregate.overall_label}")
print(f"Positive: {aggregate.n_positive}, Negative: {aggregate.n_negative}")

# Overall market
market_agg, market_scored = get_market_sentiment()
```

### Command-line interface
```bash
# Score a specific ticker
python sentiment.py TSLA

# Score overall market (no ticker)
python sentiment.py
```

## How It Works

1. **News Scraping** (`finviz_news_scraper.py`)
   - Scrapes [Finviz](https://finviz.com/news) without permission (for personal use only)
   - Covers three free news feeds: Market News, Market Pulse, and Stocks News
   - Extracts headlines, timestamps, URLs, and ticker tags

2. **Sentiment Scoring** (`sentiment.py`)
   - Feeds each headline through FinBERT (a BERT model fine-tuned on financial text)
   - Returns per-headline sentiment: `positive`, `negative`, or `neutral` with confidence scores
   - Aggregates individual scores into overall market/ticker sentiment: `Bullish`, `Bearish`, or `Neutral`

3. **Web UI** (`app.py`)
   - Built with Streamlit for quick iteration
   - Caches results to avoid re-scraping the same ticker repeatedly
   - Shows asentiment with a color-coded badge and detailed headline breakdown

## Limitations & Disclaimers

**Not financial advice.** This is a personal research tool. Always do your own due diligence before making investment decisions.

- **Web scraping**: This tool scrapes Finviz without explicit permission. Use responsibly; don't hammer the site with constant requests. It's intended for personal at-home use only.
- **Model limitations**: FinBERT is trained on formal financial text. Sarcasm, satire, and nuance may not be captured perfectly.
- **News lag**: Some Finviz rows may say "Loading..." and are filtered out automatically.
- **Simple aggregation**: Current sentiment calculation is a straightforward average with no recency weighting or source-credibility scoring.
- **Coverage**: Smaller or less-frequently-traded tickers may have few or no headlines on any given day.

## Project Structure

```
ticker-sentiment/
├── app.py                      # Streamlit web app
├── finviz_news_scraper.py      # News scraping
├── sentiment.py                # Sentiment scoring
├── requirements.txt            # Python dependencies
└── README.md
```

## Acknowledgments

- **FinBERT**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on Hugging Face
- **Finviz**: Free financial data via [finviz.com](https://finviz.com)
- **Streamlit**: For the quick and easy web framework

---

**Disclaimer**: This tool is for personal educational and research purposes only. It is not financial advice, and the author makes no guarantee as to its accuracy or reliability. Always consult with a qualified financial advisor before making investment decisions.
