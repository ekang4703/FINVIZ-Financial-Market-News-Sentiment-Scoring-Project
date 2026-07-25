import streamlit as st

from sentiment import get_ticker_sentiment, get_market_sentiment, _load_model


st.set_page_config(page_title="Ticker Sentiment", page_icon="📈", layout="centered")


#Cache the model so it only loads once per session
@st.cache_resource(show_spinner=False)
def load_model_once():
    _load_model()
    return True


#Cache scrape+score results briefly so re-searching the same ticker
@st.cache_data(ttl=300, show_spinner=False)
def cached_ticker_sentiment(ticker: str):
    aggregate, scored = get_ticker_sentiment(ticker)
    #convert dataclasses to plain dicts so streamlit;s cache can use.
    scored_plain = [
        {
            "headline": item.headline,
            "url": item.url,
            "time_raw": item.time_raw,
            "source": item.source,
            "label": result.label,
            "score": result.score,
        }
        for item, result in scored
    ]
    return aggregate, scored_plain


@st.cache_data(ttl=300, show_spinner=False)
def cached_market_sentiment():
    aggregate, scored = get_market_sentiment()
    scored_plain = [
        {
            "headline": item.headline,
            "url": item.url,
            "time_raw": item.time_raw,
            "label": result.label,
            "score": result.score,
        }
        for item, result in scored
    ]
    return aggregate, scored_plain


LABEL_COLOR = {
    "Bullish": "#16a34a",
    "Bearish": "#dc2626",
    "Neutral": "#6b7280",
}

PER_HEADLINE_COLOR = {
    "positive": "#16a34a",
    "negative": "#dc2626",
    "neutral": "#6b7280",
}


def render_aggregate(aggregate, title: str):
    color = LABEL_COLOR.get(aggregate.overall_label, "#6b7280")
    st.markdown(
        f"""
        <div style="padding: 1.2rem; border-radius: 10px; background: {color}1a;
                    border: 1px solid {color}55; margin-bottom: 1rem;">
            <div style="font-size: 0.85rem; color: #888; margin-bottom: 0.2rem;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {color};">
                {aggregate.overall_label}
            </div>
            <div style="font-size: 0.9rem; color: #888; margin-top: 0.3rem;">
                Based on {aggregate.n_articles} headlines
                &nbsp;·&nbsp; {aggregate.n_positive} positive
                &nbsp;·&nbsp; {aggregate.n_negative} negative
                &nbsp;·&nbsp; {aggregate.n_neutral} neutral
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_headline_list(scored_plain):
    for item in scored_plain:
        color = PER_HEADLINE_COLOR.get(item["label"], "#6b7280")
        source = item.get("source")
        meta = f"{item['time_raw']}" + (f" · {source}" if source else "")
        st.markdown(
            f"""
            <div style="padding: 0.6rem 0; border-bottom: 1px solid #2a2a2a;">
                <span style="display:inline-block; padding: 0.1rem 0.5rem; border-radius: 5px;
                             background: {color}22; color: {color}; font-size: 0.75rem;
                             font-weight: 600; text-transform: uppercase; margin-right: 0.5rem;">
                    {item['label']}
                </span>
                <a href="{item['url']}" target="_blank" style="text-decoration: none; color: inherit;">
                    {item['headline']}
                </a>
                <div style="font-size: 0.75rem; color: #888; margin-top: 0.15rem;">{meta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


#UI

st.title("📈 Ticker Sentiment")
st.caption(
    "Scrapes recent Finviz headlines and scores them with FinBERT. "
    "Unofficial tool for personal use -- not financial advice."
)

with st.spinner("Loading sentiment model (first run only)..."):
    load_model_once()

ticker = st.text_input("Enter a ticker", placeholder="e.g. TSLA, AAPL, NVDA").strip().upper()
search_clicked = st.button("Search", type="primary")

st.divider()

if search_clicked and ticker:
    with st.spinner(f"Fetching and scoring news for {ticker}..."):
        aggregate, scored_plain = cached_ticker_sentiment(ticker)

    if aggregate.n_articles == 0:
        st.warning(
            f"No recent headlines found for **{ticker}** in the current Finviz Stocks News feed. "
            "This can happen for less-covered tickers, or simply a quiet news day. Try again later."
        )
    else:
        render_aggregate(aggregate, f"{ticker} — Current News Sentiment")
        st.subheader("Recent headlines")
        render_headline_list(scored_plain)

elif search_clicked and not ticker:
    st.info("Enter a ticker above to get started.")

#Sidebar
with st.sidebar:
    st.subheader("Overall Market Mood")
    if st.button("Refresh market sentiment"):
        cached_market_sentiment.clear()

    with st.spinner("Scoring general market news..."):
        market_aggregate, market_scored = cached_market_sentiment()

    render_aggregate(market_aggregate, "Market-wide (general news)")
    with st.expander("See headlines"):
        render_headline_list(market_scored[:20])
