"""
Prototype scraper for Finviz's free news pages.

Covers:
  - https://finviz.com/news          (Market News, "By Time" view)
  - https://finviz.com/news?v=6      (Market Pulse)
  - https://finviz.com/news?v=3      (Stocks News)

Notes / limitations:
  - scrapes finviz.com without permission. OK for simple at home project but do not use for public or for profit work
  - some news rows lag behind due to JS limitations, as a result they just say "Loading..." we filter those out wihout attempting
    to see what they say.
  - Do not constantly request from site, use a real User-Agent and don't scrape the
    page over and over. Reuse data when possible
  - The site can change anytime. If this breaks, check
    _parse_news_table() first, since that is where page is read from
"""

from __future__ import annotations

import re
import time
import dataclasses
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup


HEADERS = {
    # Pretend to be a real browser. requests default UA gets blocked a lot.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
}

NEWS_URLS = {
    "market_news": "https://finviz.com/news",
    "market_pulse": "https://finviz.com/news?v=6",
    "stocks_news": "https://finviz.com/news?v=3",
}


@dataclasses.dataclass
class StocksNewsItem:
    time_raw: str                  # e.g. "33 min", "1 hour", "Jun-28"
    headline: str
    url: str
    tickers: list[dict]            # [{"ticker": "SMCI", "change": "+15.66%"}, ...]
    source: str                    # e.g. "Bloomberg", "Reuters"


def _parse_stocks_news_table(html: str) -> list[StocksNewsItem]:
    
    #Parse finviz.com/news?v=3 ("Stocks News").
    soup = BeautifulSoup(html, "html.parser")
    items: list[StocksNewsItem] = []

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 1:
            continue
        if len(row.get_text(strip=True)) > 600:
            continue  # too much text = probably a big nested table, not a real row

        links = row.find_all("a")
        if not links:
            continue

        headline_link = None
        ticker_links = []
        for link in links:
            href = link.get("href", "")
            if "stock?t=" in href:
                ticker_links.append(link)
            elif href.startswith("http") and headline_link is None:
                headline_link = link

        if headline_link is None or not ticker_links:
            continue  # not a real news row, skip it (probably nav/footer junk)

        headline = headline_link.get_text(strip=True)
        if not headline:
            continue
        headline_url = headline_link["href"]  # save this now, before deleting tags below

        tickers = []
        ticker_change_re = re.compile(r"^([A-Za-z.]+)\s*([+-]\d+(?:\.\d+)?%)?$")
        for tl in ticker_links:
            text = tl.get_text(strip=True)
            if not text:
                continue
            m = ticker_change_re.match(text)
            if m:
                tickers.append({"ticker": m.group(1), "change": m.group(2)})
            else:
                # Couldn't match the pattern, just keep the raw text instead of dropping it
                tickers.append({"ticker": text, "change": None})

        time_raw = cells[0].get_text(strip=True)
        '''
        To find the source name, delete the headline link, the ticker links, and the time 
        cell from the row. Whatever text is left over is the source. This works better than 
        trying to string-replace, since the price change text isn't always formatted properly
        inside the ticker link.
        '''
        headline_link.decompose()
        for tl in ticker_links:
            tl.decompose()
        cells[0].decompose()
        remainder = row.get_text(" ", strip=True)
        remainder = re.sub(r"\+\d+\s*More", "", remainder)   # remove "+2 More" tag
        remainder = re.sub(r"[+-]\d+(\.\d+)?%", "", remainder)  # remove leftover % text
        source = remainder.strip(" +").strip()


        items.append(
            StocksNewsItem(
                time_raw=time_raw,
                headline=headline,
                url=headline_url,
                tickers=tickers,
                source=source,
            )
        )

    return items


def get_stocks_news() -> list[StocksNewsItem]:
    """Get and parse finviz.com/news?v=3 (Stocks News, tagged by ticker)."""
    html = fetch_page(NEWS_URLS["stocks_news"])
    return _parse_stocks_news_table(html)


def filter_by_ticker(items: list[StocksNewsItem], ticker: str) -> list[StocksNewsItem]:
    """Only keep items that are tagged with this ticker."""
    ticker = ticker.upper()
    return [item for item in items if any(t["ticker"] == ticker for t in item.tickers)]


@dataclasses.dataclass
class NewsItem:
    time_raw: str          # e.g. "02:56AM" or "Jun-28"
    headline: str
    url: str
    source_page: str       # which NEWS_URLS page this came from


def fetch_page(url: str, timeout: int = 10) -> str:
    """Download the raw HTML for a Finviz news page."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _parse_news_table(html: str, source_page: str) -> list[NewsItem]:
    """
    Pull the news items out of a Finviz news page.

    Each story is a table row (<tr>) with a time in one cell and a
    headline link in another. We look for every link that looks like
    a news headline, then grab the time from the same row.
    """
    soup = BeautifulSoup(html, "html.parser")
    items: list[NewsItem] = []

    # A real news row has a <td> with the time and a <td> with a link
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        '''
        sometimes Finviz stuffs a whole mini-table of headlines into one big row, we skip 
        these, and that row has way more text than a normal one. Real single-story rows are short,
        so we skip anything too long.
        '''
        if len(row.get_text(strip=True)) > 300:
            continue

        link = row.find("a")
        if link is None or not link.get("href"):
            continue

        href = link["href"]
        '''
        Real news stories start with http. On the news pages finviz has nav links that will get scraped along
        with headlines, these start with # "/" or "news". So we just auto filter these out
        '''
        if not href.startswith("http"):
            continue

        headline = link.get_text(strip=True)
        if not headline or headline.lower() == "loading…":
            continue

        # Grab the time from the cell right before the headline cell.
        link_cell_index = next(
            (i for i, c in enumerate(cells) if c.find("a") is link), None
        )
        if link_cell_index is not None and link_cell_index > 0:
            time_cell = cells[link_cell_index - 1].get_text(strip=True)
        else:
            time_cell = cells[0].get_text(strip=True)

        items.append(
            NewsItem(
                time_raw=time_cell,
                headline=headline,
                url=link["href"],
                source_page=source_page,
            )
        )

    return items


def get_news(page: str = "market_news") -> list[NewsItem]:
    #locates and parces through one of three free news pages on finviz
    if page not in NEWS_URLS:
        raise ValueError(f"page must be one of {list(NEWS_URLS)}")

    html = fetch_page(NEWS_URLS[page])
    return _parse_news_table(html, page)


def get_all_news(delay_seconds: float = 2.0) -> list[NewsItem]:
    #Get all three free news pages
    all_items: list[NewsItem] = []
    pages = list(NEWS_URLS.keys())

    for i, page in enumerate(pages):
        all_items.extend(get_news(page))
        if i < len(pages) - 1:
            time.sleep(delay_seconds)

    return all_items


def filter_by_company(items: list[NewsItem], company_or_ticker: str) -> list[NewsItem]:
    #filter: keep headlines that mention the company name or ticker
    needle = company_or_ticker.lower()
    return [item for item in items if needle in item.headline.lower()]


if __name__ == "__main__":
    print("Fetching Finviz Market News...\n")
    news = get_news("market_news")
    print(f"Parsed {len(news)} headlines.\n")
    for item in news[:10]:
        print(f"[{item.time_raw}] {item.headline}")
        print(f"    {item.url}\n")

    print("\n" + "=" * 60)
    print("Fetching Finviz Stocks News (ticker-tagged)...\n")
    stock_news = get_stocks_news()
    print(f"Parsed {len(stock_news)} ticker-tagged headlines.\n")
    for item in stock_news[:10]:
        tickers_str = ", ".join(
            f"{t['ticker']}{' ' + t['change'] if t['change'] else ''}"
            for t in item.tickers
        )
        print(f"[{item.time_raw}] {item.headline}")
        print(f"    Tickers: {tickers_str}")
        print(f"    Source: {item.source}")
        print(f"    {item.url}\n")

    #filter Stocks News down to one ticker
    #tsla_news = filter_by_ticker(stock_news, "TSLA")