from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


TICKERS = [
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "META",
]

RAW_DIR = Path("data/raw/news")
RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def get_news_for_ticker(ticker):

    print(f"\nDownloading news for {ticker}...")

    stock = yf.Ticker(ticker)

    try:
        news_items = stock.news
    except Exception as e:
        print(
            f"ERROR downloading {ticker}: {e}"
        )
        return []

    rows = []

    for item in news_items:

        try:

            content = item.get(
                "content",
                {}
            )

            title = content.get(
                "title",
                ""
            )

            provider = content.get(
                "provider",
                {}
            )

            source = provider.get(
                "displayName",
                ""
            )

            canonical_url = content.get(
                "canonicalUrl",
                {}
            )

            url = canonical_url.get(
                "url",
                ""
            )

            published = content.get(
                "pubDate",
                None
            )

            if not title:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "published_at": published,
                    "headline": title,
                    "source": source,
                    "url": url,
                    "collected_at": datetime.now().isoformat()
                }
            )

        except Exception as e:

            print(
                f"Skipping malformed item: {e}"
            )

    return rows


def main():

    all_news = []

    for ticker in TICKERS:

        rows = get_news_for_ticker(
            ticker
        )

        print(
            f"Found {len(rows)} articles"
        )

        all_news.extend(
            rows
        )

    if not all_news:

        print(
            "\nNo news was downloaded."
        )

        return

    df = pd.DataFrame(
        all_news
    )

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(
        subset=["published_at"]
    )

    df = df.drop_duplicates(
        subset=[
            "ticker",
            "headline",
            "published_at"
        ]
    )

    df = df.sort_values(
        "published_at"
    ).reset_index(
        drop=True
    )

    output = (
        RAW_DIR /
        "latest_news.parquet"
    )

    df.to_parquet(
        output,
        index=False
    )

    print("\nDONE")
    print(
        f"Total articles: {len(df)}"
    )
    print(
        f"Date range: "
        f"{df['published_at'].min()} "
        f"to "
        f"{df['published_at'].max()}"
    )
    print(
        f"Saved to: {output}"
    )


if __name__ == "__main__":

    main()
