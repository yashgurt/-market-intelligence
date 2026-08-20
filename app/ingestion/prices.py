from pathlib import Path

import yfinance as yf


TICKERS = ["NVDA", "TSLA", "AAPL"]

RAW_DIR = Path("data/raw")

RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_prices(
    tickers=TICKERS,
    start="2011-01-01",
    end="2020-07-01",
):

    for ticker in tickers:

        print(f"\nDownloading {ticker}...")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            print(f"WARNING: No data returned for {ticker}")
            continue

        if hasattr(df.columns, "levels"):
            df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in df.columns
            ]

        df = df.loc[:, ~df.columns.duplicated()]
        df = df.reset_index()

        output = RAW_DIR / f"{ticker}.parquet"

        df.to_parquet(output, index=False)

        print(f"Saved {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"Location: {output}")


if __name__ == "__main__":
    download_prices()
