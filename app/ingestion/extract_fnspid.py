from pathlib import Path
import pandas as pd
import requests

TICKERS = {"NVDA", "TSLA", "AAPL", "MSFT", "META"}

URL = (
    "https://huggingface.co/datasets/"
    "Zihan1004/FNSPID/resolve/main/"
    "Stock_news/All_external.csv"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "fnspid_5stocks.parquet"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to FNSPID...")
    print("Streaming dataset in chunks...\n")

    response = requests.get(
        URL,
        stream=True,
        timeout=(30, 300)
    )

    response.raise_for_status()

    chunks = pd.read_csv(
        response.raw,
        chunksize=100_000,
        low_memory=False
    )

    matched_chunks = []
    total_rows = 0
    matched_rows = 0

    for chunk_number, chunk in enumerate(chunks, start=1):
        total_rows += len(chunk)

        filtered = chunk[
            chunk["Stock_symbol"]
            .astype(str)
            .str.upper()
            .isin(TICKERS)
        ].copy()

        if not filtered.empty:
            matched_chunks.append(filtered)
            matched_rows += len(filtered)

        if chunk_number % 10 == 0:
            print(
                f"Processed {total_rows:,} rows | "
                f"Matched {matched_rows:,}"
            )

    print("\nCombining results...")

    if not matched_chunks:
        print("No matching articles found.")
        return

    df = pd.concat(
        matched_chunks,
        ignore_index=True
    )

    columns = [
        "Date",
        "Article_title",
        "Stock_symbol",
        "Publisher",
        "Article",
    ]

    columns = [
        col for col in columns
        if col in df.columns
    ]

    df = df[columns].rename(
        columns={
            "Date": "published_at",
            "Article_title": "headline",
            "Stock_symbol": "ticker",
            "Publisher": "source",
            "Article": "article",
        }
    )

    df["ticker"] = df["ticker"].astype(str).str.upper()

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(
        subset=["published_at", "headline", "ticker"]
    )

    df = df.sort_values(
        ["published_at", "ticker"]
    ).reset_index(drop=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Articles: {len(df):,}")
    print(
        f"Date range: {df['published_at'].min()} "
        f"to {df['published_at'].max()}"
    )
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nArticles per ticker:")
    print(df["ticker"].value_counts())


if __name__ == "__main__":
    main()
