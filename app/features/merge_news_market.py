from pathlib import Path

import pandas as pd


MARKET_FILE = Path("data/processed/market_features.parquet")
NEWS_FILE = Path("data/processed/daily_news_features.parquet")

OUTPUT_FILE = Path(
    "data/processed/market_news_features.parquet"
)


def main():

    print("Loading market features...")

    market = pd.read_parquet(MARKET_FILE)

    print(f"Market rows: {len(market):,}")

    print("\nLoading news features...")

    news = pd.read_parquet(NEWS_FILE)

    print(f"News ticker-days: {len(news):,}")

    # --------------------------------------------------
    # NORMALIZE DATES
    # --------------------------------------------------

    market["Date"] = pd.to_datetime(
        market["Date"]
    ).dt.normalize()

    news["Date"] = pd.to_datetime(
        news["Date"]
    ).dt.normalize()

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    print("\nMerging market + news data...")

    df = market.merge(
        news,
        on=["Date", "ticker"],
        how="left"
    )

    # --------------------------------------------------
    # DAYS WITH NO NEWS
    # --------------------------------------------------

    sentiment_columns = [
        "news_count",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_score"
    ]

    df["news_available"] = (
        df["news_count"]
        .notna()
        .astype(int)
    )

    # No-news day:
    # count = 0
    # sentiment values = 0

    df["news_count"] = (
        df["news_count"]
        .fillna(0)
    )

    for column in sentiment_columns[1:]:

        df[column] = (
            df[column]
            .fillna(0)
        )

    # --------------------------------------------------
    # SORT
    # --------------------------------------------------

    df = df.sort_values(
        ["Date", "ticker"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"Final rows: {len(df):,}")

    print(
        f"Rows with news: "
        f"{df['news_available'].sum():,}"
    )

    print(
        f"Rows without news: "
        f"{(df['news_available'] == 0).sum():,}"
    )

    print("\nNews coverage by ticker:")

    print(
        df.groupby("ticker")[
            "news_available"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print("\nFinal columns:")

    print(df.columns.tolist())

    print("\nSample:")

    columns_to_show = [

        "Date",
        "ticker",
        "Close",
        "return_1d",

        "news_count",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_score",

        "future_return_1d",
        "label_up_1d"
    ]

    print(
        df[
            columns_to_show
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
