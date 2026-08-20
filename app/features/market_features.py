from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def create_features(df, ticker):
    df = df.copy()

    # Make sure data is chronological
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["ticker"] = ticker

    # -------------------------
    # PAST-ONLY FEATURES
    # -------------------------

    # Previous returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # Moving averages
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()

    # Price position relative to moving averages
    df["price_vs_ma5"] = df["Close"] / df["ma_5"] - 1
    df["price_vs_ma20"] = df["Close"] / df["ma_20"] - 1

    # Volatility using ONLY previous 20 daily returns
    df["volatility_20d"] = (
        df["return_1d"]
        .rolling(20)
        .std()
    )

    # Volume compared with historical average
    df["volume_ma20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma20"]

    # Intraday range
    df["intraday_range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    # -------------------------
    # FUTURE LABELS
    # -------------------------
    # IMPORTANT:
    # These are labels, NOT model inputs.

    df["future_return_1d"] = (
        df["Close"].shift(-1) / df["Close"] - 1
    )

    df["future_return_5d"] = (
        df["Close"].shift(-5) / df["Close"] - 1
    )

    df["future_return_20d"] = (
        df["Close"].shift(-20) / df["Close"] - 1
    )

    # Classification labels
    df["label_up_1d"] = (
        df["future_return_1d"] > 0
    ).astype(int)

    df["label_up_5d"] = (
        df["future_return_5d"] > 0
    ).astype(int)

    # Remove rows where rolling features
    # or future labels do not exist
    df = df.dropna().reset_index(drop=True)

    return df


def build_dataset():
    all_data = []

    for path in sorted(RAW_DIR.glob("*.parquet")):
        ticker = path.stem

        print(f"Processing {ticker}...")

        df = pd.read_parquet(path)

        features = create_features(df, ticker)

        print(
            f"  Raw rows: {len(df)} | "
            f"Training rows: {len(features)}"
        )

        all_data.append(features)

    dataset = pd.concat(
        all_data,
        ignore_index=True
    )

    dataset = dataset.sort_values(
        ["Date", "ticker"]
    ).reset_index(drop=True)

    output = PROCESSED_DIR / "market_features.parquet"

    dataset.to_parquet(
        output,
        index=False
    )

    print("\nDONE")
    print(f"Total training rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Saved to: {output}")

    return dataset


if __name__ == "__main__":
    build_dataset()
