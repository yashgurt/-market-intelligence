from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


INPUT_FILE = Path("data/processed/fnspid_5stocks.parquet")
OUTPUT_FILE = Path("data/processed/daily_news_features.parquet")

MODEL_NAME = "ProsusAI/finbert"

BATCH_SIZE = 16


def main():

    print("Loading news data...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Articles loaded: {len(df):,}")

    # Convert timestamp
    df["published_at"] = pd.to_datetime(
        df["published_at"],
        utc=True
    )

    # Convert article timestamp to trading date
    df["Date"] = (
        df["published_at"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Use headline as text
    df["text"] = df["headline"].fillna("").astype(str)

    df = df[df["text"].str.len() > 0].copy()

    print(f"Usable headlines: {len(df):,}")

    print("\nLoading FinBERT...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model.to(device)

    print(f"Using device: {device}")

    sentiments = []

    total_batches = (
        len(df) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        f"\nProcessing {len(df):,} headlines "
        f"in {total_batches:,} batches..."
    )

    for start in range(
        0,
        len(df),
        BATCH_SIZE
    ):

        batch = df["text"].iloc[
            start:start + BATCH_SIZE
        ].tolist()

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        sentiments.extend(
            probabilities.cpu().numpy()
        )

        current_batch = (
            start // BATCH_SIZE
        ) + 1

        if (
            current_batch % 20 == 0
            or current_batch == total_batches
        ):

            print(
                f"Processed batch "
                f"{current_batch:,}/{total_batches:,}"
            )

    sentiment_df = pd.DataFrame(
        sentiments,
        columns=[
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral"
        ]
    )

    df = df.reset_index(drop=True)

    df = pd.concat(
        [
            df,
            sentiment_df
        ],
        axis=1
    )

    print("\nAggregating by ticker and date...")

    daily = (
        df.groupby(
            ["Date", "ticker"]
        )
        .agg(
            news_count=(
                "headline",
                "count"
            ),

            sentiment_positive=(
                "sentiment_positive",
                "mean"
            ),

            sentiment_negative=(
                "sentiment_negative",
                "mean"
            ),

            sentiment_neutral=(
                "sentiment_neutral",
                "mean"
            )
        )
        .reset_index()
    )

    # Create one overall sentiment score
    daily["sentiment_score"] = (
        daily["sentiment_positive"]
        -
        daily["sentiment_negative"]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    daily.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 50)
    print("DONE")
    print("=" * 50)

    print(
        f"Daily ticker observations: "
        f"{len(daily):,}"
    )

    print(
        f"Date range: "
        f"{daily['Date'].min()} "
        f"to "
        f"{daily['Date'].max()}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nArticles aggregated per ticker:")

    print(
        df["ticker"]
        .value_counts()
    )

    print("\nSample:")

    print(
        daily.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
