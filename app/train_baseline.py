from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================
# DATA
# ============================================================

DATA_PATH = Path(
    "data/processed/market_news_features.parquet"
)


# ============================================================
# FEATURE GROUPS
# ============================================================

MARKET_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "price_vs_ma5",
    "price_vs_ma20",
    "volatility_20d",
    "volume_ratio",
    "intraday_range",
]


NEWS_FEATURES = [
    "news_count",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "sentiment_score",
]


CATEGORICAL_FEATURES = [
    "ticker",
]


TARGET = "label_up_1d"


# ============================================================
# PREPROCESSING + MODEL
# ============================================================

def build_pipeline(numeric_features):

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
    ])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        ),
    ])

    return pipeline


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, pipeline, X, y):

    if len(X) == 0:
        print(f"\n{name}")
        print("No rows available.")
        return {
            "accuracy": None,
            "auc": None,
            "rows": 0,
        }

    probabilities = pipeline.predict_proba(X)[:, 1]

    predictions = pipeline.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    print(f"\n{name}")

    print("-" * 55)

    print(f"Rows:     {len(X)}")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"ROC-AUC:  {auc:.4f}")

    return {
        "accuracy": accuracy,
        "auc": auc,
        "rows": len(X),
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    experiment_name,
    train,
    valid,
    test,
    numeric_features,
):

    print()

    print("=" * 60)

    print(f"EXPERIMENT: {experiment_name}")

    print("=" * 60)

    features = (
        numeric_features
        +
        CATEGORICAL_FEATURES
    )

    X_train = train[features]

    y_train = train[TARGET]

    X_valid = valid[features]

    y_valid = valid[TARGET]

    X_test = test[features]

    y_test = test[TARGET]

    pipeline = build_pipeline(
        numeric_features
    )

    print("\nTraining...")

    pipeline.fit(
        X_train,
        y_train
    )

    valid_results = evaluate(
        "VALIDATION RESULTS",
        pipeline,
        X_valid,
        y_valid,
    )

    test_results = evaluate(
        "TEST RESULTS",
        pipeline,
        X_test,
        y_test,
    )

    return {
        "experiment": experiment_name,

        "train_rows": len(train),

        "valid_rows": valid_results["rows"],

        "test_rows": test_results["rows"],

        "valid_accuracy":
            valid_results["accuracy"],

        "valid_auc":
            valid_results["auc"],

        "test_accuracy":
            test_results["accuracy"],

        "test_auc":
            test_results["auc"],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading dataset...")

    df = pd.read_parquet(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        ["Date", "ticker"]
    ).reset_index(drop=True)


    print(f"Rows: {len(df)}")

    print(
        "Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )


    # ========================================================
    # CHRONOLOGICAL SPLIT
    #
    # Train: 2011-2017
    # Valid: 2018
    # Test:  2019-2020
    # ========================================================

    train_all = df[
        df["Date"] < "2018-01-01"
    ].copy()

    valid_all = df[
        (
            df["Date"] >= "2018-01-01"
        )
        &
        (
            df["Date"] < "2019-01-01"
        )
    ].copy()

    test_all = df[
        df["Date"] >= "2019-01-01"
    ].copy()


    print("\nDATA SPLIT")

    print(f"Train: {len(train_all)} rows")

    print(f"Valid: {len(valid_all)} rows")

    print(f"Test:  {len(test_all)} rows")


    # ========================================================
    # NEWS-COVERED SUBSETS
    # ========================================================

    train_news = train_all[
        train_all["news_available"] == 1
    ].copy()

    valid_news = valid_all[
        valid_all["news_available"] == 1
    ].copy()

    test_news = test_all[
        test_all["news_available"] == 1
    ].copy()


    print("\nNEWS-COVERED ROWS")

    print(
        f"Train with news: {len(train_news)}"
    )

    print(
        f"Valid with news: {len(valid_news)}"
    )

    print(
        f"Test with news:  {len(test_news)}"
    )


    print("\nNEWS COVERAGE BY TICKER")

    print(
        df.groupby("ticker")[
            "news_available"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    # ========================================================
    # EXPERIMENTS
    # ========================================================

    results = []


    # --------------------------------------------------------
    # 1. MARKET ONLY — ALL ROWS
    # --------------------------------------------------------

    result = run_experiment(

        "MARKET ONLY — ALL ROWS",

        train_all,

        valid_all,

        test_all,

        MARKET_FEATURES,

    )

    results.append(result)


    # --------------------------------------------------------
    # 2. MARKET ONLY — NEWS-COVERED ROWS
    #
    # FAIR BASELINE FOR NEWS EXPERIMENTS
    # --------------------------------------------------------

    result = run_experiment(

        "MARKET ONLY — NEWS-COVERED ROWS",

        train_news,

        valid_news,

        test_news,

        MARKET_FEATURES,

    )

    results.append(result)


    # --------------------------------------------------------
    # 3. NEWS ONLY — NEWS-COVERED ROWS
    # --------------------------------------------------------

    result = run_experiment(

        "NEWS ONLY — NEWS-COVERED ROWS",

        train_news,

        valid_news,

        test_news,

        NEWS_FEATURES,

    )

    results.append(result)


    # --------------------------------------------------------
    # 4. MARKET + NEWS — NEWS-COVERED ROWS
    # --------------------------------------------------------

    result = run_experiment(

        "MARKET + NEWS — NEWS-COVERED ROWS",

        train_news,

        valid_news,

        test_news,

        MARKET_FEATURES
        +
        NEWS_FEATURES,

    )

    results.append(result)


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    print()

    print("=" * 60)

    print("FINAL COMPARISON")

    print("=" * 60)


    print(

        results_df[
            [

                "experiment",

                "train_rows",

                "valid_rows",

                "test_rows",

                "valid_accuracy",

                "valid_auc",

                "test_accuracy",

                "test_auc",

            ]

        ]

        .to_string(

            index=False,

            float_format=lambda x:
                f"{x:.4f}"

        )

    )


if __name__ == "__main__":

    main()
