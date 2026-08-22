from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


DATA_PATH = Path(
    "data/processed/market_news_features.parquet"
)


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


def evaluate(name, pipeline, X, y):

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = pipeline.predict(X)

    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, probabilities)

    print(f"\n{name}")
    print("-" * 55)
    print(f"Rows:     {len(y)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

    return {
        "accuracy": accuracy,
        "auc": auc,
    }


def run_experiment(
    name,
    train,
    valid,
    test,
    numeric_features,
):

    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {name}")
    print("=" * 60)

    features = (
        numeric_features
        + CATEGORICAL_FEATURES
    )

    X_train = train[features]
    y_train = train["label_up_1d"]

    X_valid = valid[features]
    y_valid = valid["label_up_1d"]

    X_test = test[features]
    y_test = test["label_up_1d"]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            model,
        ),
    ])

    print("\nTraining...")

    pipeline.fit(
        X_train,
        y_train,
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
        "experiment": name,
        "valid_accuracy": valid_results["accuracy"],
        "valid_auc": valid_results["auc"],
        "test_accuracy": test_results["accuracy"],
        "test_auc": test_results["auc"],
    }


def main():

    print("\nLoading dataset...")

    df = pd.read_parquet(DATA_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        ["Date", "ticker"]
    ).reset_index(drop=True)

    print(f"Rows: {len(df)}")

    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    train = df[
        df["Date"] < "2018-01-01"
    ].copy()

    valid = df[
        (df["Date"] >= "2018-01-01")
        &
        (df["Date"] < "2019-01-01")
    ].copy()

    test = df[
        df["Date"] >= "2019-01-01"
    ].copy()

    print("\nDATA SPLIT")
    print(f"Train: {len(train)}")
    print(f"Valid: {len(valid)}")
    print(f"Test:  {len(test)}")

    results = []

    results.append(
        run_experiment(
            "TREE — MARKET ONLY",
            train,
            valid,
            test,
            MARKET_FEATURES,
        )
    )

    results.append(
        run_experiment(
            "TREE — MARKET + NEWS",
            train,
            valid,
            test,
            MARKET_FEATURES + NEWS_FEATURES,
        )
    )

    print("\n" + "=" * 60)
    print("FINAL TREE MODEL COMPARISON")
    print("=" * 60)

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
