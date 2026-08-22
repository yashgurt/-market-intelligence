from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


CATEGORICAL_FEATURES = [
    "ticker",
]


def run_experiment(
    name,
    train,
    valid,
    test,
    target,
):

    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {name}")
    print("=" * 60)

    features = (
        MARKET_FEATURES
        + CATEGORICAL_FEATURES
    )

    X_train = train[features]
    y_train = train[target]

    X_valid = valid[features]
    y_valid = valid[target]

    X_test = test[features]
    y_test = test[target]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
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
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            MARKET_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
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

    def evaluate(name, X, y):

        probabilities = (
            pipeline.predict_proba(X)[:, 1]
        )

        predictions = pipeline.predict(X)

        accuracy = accuracy_score(
            y,
            predictions,
        )

        auc = roc_auc_score(
            y,
            probabilities,
        )

        print(f"\n{name}")
        print("-" * 50)
        print(f"Rows:     {len(y)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC:  {auc:.4f}")

        return accuracy, auc

    valid_accuracy, valid_auc = evaluate(
        "VALIDATION RESULTS",
        X_valid,
        y_valid,
    )

    test_accuracy, test_auc = evaluate(
        "TEST RESULTS",
        X_test,
        y_test,
    )

    return {
        "experiment": name,
        "valid_accuracy": valid_accuracy,
        "valid_auc": valid_auc,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
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

    print(f"\nTrain: {len(train)}")
    print(f"Valid: {len(valid)}")
    print(f"Test:  {len(test)}")

    results = []

    results.append(
        run_experiment(
            "LOGISTIC — 1 DAY DIRECTION",
            train,
            valid,
            test,
            "label_up_1d",
        )
    )

    results.append(
        run_experiment(
            "LOGISTIC — 5 DAY DIRECTION",
            train,
            valid,
            test,
            "label_up_5d",
        )
    )

    print("\n" + "=" * 60)
    print("HORIZON COMPARISON")
    print("=" * 60)

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
