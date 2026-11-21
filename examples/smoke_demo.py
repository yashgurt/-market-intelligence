"""Offline smoke demo of the existing feature-engineering function, not a market forecast."""
import numpy as np
import pandas as pd
from app.features.market_features import create_features

def main():
    n = 90
    t = np.arange(n)
    close = 100 + 0.3 * t + np.sin(t / 3)
    prices = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=n),
                           "Close": close, "High": close + 1, "Low": close - 1,
                           "Volume": 1000 + 7 * t})
    features = create_features(prices, "DEMO")
    assert len(features) == 50
    assert features["Date"].is_monotonic_increasing
    assert features["ticker"].eq("DEMO").all()
    assert features["future_return_1d"].notna().all()
    print("Synthetic smoke demo: 90 input rows -> 50 feature rows")
    print(features[["Date", "ticker", "return_1d", "volatility_20d", "label_up_1d"]].head(5).to_string(index=False))
    print("PASS: chronological feature engineering; no downloads or model benchmark.")

if __name__ == "__main__":
    main()
