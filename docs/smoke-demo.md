# Offline smoke test

Actual output from the synthetic feature demo, not prediction metrics:

```text
Synthetic smoke demo: 90 input rows -> 50 feature rows
      Date ticker  return_1d  volatility_20d  label_up_1d
2020-01-21   DEMO   0.005901        0.002390            1
2020-01-22   DEMO   0.005479        0.002341            1
2020-01-23   DEMO   0.004773        0.002283            1
2020-01-24   DEMO   0.003862        0.002236            1
2020-01-25   DEMO   0.002844        0.002216            1
PASS: chronological feature engineering; no downloads or model benchmark.
```
