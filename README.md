# Market Intelligence

## GitHub release tracking

The ingestion pipeline can collect published releases from public GitHub repositories
using the GitHub REST API. It requires Python with pandas and pyarrow installed.

```sh
python -m app.ingestion.github_releases pytorch/pytorch ranaroussi/yfinance --limit 5
```

Run from the repository root. The command writes a Parquet snapshot to
`data/raw/github/latest_releases.parquet`. Use `--output` to choose another path.
An empty result or a failed API request leaves the previous snapshot in place.
