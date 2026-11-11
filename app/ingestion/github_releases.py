"""Collect public GitHub releases into a Parquet snapshot."""

import argparse
import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_REPOS = ("pytorch/pytorch", "huggingface/transformers", "ranaroussi/yfinance")
OUTPUT = Path("data/raw/github/latest_releases.parquet")
REPOSITORY = re.compile(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+")


def fetch_releases(repo, limit):
    if not REPOSITORY.fullmatch(repo) or repo.split("/")[1] in {".", ".."}:
        raise ValueError("expected a repository in owner/name format")

    request = Request(
        f"https://api.github.com/repos/{repo}/releases?per_page={limit}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "market-intelligence",
        },
    )
    with urlopen(request, timeout=20) as response:
        releases = json.load(response)
    if not isinstance(releases, list) or any(
        not isinstance(release, dict) for release in releases
    ):
        raise ValueError("unexpected GitHub release response")

    return [
        {
            "repo": repo,
            "tag_name": release["tag_name"],
            "name": release.get("name") or release["tag_name"],
            "published_at": release["published_at"],
            "prerelease": release["prerelease"],
            "url": release["html_url"],
        }
        for release in releases
        if release.get("published_at") and not release.get("draft")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repos", nargs="*", metavar="OWNER/REPO")
    parser.add_argument("--limit", type=int, default=10, help="Releases per repo (1–100)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")

    rows = []
    failed = False
    for repo in dict.fromkeys(args.repos or DEFAULT_REPOS):
        try:
            releases = fetch_releases(repo, args.limit)
        except HTTPError as error:
            if error.code == 429 or (
                error.code == 403
                and (error.headers.get("X-RateLimit-Remaining") == "0"
                     or error.headers.get("Retry-After"))
            ):
                print(f"{repo}: GitHub rate limit reached. Try again later.", file=sys.stderr)
                return 2
            print(f"{repo}: GitHub returned HTTP {error.code}", file=sys.stderr)
            failed = True
        except (URLError, TimeoutError, ValueError, KeyError) as error:
            print(f"{repo}: {error}", file=sys.stderr)
            failed = True
        else:
            print(f"{repo}: {len(releases)} releases")
            rows.extend(releases)

    if failed:
        return 1
    if not rows:
        print("No published releases found.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        snapshot = Path(directory) / "releases.parquet"
        pd.DataFrame(rows).to_parquet(snapshot, index=False)
        snapshot.replace(args.output)
    print(f"Wrote {len(rows)} releases to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
