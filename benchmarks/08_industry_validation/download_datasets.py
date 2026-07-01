#!/usr/bin/env python3
"""
Download the LongMemEval and LoCoMo datasets used to benchmark MATHIR against
Mem0/Zep/Letta's published numbers.

Usage:
    python download_datasets.py --dataset longmemeval|locomo|both

LongMemEval (the "S" variant, cleaned):
    Source: HuggingFace `xiaowu0162/longmemeval-cleaned`, file
    `longmemeval_s_cleaned.json`. MIT license, no login required.
    Saved to: benchmarks/05_test_data/longmemeval/longmemeval_s_cleaned.json

LoCoMo:
    Source: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
    Single JSON file, no login required.
    Saved to: benchmarks/05_test_data/locomo/locomo10.json
    NOTE: LoCoMo's data is licensed CC BY-NC 4.0 (non-commercial). Fine for
    local benchmarking, but do not redistribute it or use it to make
    commercial scoring claims.

Both downloads are idempotent: if the target file already exists and looks
valid (non-empty, valid JSON), the download is skipped. On failure (network
error, 404, etc.) the script prints the exact URL that failed and exits
non-zero -- it never silently continues with missing data.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_ROOT = BENCHMARKS_ROOT / "05_test_data"

LONGMEMEVAL_REPO = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_FILE = "longmemeval_s_cleaned.json"
LONGMEMEVAL_DIR = TEST_DATA_ROOT / "longmemeval"
LONGMEMEVAL_PATH = LONGMEMEVAL_DIR / LONGMEMEVAL_FILE

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_DIR = TEST_DATA_ROOT / "locomo"
LOCOMO_PATH = LOCOMO_DIR / "locomo10.json"


def _is_valid_json_file(path: Path) -> bool:
    """A file is considered valid if it exists, is non-empty, and parses as JSON."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _download_via_urllib(url: str, dest: Path) -> None:
    """Direct HTTPS download fallback (no huggingface_hub / requests)."""
    req = urllib.request.Request(url, headers={"User-Agent": "mathir-benchmark/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        print(f"[ERROR] Download failed: HTTP {e.code} for URL: {url}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[ERROR] Download failed: could not reach URL: {url} ({e})", file=sys.stderr)
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def download_longmemeval() -> None:
    print(f"[longmemeval] target: {LONGMEMEVAL_PATH}")
    if _is_valid_json_file(LONGMEMEVAL_PATH):
        size_mb = LONGMEMEVAL_PATH.stat().st_size / (1024 * 1024)
        print(f"[longmemeval] already downloaded ({size_mb:.1f} MB), skipping.")
        _sanity_check_longmemeval()
        return

    LONGMEMEVAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        print(
            f"[longmemeval] downloading {LONGMEMEVAL_FILE} from HF repo "
            f"{LONGMEMEVAL_REPO} via huggingface_hub..."
        )
        try:
            local_path = hf_hub_download(
                repo_id=LONGMEMEVAL_REPO,
                filename=LONGMEMEVAL_FILE,
                repo_type="dataset",
            )
        except Exception as e:
            url = f"https://huggingface.co/datasets/{LONGMEMEVAL_REPO}/resolve/main/{LONGMEMEVAL_FILE}"
            print(f"[ERROR] huggingface_hub download failed for URL: {url} ({e})", file=sys.stderr)
            sys.exit(1)

        import shutil

        shutil.copyfile(local_path, LONGMEMEVAL_PATH)
    except ImportError:
        url = f"https://huggingface.co/datasets/{LONGMEMEVAL_REPO}/resolve/main/{LONGMEMEVAL_FILE}"
        print(
            f"[longmemeval] huggingface_hub not installed, falling back to "
            f"direct HTTPS request: {url}"
        )
        _download_via_urllib(url, LONGMEMEVAL_PATH)

    if not _is_valid_json_file(LONGMEMEVAL_PATH):
        print(
            f"[ERROR] Downloaded file at {LONGMEMEVAL_PATH} is empty or not "
            f"valid JSON. Download did not succeed.",
            file=sys.stderr,
        )
        sys.exit(1)

    size_mb = LONGMEMEVAL_PATH.stat().st_size / (1024 * 1024)
    print(f"[longmemeval] downloaded {size_mb:.1f} MB -> {LONGMEMEVAL_PATH}")
    _sanity_check_longmemeval()


def _sanity_check_longmemeval() -> None:
    try:
        with LONGMEMEVAL_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        n = len(data) if isinstance(data, list) else len(data.get("data", data))
        print(f"[longmemeval] schema check OK -- loaded {n} question instances")
    except Exception as e:
        print(f"[WARN] longmemeval schema sanity check failed: {e}", file=sys.stderr)


def download_locomo() -> None:
    print(f"[locomo] target: {LOCOMO_PATH}")
    if _is_valid_json_file(LOCOMO_PATH):
        size_mb = LOCOMO_PATH.stat().st_size / (1024 * 1024)
        print(f"[locomo] already downloaded ({size_mb:.1f} MB), skipping.")
        _sanity_check_locomo()
        return

    LOCOMO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[locomo] downloading from {LOCOMO_URL} ...")

    try:
        import requests

        resp = requests.get(LOCOMO_URL, timeout=120)
        if resp.status_code != 200:
            print(
                f"[ERROR] Download failed: HTTP {resp.status_code} for URL: {LOCOMO_URL}",
                file=sys.stderr,
            )
            sys.exit(1)
        LOCOMO_PATH.write_bytes(resp.content)
    except ImportError:
        _download_via_urllib(LOCOMO_URL, LOCOMO_PATH)

    if not _is_valid_json_file(LOCOMO_PATH):
        print(
            f"[ERROR] Downloaded file at {LOCOMO_PATH} is empty or not valid "
            f"JSON. Download did not succeed. URL: {LOCOMO_URL}",
            file=sys.stderr,
        )
        sys.exit(1)

    size_mb = LOCOMO_PATH.stat().st_size / (1024 * 1024)
    print(f"[locomo] downloaded {size_mb:.1f} MB -> {LOCOMO_PATH}")
    print("[locomo] NOTE: data is CC BY-NC 4.0 (non-commercial) -- do not redistribute.")
    _sanity_check_locomo()


def _sanity_check_locomo() -> None:
    try:
        with LOCOMO_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        n = len(data) if isinstance(data, list) else len(data)
        print(f"[locomo] schema check OK -- loaded {n} conversations")
    except Exception as e:
        print(f"[WARN] locomo schema sanity check failed: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LongMemEval / LoCoMo benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["longmemeval", "locomo", "both"],
        default="both",
        help="Which dataset(s) to download",
    )
    args = parser.parse_args()

    if args.dataset in ("longmemeval", "both"):
        download_longmemeval()
    if args.dataset in ("locomo", "both"):
        download_locomo()


if __name__ == "__main__":
    main()
