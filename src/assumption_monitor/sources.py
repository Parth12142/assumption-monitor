"""Source registry, safe downloads and checksums (implementation plan, sections 5 and 15)."""
from __future__ import annotations

import subprocess
import csv
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ALLOWED_HOSTS = {"fred.stlouisfed.org"}
REGISTRY_PATH = Path("config/source_registry.json")
RAW_DIR = Path("data/raw/fred")
LOG_PATH = Path("data/registry/download_log.csv")

REQUIRED_FIELDS = [
    "series_id", "title", "provider", "url", "units", "currency", "geography",
    "frequency", "publication_pattern", "approved", "permitted_use",
]
LOG_COLUMNS = [
    "series_id", "file", "retrieved_at_utc", "source_url", "sha256", "n_bytes",
    "n_rows", "first_date", "last_date",
]


class SourceError(Exception):
    """A problem with the registry, a download or a stored file."""


# ---------- Small helpers ----------
def utc_now():
    return datetime.now(timezone.utc)


def stamp(dt):
    return dt.strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fred_csv_url(series_id):
    if not re.fullmatch(r"[A-Z0-9_]+", series_id):
        raise SourceError(f"Invalid series id: {series_id!r}")
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def check_allowed(url):
    parts = urllib.parse.urlparse(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise SourceError(f"Download blocked: {url} is not an approved https source.")


# ---------- Registry ----------
def validate_registry(df):
    problems = []

    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            problems.append(f"missing column: {field}")

    if problems:
        return problems

    if df["series_id"].duplicated().any():
        problems.append("duplicate series_id")

    for _, row in df.iterrows():
        series_id = row["series_id"]
        approved = str(row["approved"]).strip().lower()
        terms = str(row["terms_checked_on"]).strip()

        # Check all required fields except terms_checked_on
        for field in REQUIRED_FIELDS:
            if field in {"approved", "terms_checked_on"}:
                continue

            if not str(row[field]).strip():
                problems.append(f"{series_id}: empty {field}")

        # approved must be yes or no
        if approved not in {"yes", "no"}:
            problems.append(f"{series_id}: approved must be yes or no")

        # terms_checked_on is required when approved
        if approved == "yes" and not terms:
            problems.append(
                f"{series_id}: approved but terms_checked_on is empty"
            )

    return problems


def load_registry(path=REGISTRY_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"Could not read registry {path}: {exc}") from exc
    df = pd.DataFrame(data)
    problems = validate_registry(df)
    if problems:
        raise SourceError("Registry problems: " + "; ".join(problems))
    return df


def get_entry(registry, series_id):
    match = registry[registry["series_id"] == series_id]
    if match.empty:
        raise SourceError(f"{series_id} is not in the source registry.")
    return match.iloc[0].to_dict()


# ---------- Reading and profiling a series file ----------
def read_series_file(path):
    df = pd.read_csv(path, dtype=str)
    if df.shape[1] < 2:
        raise SourceError(f"{path}: expected at least two columns (date, value).")
    return df, df.columns[0], df.columns[1]


def profile_series(path):
    df, date_col, value_col = read_series_file(path)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    raw = df[value_col].fillna("").str.strip()
    values = pd.to_numeric(raw, errors="coerce")
    is_marker = raw.isin([".", "", "nan", "NaN"])
    return {
        "file": str(path),
        "date_column": date_col,
        "value_column": value_col,
        "n_rows": int(len(df)),
        "first_date": str(dates.min().date()) if dates.notna().any() else "",
        "last_date": str(dates.max().date()) if dates.notna().any() else "",
        "n_missing": int(is_marker.sum()),
        "n_bad_values": int((values.isna() & ~is_marker).sum()),
        "n_bad_dates": int(dates.isna().sum()),
        "n_duplicate_dates": int(dates.dropna().duplicated().sum()),
        "is_sorted": bool(dates.dropna().is_monotonic_increasing),
        "min_value": float(values.min()) if values.notna().any() else None,
        "max_value": float(values.max()) if values.notna().any() else None,
    }


# ---------- Download log ----------
def _append_log(record, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)


def _log_record(series_id, path, url, retrieved_at):
    prof = profile_series(path)
    return {
        "series_id": series_id,
        "file": str(path),
        "retrieved_at_utc": retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": url,
        "sha256": sha256_file(path),
        "n_bytes": Path(path).stat().st_size,
        "n_rows": prof["n_rows"],
        "first_date": prof["first_date"],
        "last_date": prof["last_date"],
    }


def read_log(log_path=LOG_PATH):
    if not Path(log_path).exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(log_path, dtype=str).fillna("")


def verify_log(log_path=LOG_PATH):
    """Recompute every checksum. Returns a list of problems (empty list means all good)."""
    problems = []
    for _, row in read_log(log_path).iterrows():
        path = Path(row["file"])
        if not path.exists():
            problems.append(f"missing file: {path}")
        elif sha256_file(path) != row["sha256"]:
            problems.append(f"checksum changed: {path}")
    return problems


# ---------- Getting files ----------
def _fetch(url, timeout=30, retries=2, pause=2):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading: {url}")
            print(f"Attempt {attempt}/{retries}...")

            result = subprocess.run(
                [
                    "curl",
                    "--http1.1",
                    "-L",
                    "--max-time", str(timeout),
                    "-A", "Mozilla/5.0",
                    url
                ],
                capture_output=True,
                timeout=timeout + 5
            )

            if result.returncode != 0:
                raise RuntimeError(
                    result.stderr.decode(errors="replace").strip()
                )

            data = result.stdout

            print(f"Downloaded {len(data)} bytes")
            return data

        except Exception as exc:
            last_error = exc
            print(f"Attempt {attempt} failed: {exc}")

            if attempt < retries:
                print(f"Retrying in {pause} seconds...")
                time.sleep(pause)

    raise SourceError(
        f"Download failed after {retries} tries: {last_error}. "
        "Check your network or FRED connection."
    )

def download_fred(
    series_id,
    registry,
    raw_dir=RAW_DIR,
    log_path=LOG_PATH,
    timeout=60,
    retries=2,
    pause=2
):
    entry = get_entry(registry, series_id)
    if str(entry["approved"]).strip() != "yes":
        raise SourceError(
            f"{series_id} is not approved. Read its terms, then set approved to yes "
            "and fill terms_checked_on in config/source_registry.json."
        )
    url = fred_csv_url(series_id)
    check_allowed(url)

    data = _fetch(url, timeout, retries, pause)
    head = data[:200].lower()
    if not data or b"<html" in head or b"<!doctype" in head:
        raise SourceError("The server returned a web page instead of CSV data.")

    retrieved_at = utc_now()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{series_id}_{stamp(retrieved_at)}.csv"
    path.write_bytes(data)

    record = _log_record(series_id, path, url, retrieved_at)
    _append_log(record, log_path)
    return record


def register_file(series_id, source_path, registry, raw_dir=RAW_DIR, log_path=LOG_PATH, retrieved_at=None):
    """Keep a copy of a CSV you downloaded by hand, and log its checksum."""
    entry = get_entry(registry, series_id)
    source_path = Path(source_path)
    if not source_path.exists():
        raise SourceError(f"File not found: {source_path}")
    if retrieved_at is None:
        retrieved_at = datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc)

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{series_id}_{stamp(retrieved_at)}.csv"
    shutil.copy2(source_path, path)

    record = _log_record(series_id, path, entry["url"], retrieved_at)
    _append_log(record, log_path)
    return record


def latest_file(series_id, log_path=LOG_PATH):
    log = read_log(log_path)
    rows = log[log["series_id"] == series_id]
    if rows.empty:
        raise SourceError(f"No stored file for {series_id}. Run download or register-file first.")
    return Path(rows.sort_values("retrieved_at_utc").iloc[-1]["file"])


def load_series(series_id, log_path=LOG_PATH):
    """Latest stored file as a float Series indexed by date; missing markers become NaN."""
    path = latest_file(series_id, log_path)
    df, date_col, value_col = read_series_file(path)
    index = pd.to_datetime(df[date_col])
    values = pd.to_numeric(df[value_col].fillna("").str.strip(), errors="coerce")
    return pd.Series(values.to_numpy(), index=index, name=series_id).sort_index()


# ---------- Command line ----------
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Manage indicator sources.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("verify")
    p = sub.add_parser("download"); p.add_argument("series", nargs="+")
    p = sub.add_parser("register-file"); p.add_argument("series"); p.add_argument("path")
    p = sub.add_parser("profile"); p.add_argument("series")
    args = parser.parse_args(argv)

    try:
        registry = load_registry()
        if args.command == "list":
            cols = ["series_id", "units", "frequency", "approved", "terms_checked_on"]
            print(registry[cols].to_string(index=False))
        elif args.command == "download":
            for sid in args.series:
                rec = download_fred(sid, registry)
                print(f"{sid}: saved {rec['file']} ({rec['n_rows']} rows, {rec['first_date']} to {rec['last_date']})")
        elif args.command == "register-file":
            rec = register_file(args.series, args.path, registry)
            print(f"{args.series}: copied to {rec['file']}, sha256 {rec['sha256'][:12]}...")
        elif args.command == "profile":
            for key, value in profile_series(latest_file(args.series)).items():
                print(f"{key}: {value}")
        elif args.command == "verify":
            problems = verify_log()
            print("All checksums match." if not problems else "\n".join(problems))
            return 1 if problems else 0
    except SourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
