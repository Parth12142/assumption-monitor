"""Pick sentences from an EDGAR-CORPUS yearly zip file for hand labeling
(implementation plan, sections 6 and 15A)."""
from __future__ import annotations

import json
import random
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

from assumption_monitor.reader import split_with_offsets

# Parts of an annual report that talk about the future: business, risks, outlook, market risk.
SECTIONS = ["section_1", "section_1A", "section_7", "section_7A"]

CUE = re.compile(
    r"\b(?:assum\w*|expect\w*|project\w*|forecast\w*|estimat\w*|anticipat\w*|believ\w*)\b",
    re.IGNORECASE,
)
MIN_WORDS, MAX_WORDS = 6, 60
COLUMNS = ["sentence_id", "doc_id", "section", "sentence", "has_cue"]


class CorpusError(Exception):
    """A problem with the corpus file or with sampling from it."""


# ---------- Open the zip and read filings ----------
def open_zip(zip_path):
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise CorpusError(f"Could not open '{zip_path}': {exc}") from exc
    names = sorted(
        n for n in zf.namelist()
        if n.lower().endswith(".json") and not n.startswith("__MACOSX")
    )
    if not names:
        raise CorpusError("No .json filings were found inside the zip file.")
    return zf, names


def read_filing(zf, name):
    try:
        data = json.loads(zf.read(name))
    except (ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise CorpusError(f"Could not read {name}: {exc}") from exc
    if not isinstance(data, dict):
        raise CorpusError(f"{name} does not contain a JSON object.")
    return data


# ---------- Sentences ----------
def clean_sentences(text):
    """Split one section into sentences and keep only the ones that look like real sentences."""
    text = " ".join(str(text).split())
    kept = []
    for sentence, _, _ in split_with_offsets(text):
        n_words = len(sentence.split())
        if not (MIN_WORDS <= n_words <= MAX_WORDS):
            continue
        if not sentence[0].isupper() or not sentence.endswith("."):
            continue
        if sum(c.isalpha() for c in sentence) / len(sentence) < 0.7:   # mostly numbers: a table
            continue
        if sentence.isupper():                                          # a heading in capitals
            continue
        kept.append(sentence)
    return kept


def filing_sentences(data, doc_id):
    rows, seen = [], set()
    for section in SECTIONS:
        for sentence in clean_sentences(data.get(section) or ""):
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "sentence_id": f"{doc_id}-{len(rows) + 1:05d}",
                "doc_id": doc_id,
                "section": section,
                "sentence": sentence,
                "has_cue": bool(CUE.search(sentence)),
            })
    return rows


# ---------- Sampling ----------
def pick(rows, n, max_per_doc, rng):
    """Random sample of n rows, with at most max_per_doc from any one filing."""
    rows = list(rows)
    rng.shuffle(rows)
    chosen, per_doc = [], {}
    for row in rows:
        if per_doc.get(row["doc_id"], 0) >= max_per_doc:
            continue
        chosen.append(row)
        per_doc[row["doc_id"]] = per_doc.get(row["doc_id"], 0) + 1
        if len(chosen) == n:
            break
    return chosen


def build_candidates(zip_path, n_filings=60, n_cue=100, n_random=100, max_per_doc=6, seed=42,
                     exclude_docs=(), exclude_ids=()):
    """Return (table, stats): n_cue sentences with cue words and n_random without, mixed together.

    exclude_docs: filings that must not be used (for example the frozen test documents).
    exclude_ids: sentences that are already labeled.
    """
    rng = random.Random(seed)
    zf, names = open_zip(zip_path)
    names = list(names)
    rng.shuffle(names)

    pool, stats = [], {"filings_read": 0, "filings_skipped": 0}
    exclude_docs, exclude_ids = set(exclude_docs), set(exclude_ids)
    for name in names:
        if stats["filings_read"] >= n_filings:
            break
        if Path(name).stem in exclude_docs:
            continue
        try:
            rows = filing_sentences(read_filing(zf, name), Path(name).stem)
        except CorpusError:
            rows = []
        rows = [r for r in rows if r["sentence_id"] not in exclude_ids]
        if not rows:
            stats["filings_skipped"] += 1
            continue
        pool.extend(rows)
        stats["filings_read"] += 1

    cue = pick([r for r in pool if r["has_cue"]], n_cue, max_per_doc, rng)
    other = pick([r for r in pool if not r["has_cue"]], n_random, max_per_doc, rng)
    if len(cue) < n_cue or len(other) < n_random:
        raise CorpusError(
            f"Not enough sentences: found {len(cue)} with cue words and {len(other)} without "
            f"(wanted {n_cue} and {n_random}). Use more filings or a higher --max-per-doc."
        )

    chosen = cue + other
    rng.shuffle(chosen)
    stats["sentences_in_pool"] = len(pool)
    return pd.DataFrame(chosen, columns=COLUMNS), stats


# ---------- Command line ----------
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Sample sentences for hand labeling.")
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", default="data/labels/candidates.csv")
    parser.add_argument("--n-filings", type=int, default=60)
    parser.add_argument("--n-cue", type=int, default=100)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--max-per-doc", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="overwrite an existing candidates file")
    parser.add_argument("--exclude-docs", help="text file with document ids to leave out (frozen test documents)")
    parser.add_argument("--exclude-candidates", help="earlier candidates csv: its sentences are left out")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"ERROR: {out} already exists. Your labels refer to it. Use --force only if you start over.",
              file=sys.stderr)
        return 1
    exclude_docs, exclude_ids = set(), set()
    try:
        if args.exclude_docs:
            exclude_docs = {l.strip() for l in Path(args.exclude_docs).read_text().splitlines() if l.strip()}
        if args.exclude_candidates:
            exclude_ids = set(pd.read_csv(args.exclude_candidates, dtype=str)["sentence_id"])
    except (OSError, KeyError, pd.errors.ParserError) as exc:
        print(f"ERROR: could not read an exclude file: {exc}", file=sys.stderr)
        return 1
    try:
        table, stats = build_candidates(
            args.zip, args.n_filings, args.n_cue, args.n_random, args.max_per_doc, args.seed,
            exclude_docs, exclude_ids,
        )
    except CorpusError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Saved {len(table)} sentences from {table['doc_id'].nunique()} filings to {out}")
    print(f"Filings read: {stats['filings_read']}, skipped: {stats['filings_skipped']}, "
          f"sentences in pool: {stats['sentences_in_pool']}")
    print(table["has_cue"].value_counts().rename({True: "with cue word", False: "without cue word"}).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
