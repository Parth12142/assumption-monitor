"""Measure how long a text-model workload takes on THIS computer, and how much memory it uses.

Workload: make n synthetic sentences, turn them into TF-IDF features, train logistic regression.
Each size runs in its own fresh Python process so its memory peak is measured on its own.
"""
import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SIZES = [10000, 20000, 30000, 40000, 60000, 80000, 100000, 150000, 200000, 250000, 300000]


def machine_info():
    mem_total = mem_available = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                mem_total = int(line.split()[1]) / 1024        # MB
            if line.startswith("MemAvailable"):
                mem_available = int(line.split()[1]) / 1024    # MB
    return {
        "cpu_count": os.cpu_count(),
        "ram_total_mb": round(mem_total),
        "ram_available_mb": round(mem_available),
        "python": sys.version.split()[0],
    }


def run_workload(n):
    """Runs inside the worker process: returns (seconds, peak_memory_mb)."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    vocab = np.array([f"word{i}" for i in range(5000)])
    start = time.perf_counter()

    lengths = rng.integers(10, 25, size=n)
    sentences = [" ".join(rng.choice(vocab, size=k)) for k in lengths]
    cue_words = set(vocab[:50])
    labels = np.array([int(any(w in cue_words for w in s.split())) for s in sentences])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    features = vectorizer.fit_transform(sentences)
    LogisticRegression(max_iter=1000, class_weight="balanced").fit(features, labels)

    seconds = time.perf_counter() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux: kilobytes
    return seconds, peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--worker", type=int, help="internal: run one size and print the result")
    args = parser.parse_args()

    if args.worker:
        seconds, peak_mb = run_workload(args.worker)
        print(json.dumps({"n": args.worker, "seconds": seconds, "peak_mb": peak_mb}))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = machine_info()
    (out_dir / "pc_info.json").write_text(json.dumps(info, indent=2))
    print("This computer:", info)

    rows = []
    for n in sorted(args.sizes):
        for repeat in range(1, args.repeats + 1):
            result = subprocess.run(
                [sys.executable, __file__, "--worker", str(n)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"n={n}: FAILED (out of memory or an error)\n{result.stderr[-300:]}")
                break
            row = json.loads(result.stdout.strip().splitlines()[-1])
            row["repeat"] = repeat
            rows.append(row)
            print(f"n={n:>6}  repeat {repeat}: {row['seconds']:7.1f} s   {row['peak_mb']:7.0f} MB")

    with open(out_dir / "benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "repeat", "seconds", "peak_mb"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", out_dir / "benchmark.csv")


if __name__ == "__main__":
    main()
