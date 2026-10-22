"""Model A baseline: is this sentence an assumption? (implementation plan, section 7)

TF-IDF word features + logistic regression, checked with document-separated splits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from assumption_monitor import autolabel, labeler

LABELS = "data/labels/labels.csv"
SILVER_AUTO = "data/labels/auto_day9.csv"
CANDIDATES = ["data/labels/candidates.csv", "data/labels/candidates_day9.csv"]
FROZEN_DOCS = "data/labels/frozen_test_docs.txt"
DATASET = "data/model_a/dataset.csv"
MODEL_PATH = "models/model_a_baseline.joblib"
TEST_REPORT = "reports/model_a_test_metrics.json"
MIN_HAND_LABELS, MIN_PER_CLASS = 30, 5


class ModelError(Exception):
    """A problem with the data or the model."""


# ---------- Step 1: one table with every labeled sentence ----------
def build_dataset(labels_path=LABELS, silver_auto_path=SILVER_AUTO, candidates_paths=CANDIDATES,
                  frozen_docs_path=FROZEN_DOCS):
    """source: hand = you labeled it alone, audited = you checked a rule label, rules = rule label only."""
    labels = labeler.load_labels(labels_path)
    done = labels[labels["is_assumption"].isin(["0", "1"])]
    if done.empty:
        raise ModelError(f"No labels found in {labels_path}. Label some sentences first.")
    parts = [
        pd.DataFrame({
            "sentence_id": done["sentence_id"], "doc_id": done["doc_id"],
            "y": done["is_assumption"].astype(int),
            "source": np.where(done["review_status"] == "single", "hand", "audited"),
        })
    ]
    if silver_auto_path and Path(silver_auto_path).exists():
        auto = autolabel.load_auto(silver_auto_path)
        auto = auto[(auto["review_status"] == "auto") & ~auto["sentence_id"].isin(done["sentence_id"])]
        parts.append(pd.DataFrame({
            "sentence_id": auto["sentence_id"], "doc_id": auto["doc_id"],
            "y": auto["is_assumption"].astype(int), "source": "rules",
        }))
    data = pd.concat(parts, ignore_index=True)

    existing = [p for p in candidates_paths if Path(p).exists()]
    if not existing:
        raise ModelError("No candidates files found. Run the corpus step first.")
    text = pd.concat([labeler.load_candidates(p) for p in existing])
    text = text.drop_duplicates("sentence_id").set_index("sentence_id")["sentence"]
    data["sentence"] = data["sentence_id"].map(text)
    missing = data["sentence"].isna().sum()
    if missing:
        raise ModelError(f"{missing} labeled sentences are missing from the candidates files.")

    frozen = autolabel.read_doc_list(frozen_docs_path)
    data["split"] = np.where(data["doc_id"].isin(frozen), "test", "dev")
    data = data[~((data["source"] == "rules") & (data["split"] == "test"))]   # test = human labels only
    return data.reset_index(drop=True)


def check_enough(data):
    hand = data[(data["source"] == "hand") & (data["split"] == "dev")]
    if len(hand) < MIN_HAND_LABELS:
        raise ModelError(f"Only {len(hand)} hand labels outside the test documents; need at least {MIN_HAND_LABELS}.")
    counts = hand["y"].value_counts()
    for value in (0, 1):
        if counts.get(value, 0) < MIN_PER_CLASS:
            raise ModelError(f"Fewer than {MIN_PER_CLASS} hand-labeled sentences with y={value}. Label more first.")


# ---------- The model ----------
def make_model(C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000)),
    ])


def score_predictions(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "n": int(len(y_true)), "precision": float(p), "recall": float(r), "f1": float(f),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()), "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()), "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


# ---------- Step 2: cross-validation that never splits a document ----------
def cross_validate(data, use_rule_labels=False, n_splits=5):
    """Train on some documents, test on other documents. Returns (metrics, predictions)."""
    dev = data[data["split"] == "dev"].reset_index(drop=True)
    trainable = dev["source"].isin(["hand", "audited"] + (["rules"] if use_rule_labels else []))
    dev = dev[trainable].reset_index(drop=True)
    n_splits = min(n_splits, dev["doc_id"].nunique())
    if n_splits < 2:
        raise ModelError("Need at least 2 documents for cross-validation.")

    out = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(dev, dev["y"], dev["doc_id"])):
        train, valid = dev.iloc[tr], dev.iloc[va]
        valid = valid[valid["source"] == "hand"]                  # judge only on sentences you labeled alone
        if valid.empty or train["y"].nunique() < 2:
            continue
        model = make_model().fit(train["sentence"], train["y"])
        part = valid[["sentence_id", "doc_id", "sentence", "y"]].rename(columns={"y": "y_true"}).copy()
        part["score"] = model.predict_proba(valid["sentence"])[:, 1]
        part["y_pred"] = (part["score"] >= 0.5).astype(int)
        part["fold"] = fold
        out.append(part)
    if not out:
        raise ModelError("No fold could be evaluated. Check your labels.")
    predictions = pd.concat(out, ignore_index=True)
    return score_predictions(predictions["y_true"], predictions["y_pred"]), predictions


def rules_baseline(data):
    """The Day 9 rules as a 'model': predict 1 only when the rules are sure it is an assumption."""
    dev_hand = data[(data["split"] == "dev") & (data["source"] == "hand")]
    preds = []
    for s in dev_hand["sentence"]:
        r = autolabel.classify(s)
        preds.append(int(r["is_assumption"] == "1" and r["confidence"] >= autolabel.CONFIDENT))
    return score_predictions(dev_hand["y"], preds)


def compare_systems(data):
    rows = {"rules only": rules_baseline(data)}
    keep = {}
    for name, flag in [("model, your labels", False), ("model, your labels + rule labels", True)]:
        metrics, preds = cross_validate(data, use_rule_labels=flag)
        rows[name], keep[name] = metrics, preds
    table = pd.DataFrame(rows).T[["n", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]]
    return table, keep


# ---------- Step 3: train once on everything, look at the test documents once ----------
def final_model(data, use_rule_labels=False, model_path=MODEL_PATH, test_report=TEST_REPORT, force=False):
    report_path = Path(test_report)
    if report_path.exists() and not force:
        raise ModelError(f"{test_report} exists: you already looked at the test documents. "
                         "Changing the model after seeing the test result makes the result meaningless.")
    dev = data[(data["split"] == "dev") & data["source"].isin(["hand", "audited"] + (["rules"] if use_rule_labels else []))]
    model = make_model().fit(dev["sentence"], dev["y"])

    test = data[(data["split"] == "test") & (data["source"].isin(["hand", "audited"]))]
    result = {"trained_on": int(len(dev)), "used_rule_labels": bool(use_rule_labels), "test_sentences": int(len(test))}
    if len(test):
        pred = (model.predict_proba(test["sentence"])[:, 1] >= 0.5).astype(int)
        result.update(score_predictions(test["y"], pred))
    else:
        result["note"] = "No hand-labeled sentences in the test documents yet."

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2))
    return model, result


def top_words(model, k=12):
    names = np.array(model.named_steps["tfidf"].get_feature_names_out())
    weights = model.named_steps["clf"].coef_[0]
    order = np.argsort(weights)
    return {"assumption": names[order[::-1][:k]].tolist(), "not_assumption": names[order[:k]].tolist()}


# ---------- Command line ----------
def _fmt(table):
    pd.set_option("display.width", 200)
    return table.round(2).to_string()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Model A baseline.")
    parser.add_argument("command", choices=["data", "cv", "final", "words"])
    parser.add_argument("--with-rule-labels", action="store_true", help="also train on the rule labels (final)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        data = build_dataset()
        if args.command == "data":
            Path(DATASET).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(DATASET, index=False)
            print(data.groupby(["split", "source"])["y"].agg(["count", "sum"]).rename(columns={"sum": "assumptions"}))
            print(f"Saved to {DATASET}")
        elif args.command == "cv":
            check_enough(data)
            table, preds = compare_systems(data)
            print(_fmt(table))
            for name, p in preds.items():
                path = Path("reports") / f"model_a_errors_{'with_rules' if 'rule labels' in name else 'hand_only'}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                p[p["y_true"] != p["y_pred"]].sort_values("score").to_csv(path, index=False)
                print(f"Mistakes of '{name}' saved to {path}")
        elif args.command == "final":
            check_enough(data)
            _, result = final_model(data, args.with_rule_labels, force=args.force)
            print(json.dumps(result, indent=2))
            print(f"Model saved to {MODEL_PATH}. Test result saved to {TEST_REPORT}.")
        else:
            words = top_words(joblib.load(MODEL_PATH))
            for key, value in words.items():
                print(f"{key}: {', '.join(value)}")
    except (ModelError, labeler.LabelError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
