"""A small terminal tool for hand-labeling sentences (implementation plan, section 6)."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pandas as pd

TOPICS = ["fuel", "supplier", "demand", "rates", "currency", "labor", "regulation", "other"]
LABEL_COLUMNS = [
    "sentence_id", "doc_id", "is_assumption", "topic", "labeler",
    "label_version", "review_status", "is_synthetic",
]
STATUSES = {"single", "needs_review", "reviewed"}
CANDIDATE_COLUMNS = ["sentence_id", "doc_id", "section", "sentence"]


class LabelError(Exception):
    """A problem with the candidate or label files."""


# ---------- Files ----------
def load_candidates(path):
    path = Path(path)
    if not path.exists():
        raise LabelError(f"Candidates file not found: {path}. Run the corpus step first.")
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in CANDIDATE_COLUMNS if c not in df.columns]
    if missing:
        raise LabelError(f"Candidates file is missing columns: {missing}")
    return df


def load_labels(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise LabelError(f"Labels file is missing columns: {missing}")
    return df[LABEL_COLUMNS]


def save_labels(df, path):
    """Write to a temporary file first, so a crash can never leave a half-written labels file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[LABEL_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(path)


# ---------- Checks ----------
def validate_labels(labels, candidates=None):
    """Return a list of problems (an empty list means the labels are clean)."""
    problems = []
    if labels["sentence_id"].duplicated().any():
        problems.append("duplicate sentence_id")
    for _, r in labels.iterrows():
        sid = r["sentence_id"]
        if r["review_status"] not in STATUSES:
            problems.append(f"{sid}: unknown review_status '{r['review_status']}'")
        if r["is_assumption"] not in {"0", "1", ""}:
            problems.append(f"{sid}: is_assumption must be 0, 1 or empty")
        if r["is_assumption"] == "" and r["review_status"] != "needs_review":
            problems.append(f"{sid}: empty is_assumption is only allowed for needs_review")
        if r["is_assumption"] == "1" and r["topic"] not in TOPICS:
            problems.append(f"{sid}: assumption needs a topic from {TOPICS}")
        if r["is_assumption"] != "1" and r["topic"] != "":
            problems.append(f"{sid}: topic must be empty unless is_assumption is 1")
        if str(r["is_synthetic"]).lower() not in {"true", "false"}:
            problems.append(f"{sid}: is_synthetic must be True or False")
    if candidates is not None:
        unknown = set(labels["sentence_id"]) - set(candidates["sentence_id"])
        if unknown:
            problems.append(f"{len(unknown)} labeled sentence_id(s) are not in the candidates file")
    return problems


def status(labels):
    done = labels[labels["is_assumption"] != ""]
    return {
        "labeled": len(done),
        "assumptions": int((done["is_assumption"] == "1").sum()),
        "not_assumptions": int((done["is_assumption"] == "0").sum()),
        "needs_review": int((labels["review_status"] == "needs_review").sum()),
        "topics": done[done["is_assumption"] == "1"]["topic"].value_counts().to_dict(),
    }


# ---------- The interactive tool ----------
def ask_topic(input_fn, print_fn):
    print_fn("Topic:  " + "   ".join(f"{i}={t}" for i, t in enumerate(TOPICS, start=1)) + "   b=back")
    while True:
        answer = input_fn("topic > ").strip().lower()
        if answer == "b":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(TOPICS):
            return TOPICS[int(answer) - 1]
        print_fn(f"Type a number from 1 to {len(TOPICS)}, or b to go back.")


def run_labeler(candidates_path, labels_path, labeler="me", label_version="v1",
                input_fn=input, print_fn=print, only_ids=None):
    candidates = load_candidates(candidates_path)
    labels = load_labels(labels_path)

    done = set(labels["sentence_id"])
    queue = [r for r in candidates.to_dict("records") if r["sentence_id"] not in done]
    if only_ids is not None:                       # label only these sentences (for example the unsure ones)
        queue = [r for r in queue if r["sentence_id"] in only_ids]
    total = len(candidates)
    print_fn(f"{len(labels)} already labeled, {len(queue)} to go. "
             "Keys: a=assumption  n=not  s=unsure  u=undo  q=quit")

    history, i = [], 0
    while i < len(queue):
        row = queue[i]
        print_fn("\n" + "-" * 100)
        print_fn(f"[{len(labels) + 1}/{total}]  {row['doc_id']}  ({row['section']})")
        print_fn(textwrap.fill(row["sentence"], 100))
        choice = input_fn("a / n / s / u / q > ").strip().lower()

        if choice == "q":
            break
        if choice == "u":
            if history:
                last = history.pop()
                labels = labels[labels["sentence_id"] != last].reset_index(drop=True)
                save_labels(labels, labels_path)
                i -= 1
                print_fn("Undone.")
            else:
                print_fn("Nothing to undo in this session.")
            continue

        if choice == "n":
            is_assumption, topic, review = "0", "", "single"
        elif choice == "s":
            is_assumption, topic, review = "", "", "needs_review"
        elif choice == "a":
            topic = ask_topic(input_fn, print_fn)
            if topic is None:
                continue
            is_assumption, review = "1", "single"
        else:
            print_fn("Please type a, n, s, u or q.")
            continue

        new = {
            "sentence_id": row["sentence_id"], "doc_id": row["doc_id"],
            "is_assumption": is_assumption, "topic": topic, "labeler": labeler,
            "label_version": label_version, "review_status": review, "is_synthetic": "False",
        }
        labels = pd.concat([labels, pd.DataFrame([new])], ignore_index=True)
        save_labels(labels, labels_path)            # saved after every single answer
        history.append(row["sentence_id"])
        i += 1

    print_fn(f"\nSaved. {len(labels)} labeled in total.")
    return labels


# ---------- Command line ----------
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Hand-label sentences in the terminal.")
    parser.add_argument("command", choices=["label", "status", "check"])
    parser.add_argument("--candidates", default="data/labels/candidates.csv")
    parser.add_argument("--labels", default="data/labels/labels.csv")
    parser.add_argument("--labeler", default="me")
    parser.add_argument("--label-version", default="v1")
    parser.add_argument("--only-unsure-from", help="auto_labels.csv: label only the sentences the rules were unsure about")
    args = parser.parse_args(argv)

    try:
        if args.command == "label":
            only_ids = None
            if args.only_unsure_from:
                auto = pd.read_csv(args.only_unsure_from, dtype=str).fillna("")
                only_ids = set(auto.loc[auto["review_status"] == "needs_review", "sentence_id"])
            run_labeler(args.candidates, args.labels, args.labeler, args.label_version, only_ids=only_ids)
        elif args.command == "status":
            for key, value in status(load_labels(args.labels)).items():
                print(f"{key}: {value}")
        else:
            problems = validate_labels(load_labels(args.labels), load_candidates(args.candidates))
            print("Labels look clean." if not problems else "\n".join(problems))
            return 1 if problems else 0
    except LabelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
