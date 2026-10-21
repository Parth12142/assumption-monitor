"""Rule-based pre-labeling with a small human audit
(implementation plan, section 6: start with rule-based labels, correct by hand)."""
from __future__ import annotations

import math
import random
import re
import sys
import textwrap
from pathlib import Path

import pandas as pd

from assumption_monitor.labeler import (
    LABEL_COLUMNS, TOPICS, LabelError, ask_topic, load_candidates, load_labels, save_labels,
)

RULES_NAME = "rules_v1"
CONFIDENT = 0.80                                   # at or above this, the rule label is used as it is
AUTO_COLUMNS = LABEL_COLUMNS + ["confidence", "reason"]
AUDIT_COLUMNS = ["sentence_id", "auto_is_assumption", "auto_topic",
                 "human_is_assumption", "human_topic", "agreed"]


def _rx(pattern):
    return re.compile(pattern, re.IGNORECASE)


# ---------- The rules (change them after every audit, and write the change in the decision log) ----------
TOPIC_WORDS = {
    "fuel": _rx(r"\b(oil|crude|diesel|gasoline|petroleum|natural gas|fuel|energy (price|cost)s?|electricity)\b"),
    "supplier": _rx(r"\b(suppliers?|vendors?|supply chain|raw materials?|components?|shortages?|deliver(y|ies))\b"),
    "demand": _rx(r"\b(demand|customers?|consumer (spending|confidence)|market growth|sales volumes?|orders|bookings)\b"),
    "rates": _rx(r"\b(interest rates?|borrowing costs?|libor|sofr|federal funds|basis points|prime rate)\b"),
    "currency": _rx(r"\b(exchange rates?|currenc(y|ies)|foreign exchange|dollar|euro|rupee|yen|pound sterling)\b"),
    "labor": _rx(r"\b(wages?|labou?r|workforce|hiring|unions?|employee (turnover|costs?))\b"),
    "regulation": _rx(r"\b(regulat\w+|legislation|laws?|tax(es)?|tariffs?|compliance|approvals?)\b"),
}
BOILERPLATE = _rx(r"forward-looking statements?|actual results (may|could|might|will) differ|safe harbor|"
                  r"securities litigation reform act|undue reliance|undertake no (obligation|duty)")
ASSUMED_THING = _rx(r"liabilit\w+ assumed|assumed liabilit\w+|"
                    r"assum\w+ (\w+ ){0,2}(debt|liabilit\w+|obligations?|leases?|contracts?)")
VALUATION = _rx(r"fair value|black-scholes|stock option|option pricing|risk-free|dividend yield|"
                r"expected volatility|discount rate|goodwill|impairment")
EXPLICIT = _rx(r"\b(we|management|the company|it)\s+(assume|assumes|assumed)\b|"
               r"\bbased on (the|our|these) assumptions?\b|\bassuming that\b|"
               r"\bassumptions? (is|are|that|include|underlying)\b|"
               r"\b(budget|plan|forecast|projections?|model|outlook|guidance)s?\s+(assume|assumes|is based on|are based on)\b")
EXPECT = _rx(r"\b(expect|expects|expected|anticipate|anticipates|anticipated|projected|projects that|"
             r"forecast|forecasts|forecasted)\b")
FUTURE = _rx(r"\b(will|would|to (remain|continue|stay|increase|decrease|decline|rise|grow|fall|be|improve)|"
             r"through (fiscal )?20\d\d|next (year|fiscal year|quarter|twelve months)|coming (year|months)|"
             r"over the (next|coming))\b")
HEDGE = _rx(r"\b(could|may|might|can)\b")
OWN_RESULT = _rx(r"\b(our|the company'?s)\s+((net|total|gross|operating)\s+)?"
                 r"(revenues?|sales|earnings|income|profits?|margins?|expenses|cash flows?|results)\b")
OWN_PLAN = _rx(r"\bwe\s+(plan|intend|aim|seek|are planning|will)\s+to\b")


def guess_topic(sentence):
    """The topic whose word appears first in the sentence ('other' if none appears)."""
    best, best_pos = "other", None
    for topic, pattern in TOPIC_WORDS.items():
        m = pattern.search(sentence)
        if m and (best_pos is None or m.start() < best_pos):
            best, best_pos = topic, m.start()
    return best


def _result(is_assumption, topic, confidence, reason):
    return {"is_assumption": is_assumption, "topic": topic, "confidence": confidence, "reason": reason}


def classify(sentence):
    """Return the rule label for one sentence: is_assumption ('1', '0' or '' for unsure), topic, confidence, reason."""
    s = " ".join(sentence.split())
    topic = guess_topic(s)

    if BOILERPLATE.search(s):
        return _result("0", "", 0.95, "legal boilerplate")
    if ASSUMED_THING.search(s):
        return _result("0", "", 0.90, "assumed liabilities or contracts (accounting, not an outlook)")
    if VALUATION.search(s) and (EXPLICIT.search(s) or EXPECT.search(s)):
        return _result("", "", 0.50, "valuation inputs: decide by hand")
    if EXPLICIT.search(s):
        if topic == "other":
            return _result("1", "other", 0.70, "explicit assumption wording, topic unclear")
        return _result("1", topic, 0.90, "explicit assumption wording")

    has_expect, has_future, hedged = EXPECT.search(s), FUTURE.search(s), HEDGE.search(s)
    if has_expect and has_future and topic != "other" and not hedged:
        if OWN_RESULT.search(s):
            return _result("1", topic, 0.70, "outside driver, but also the company's own results")
        return _result("1", topic, 0.85, "expectation about an outside driver")
    if hedged and not has_expect:
        return _result("0", "", 0.85, "risk or possibility, no stated expectation")
    if OWN_PLAN.search(s) and topic == "other":
        return _result("0", "", 0.80, "the company's own plan")
    if not has_expect and not has_future:
        return _result("0", "", 0.85, "no forward-looking wording")
    return _result("", "", 0.50, "mixed signals: decide by hand")


# ---------- Step 1: label every candidate with the rules ----------
def autolabel(candidates_path, out_path):
    candidates = load_candidates(candidates_path)
    rows = []
    for r in candidates.to_dict("records"):
        res = classify(r["sentence"])
        confident = res["confidence"] >= CONFIDENT
        rows.append({
            "sentence_id": r["sentence_id"], "doc_id": r["doc_id"],
            "is_assumption": res["is_assumption"] if confident else "",
            "topic": res["topic"] if confident else "",
            "labeler": RULES_NAME, "label_version": "v1",
            "review_status": "auto" if confident else "needs_review",
            "is_synthetic": "False",
            "confidence": f"{res['confidence']:.2f}", "reason": res["reason"],
        })
    table = pd.DataFrame(rows, columns=AUTO_COLUMNS)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)
    return table


def summarize(table):
    auto = table[table["review_status"] == "auto"]
    return {
        "sentences": len(table),
        "auto_assumption": int((auto["is_assumption"] == "1").sum()),
        "auto_not_assumption": int((auto["is_assumption"] == "0").sum()),
        "needs_review": int((table["review_status"] == "needs_review").sum()),
    }


# ---------- Step 2: a small human audit of the rule labels ----------
def load_auto(path):
    path = Path(path)
    if not path.exists():
        raise LabelError(f"Auto labels not found: {path}. Run the rules first.")
    return pd.read_csv(path, dtype=str).fillna("")


def load_audit(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    return pd.read_csv(path, dtype=str).fillna("")[AUDIT_COLUMNS]


def save_audit(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[AUDIT_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(path)


def run_audit(candidates_path, auto_path, labels_path, audit_path, n_per_class=20, seed=42,
              input_fn=input, print_fn=print):
    """Show a random sample of the rule labels. You agree (y) or change (c). Your answers become real labels."""
    candidates = load_candidates(candidates_path).set_index("sentence_id")
    auto = load_auto(auto_path)
    labels = load_labels(labels_path)
    audit = load_audit(audit_path)

    taken = set(labels["sentence_id"]) | set(audit["sentence_id"])
    pool = auto[(auto["review_status"] == "auto") & ~auto["sentence_id"].isin(taken)]
    rng = random.Random(seed)

    sample = []
    for value in ["1", "0"]:
        already = int((audit["auto_is_assumption"] == value).sum())
        ids = sorted(pool.loc[pool["is_assumption"] == value, "sentence_id"])
        rng.shuffle(ids)
        sample += ids[:max(0, n_per_class - already)]
    rng.shuffle(sample)

    auto_by_id = auto.set_index("sentence_id")
    print_fn(f"{len(sample)} sentences to check. Keys: y=agree  c=change  s=skip  q=quit")

    for number, sid in enumerate(sample, start=1):
        row = auto_by_id.loc[sid]
        shown = "assumption, topic " + row["topic"] if row["is_assumption"] == "1" else "NOT an assumption"
        print_fn("\n" + "-" * 100)
        print_fn(f"[{number}/{len(sample)}]  {row['doc_id']}")
        print_fn(textwrap.fill(candidates.loc[sid, "sentence"], 100))
        print_fn(f"RULES SAY: {shown}   (because: {row['reason']})")

        while True:
            choice = input_fn("y / c / s / q > ").strip().lower()
            if choice in {"y", "c", "s", "q"}:
                break
            print_fn("Please type y, c, s or q.")
        if choice == "q":
            break
        if choice == "s":
            continue

        if choice == "y":
            human_is, human_topic = row["is_assumption"], row["topic"]
        else:
            human_is, human_topic = None, ""
            while human_is is None:
                kind = input_fn("your label: a = assumption, n = not > ").strip().lower()
                if kind == "n":
                    human_is = "0"
                elif kind == "a":
                    topic = ask_topic(input_fn, print_fn)
                    if topic is not None:
                        human_is, human_topic = "1", topic
                else:
                    print_fn("Please type a or n.")

        labels = pd.concat([labels, pd.DataFrame([{
            "sentence_id": sid, "doc_id": row["doc_id"], "is_assumption": human_is, "topic": human_topic,
            "labeler": "me", "label_version": "v1", "review_status": "reviewed", "is_synthetic": "False",
        }])], ignore_index=True)
        audit = pd.concat([audit, pd.DataFrame([{
            "sentence_id": sid, "auto_is_assumption": row["is_assumption"], "auto_topic": row["topic"],
            "human_is_assumption": human_is, "human_topic": human_topic,
            "agreed": str(human_is == row["is_assumption"] and human_topic == row["topic"]),
        }])], ignore_index=True)
        save_labels(labels, labels_path)
        save_audit(audit, audit_path)

    print_fn(f"\nSaved. {len(audit)} sentences audited in total.")
    return audit


# ---------- Step 3: how good are the rules? ----------
def wilson_interval(k, n, z=1.96):
    """A plausible range for the true agreement rate, given k agreements out of n checks."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def audit_report(audit, auto=None, candidates=None, show=8):
    lines = []
    n = len(audit)
    if n == 0:
        return "No audited sentences yet."
    agreed = int((audit["agreed"] == "True").sum())
    low, high = wilson_interval(agreed, n)
    lines.append(f"Checked {n} rule labels: you agreed with {agreed} ({agreed / n:.0%}). "
                 f"Plausible true agreement: {low:.0%} to {high:.0%}.")
    for value, name in [("1", "labeled assumption"), ("0", "labeled not an assumption")]:
        part = audit[audit["auto_is_assumption"] == value]
        if len(part):
            ok = int((part["agreed"] == "True").sum())
            lines.append(f"  {name}: {ok} of {len(part)} correct ({ok / len(part):.0%})")
    wrong = audit[audit["agreed"] != "True"]
    if len(wrong) and auto is not None and candidates is not None:
        lines.append("\nExamples you changed (use them to improve the rules):")
        auto_by_id = auto.set_index("sentence_id")
        cand_by_id = candidates.set_index("sentence_id")
        for _, r in wrong.head(show).iterrows():
            lines.append(f"- {textwrap.shorten(cand_by_id.loc[r['sentence_id'], 'sentence'], 150)}")
            lines.append(f"    rules: {r['auto_is_assumption'] or '-'}/{r['auto_topic'] or '-'} "
                         f"({auto_by_id.loc[r['sentence_id'], 'reason']})  "
                         f"you: {r['human_is_assumption']}/{r['human_topic'] or '-'}")
    return "\n".join(lines)


# ---------- Freeze the final-test documents (before any tuning) ----------
def freeze_test_docs(candidates_path, out_path, share=0.2, seed=42):
    """Choose a share of the documents as the final test set and write their ids, one per line."""
    out = Path(out_path)
    if out.exists():
        raise LabelError(f"{out} already exists. The test documents are frozen and must not change.")
    docs = sorted(set(load_candidates(candidates_path)["doc_id"]))
    if len(docs) < 5:
        raise LabelError("Need at least 5 documents to freeze a test set.")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(docs, max(1, round(len(docs) * share))))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chosen) + "\n")
    return chosen


def read_doc_list(path):
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise LabelError(f"Document list not found: {p}")
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


# ---------- Compare the rules with your own manual labels ----------
def compare_with_manual(auto, labels, exclude_docs=()):
    """How well do the rules match labels you made by hand (without seeing the rules)?"""
    manual = labels[(labels["is_assumption"] != "") & (labels["review_status"] == "single")]
    manual = manual[~manual["doc_id"].isin(exclude_docs)]
    merged = manual.merge(
        auto[["sentence_id", "is_assumption", "topic", "review_status", "reason"]],
        on="sentence_id", suffixes=("_hand", "_rules"),
    )
    n = len(merged)
    if n == 0:
        return {"n": 0}
    sure = merged[merged["review_status_rules"] == "auto"]
    right = sure[sure["is_assumption_rules"] == sure["is_assumption_hand"]]
    said_yes = sure[sure["is_assumption_rules"] == "1"]
    true_yes = merged[merged["is_assumption_hand"] == "1"]
    hits = said_yes[said_yes["is_assumption_hand"] == "1"]
    topic_ok = hits[hits["topic_rules"] == hits["topic_hand"]]
    wrong = pd.concat([
        sure[sure["is_assumption_rules"] != sure["is_assumption_hand"]],
        true_yes[true_yes["review_status_rules"] != "auto"],
    ]).drop_duplicates("sentence_id")
    return {
        "n": n,
        "coverage": len(sure) / n,
        "accuracy_when_sure": len(right) / len(sure) if len(sure) else 0.0,
        "precision": len(hits) / len(said_yes) if len(said_yes) else 0.0,
        "recall": len(hits) / len(true_yes) if len(true_yes) else 0.0,
        "topic_accuracy": len(topic_ok) / len(hits) if len(hits) else 0.0,
        "mistakes": wrong,
    }


def compare_report(result, candidates, show=8):
    if result["n"] == 0:
        return "No manual labels to compare with yet."
    lines = [
        f"Compared with {result['n']} sentences you labeled by hand:",
        f"  rules were sure about {result['coverage']:.0%} of them (the rest go to you)",
        f"  when sure, they matched your label {result['accuracy_when_sure']:.0%} of the time",
        f"  assumptions: precision {result['precision']:.0%}, recall {result['recall']:.0%}",
        f"  topic correct for {result['topic_accuracy']:.0%} of the assumptions they found",
    ]
    mistakes = result["mistakes"]
    if len(mistakes):
        by_id = candidates.set_index("sentence_id")
        lines.append("\nExamples where the rules and you differ (use them to improve the rules):")
        for _, r in mistakes.head(show).iterrows():
            lines.append(f"- {textwrap.shorten(by_id.loc[r['sentence_id'], 'sentence'], 150)}")
            lines.append(f"    you: {r['is_assumption_hand']}/{r['topic_hand'] or '-'}   "
                         f"rules: {r['is_assumption_rules'] or 'unsure'}/{r['topic_rules'] or '-'} ({r['reason']})")
    return "\n".join(lines)


# ---------- Command line ----------
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Rule-based pre-labeling and a small human audit.")
    parser.add_argument("command", choices=["run", "audit", "report", "compare", "freeze"])
    parser.add_argument("--candidates", default="data/labels/candidates.csv")
    parser.add_argument("--auto", default="data/labels/auto_labels.csv")
    parser.add_argument("--labels", default="data/labels/labels.csv")
    parser.add_argument("--audit-log", default="data/labels/audit_log.csv")
    parser.add_argument("--n-per-class", type=int, default=20)
    parser.add_argument("--test-docs", default="data/labels/frozen_test_docs.txt")
    parser.add_argument("--share", type=float, default=0.2)
    parser.add_argument("--exclude-docs", help="file with document ids to leave out (the frozen test documents)")
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            table = autolabel(args.candidates, args.auto)
            for key, value in summarize(table).items():
                print(f"{key}: {value}")
            print(f"Saved to {args.auto}")
        elif args.command == "audit":
            run_audit(args.candidates, args.auto, args.labels, args.audit_log, args.n_per_class)
        elif args.command == "freeze":
            chosen = freeze_test_docs(args.candidates, args.test_docs, args.share)
            print(f"Froze {len(chosen)} test documents in {args.test_docs}")
        elif args.command == "compare":
            result = compare_with_manual(load_auto(args.auto), load_labels(args.labels), read_doc_list(args.exclude_docs))
            print(compare_report(result, load_candidates(args.candidates)))
        else:
            print(audit_report(load_audit(args.audit_log), load_auto(args.auto), load_candidates(args.candidates)))
    except LabelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
