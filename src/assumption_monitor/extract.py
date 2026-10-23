"""Find numbers in a sentence and keep what they mean (implementation plan, section 8).

A regex match alone does not make a number a baseline: each figure also gets a kind, a unit,
a limit type, a role, a nearby metric phrase and a status.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

MAGNITUDES = {
    "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "million": 1e6, "mn": 1e6, "crore": 1e7, "crores": 1e7,
    "cr": 1e7, "billion": 1e9, "bn": 1e9, "trillion": 1e12,
}
CURRENCY_PREFIX = {
    "rs": "INR", "rs.": "INR", "₹": "INR", "inr": "INR", "usd": "USD", "us$": "USD", "$": "USD",
    "eur": "EUR", "€": "EUR", "gbp": "GBP", "£": "GBP",
}
CURRENCY_WORDS = {"rupee": "INR", "rupees": "INR", "dollar": "USD", "dollars": "USD", "euro": "EUR", "euros": "EUR"}

NUM = r"(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?"
ATOM = re.compile(
    rf"""(?<![\w.,])
        (?P<sign>[-−])?
        (?P<prefix>(?:Rs\.?|INR|USD|US\$|EUR|GBP|[$€£₹])\s?)?
        (?P<num>{NUM})
        (?:\s?(?P<mag>thousand|lakhs?|million|crores?|billion|trillion|mn|bn|cr)\b)?
        (?:\s?(?P<suffix>%|percentage\s+points?|per\s?cent|basis\s+points?|bps\b|pp\b
                         |(?:US\s)?(?:dollars?|rupees?|euros?)\b))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

LIMIT_WORDS = [
    (re.compile(r"\b(?:no more than|not more than|at most|up to|a maximum of|maximum of|not (?:to )?exceed(?:ing)?|capped at)\s*$", re.I), "upper", True),
    (re.compile(r"\b(?:below|under|less than|lower than|fewer than|not above|beneath)\s*$", re.I), "upper", False),
    (re.compile(r"\b(?:at least|no less than|not less than|a minimum of|minimum of|no lower than)\s*$", re.I), "lower", True),
    (re.compile(r"\b(?:above|over|more than|greater than|higher than|in excess of|exceed(?:s|ed|ing)?|beyond)\s*$", re.I), "lower", False),
    (re.compile(r"\b(?:about|around|approx\.?|approximately|roughly|near|nearly|close to|circa)\s*$", re.I), "approx", None),
]
UP_WORDS = r"increas\w*|rise[sn]?|rose|rising|grow\w*|grew|expand\w*|gain\w*|higher|up"
DOWN_WORDS = r"decreas\w*|declin\w*|fall\w*|fell|drop\w*|reduc\w*|cut|cuts|lower|lose\w*|down"
CHANGE_UP = re.compile(rf"\b(?:{UP_WORDS})\b", re.I)
CHANGE_DOWN = re.compile(rf"\b(?:{DOWN_WORDS})\b", re.I)

UNIT = re.compile(
    r"^\s*(?:/|per\s+)(?:(?P<good>barrel|bbl|litre|liter|gallon|share|tonne|ton|kg|kwh|mwh|mmbtu|unit|hour|bushel|ounce|oz)"
    r"|(?:(?:US\s)?(?P<cur>dollar|usd|euro|eur)))\b", re.I)
PERIOD = re.compile(
    r"^\s*(?:(?:/|per\s+)(?P<per>year|annum|quarter|month|week|day)\b"
    r"|(?P<adv>annually|yearly|monthly|quarterly|a year|each year|every year)\b)", re.I)
WHEN = re.compile(
    r"\b(?P<when>(?:by|through|in|during|for|over)\s+(?:fiscal\s+)?20\d\d|next (?:fiscal )?year|coming year|next quarter)\b", re.I)

METRIC_PHRASES = [
    r"exchange rates?", r"interest rates?", r"borrowing costs?", r"inflation(?: rate)?", r"tax rates?",
    r"diesel (?:prices?|costs?)", r"fuel (?:prices?|costs?)", r"(?:crude )?oil prices?", r"price of (?:crude )?oil",
    r"natural gas prices?", r"energy (?:prices?|costs?)", r"freight rates?", r"on-time delivery(?: rate)?",
    r"delivery rates?", r"customer churn|churn", r"(?:market |demand |sales |revenue |gdp )growth",
    r"(?:operating |net |gross )?margins?", r"revenues?", r"net sales|sales", r"wages?", r"labou?r costs?",
    r"unemployment(?: rate)?", r"demand", r"utili[sz]ation", r"occupancy", r"diesel|petrol|gasoline",
]
METRIC = re.compile("|".join(f"(?:{p})" for p in METRIC_PHRASES), re.I)

FIGURE_FIELDS = [
    "span", "kind", "value", "low", "high", "currency", "magnitude", "unit", "period", "limit_type",
    "inclusive", "role", "change_direction", "metric_hint", "can_be_baseline", "status", "reason",
]


# ---------- Step 1: find the number pieces ----------
def _atoms(text):
    atoms = []
    for m in ATOM.finditer(text):
        suffix = " ".join((m.group("suffix") or "").lower().split())
        prefix = (m.group("prefix") or "").strip().lower()
        mag = (m.group("mag") or "").lower()
        currency = CURRENCY_PREFIX.get(prefix, "")
        if suffix == "%" or suffix in ("percent", "per cent"):
            kind = "percent"
        elif suffix == "bps" or suffix.startswith("basis"):
            kind = "basis_points"
        elif suffix.startswith("percentage") or suffix == "pp":
            kind = "percentage_points"
        elif suffix:
            kind, currency = "money", CURRENCY_WORDS.get(suffix.split()[-1], currency)
        elif currency:
            kind = "money"
        else:
            kind = "bare"
        atoms.append({
            "start": m.start(), "end": m.end(), "span": m.group(0).strip(),
            "amount": float(m.group("num").replace(",", "")) * (-1 if m.group("sign") else 1),
            "factor": MAGNITUDES.get(mag, 1.0) if kind not in ("percent", "basis_points", "percentage_points") else 1.0,
            "magnitude": mag, "kind": kind, "currency": currency,
        })
    return atoms


def _value(atom):
    value = atom["amount"] * atom["factor"]
    return value / 100 if atom["kind"] == "basis_points" else value


# ---------- Step 2: join "between X and Y" into one range ----------
RANGE_SEP = re.compile(r"^\s*(?P<sep>and|to|-|–|—|−)\s*$", re.I)


def _merge_ranges(text, atoms):
    merged, i = [], 0
    while i < len(atoms):
        a = atoms[i]
        b = atoms[i + 1] if i + 1 < len(atoms) else None
        if b is not None:
            sep = RANGE_SEP.match(text[a["end"]:b["start"]])
            before = text[max(0, a["start"] - 14):a["start"]].lower()
            framed = bool(re.search(r"\b(between|from|ranging from|range of)\s*$", before))
            kinds = {a["kind"], b["kind"]} - {"bare"}
            same = len(kinds) <= 1 and (not a["currency"] or not b["currency"] or a["currency"] == b["currency"])
            sep_text = sep.group("sep").lower() if sep else ""
            allowed = bool(sep) and same and (
                framed or (sep_text in ("to", "-", "–", "—", "−") and kinds)
            ) and not (sep_text == "and" and not framed)
            if allowed:
                kind = next(iter(kinds), "bare")
                lo, hi = dict(a), dict(b)
                for part in (lo, hi):                     # a half without a unit takes it from the other half
                    other = hi if part is lo else lo
                    if part["kind"] == "bare":
                        part.update(kind=other["kind"], currency=other["currency"])
                    if part["factor"] == 1.0 and other["factor"] != 1.0 and part["kind"] in ("money", "bare"):
                        part["factor"], part["magnitude"] = other["factor"], other["magnitude"]
                merged.append({**hi, "start": a["start"], "span": text[a["start"]:b["end"]].strip(),
                               "low": _value(lo), "high": _value(hi), "is_range": True, "kind": kind})
                i += 2
                continue
        merged.append({**a, "is_range": False})
        i += 1
    return merged


# ---------- Step 3: give each figure its meaning ----------
NEGATION = re.compile(r"\b(?:not|never|without|unlikely|n't)\b|\bno\b(?!\s+(?:more|less|lower)\s+than)", re.I)


def _limit(window):
    """Return (limit type, inclusive?, text before the limit phrase)."""
    window = window.rstrip()
    for pattern, kind, inclusive in LIMIT_WORDS:
        m = pattern.search(window)
        if m:
            return kind, inclusive, window[:m.start()]
    return "point", None, ""


def _metric_for(text, atom, phrases):
    best, best_gap = "", None
    for start, end, name in phrases:
        gap = atom["start"] - end if end <= atom["start"] else (start - atom["end"] if start >= atom["end"] else 0)
        limit = 80 if end <= atom["start"] else 40
        if gap < 0 or gap > limit:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = name, gap
    return best


def extract_figures(sentence):
    text = " ".join(sentence.split())
    atoms = _merge_ranges(text, _atoms(text))
    phrases = [(m.start(), m.end(), " ".join(m.group(0).lower().split())) for m in METRIC.finditer(text)]

    figures, previous_end = [], 0
    for a in atoms:
        tail = text[a["end"]:a["end"] + 45]
        unit, role_unit = "", False
        u = UNIT.match(tail)
        if u:
            unit = "per " + (u.group("good") or u.group("cur") or "").lower()
            unit = unit.replace("bbl", "barrel").replace("liter", "litre")
            if u.group("cur"):
                unit = "per " + {"dollar": "USD", "usd": "USD", "euro": "EUR", "eur": "EUR"}[u.group("cur").lower()]
            role_unit = bool(u.group("good"))
        if a["kind"] == "bare" and not u:
            previous_end = a["end"]
            continue                                       # a year, a section number or a count: not a figure
        if a["kind"] == "bare" and not a["is_range"] and not role_unit:
            previous_end = a["end"]
            continue

        window = text[previous_end:a["start"]]
        limit_type, inclusive, before_limit = _limit(window)
        negated = limit_type in ("upper", "lower") and bool(NEGATION.search(before_limit))
        if a["is_range"]:
            limit_type, inclusive = "range", None
        near = text[max(previous_end, a["start"] - 30):a["start"]]
        change = ""
        if CHANGE_UP.search(near):
            change = "up"
        if CHANGE_DOWN.search(near) and (not change or CHANGE_DOWN.search(near).start() > CHANGE_UP.search(near).start()):
            change = "down"
        after_word = re.match(r"\s*(higher|more|greater|bigger|up|lower|less|smaller|down)\b", tail, re.I)
        if after_word and not change:                     # "3.5% higher than last year"
            change = "up" if after_word.group(1).lower() in ("higher", "more", "greater", "bigger", "up") else "down"
        period = ""
        p = PERIOD.match(tail[len(u.group(0)):] if u else tail) if tail else None
        if p:
            period = ("per " + p.group("per").lower()) if p.group("per") else p.group("adv").lower()
        else:
            w = WHEN.search(tail)
            period = w.group("when").lower() if w else ""

        if change and a["kind"] in ("percent", "percentage_points", "basis_points", "money") and not role_unit:
            role = "change"
        elif role_unit:
            role = "unit_price"
        elif a["kind"] == "money" and not unit:
            role = "amount"
        else:
            role = "level"
        metric = _metric_for(text, a, phrases)

        fig = {
            "span": a["span"],
            "kind": {"bare": "number", "basis_points": "percentage_points"}.get(a["kind"], a["kind"]),
            "value": None if a["is_range"] else _value(a),
            "low": a.get("low"), "high": a.get("high"),
            "currency": a["currency"], "magnitude": a["magnitude"], "unit": unit, "period": period,
            "limit_type": limit_type, "inclusive": inclusive, "role": role,
            "change_direction": change if role == "change" else "", "metric_hint": metric,
        }
        fig["can_be_baseline"] = bool(metric) and role in ("level", "unit_price") and limit_type in ("point", "approx")
        fig["_negated"] = negated
        figures.append(fig)
        previous_end = a["end"]

    # Several numbers need a clear metric each, otherwise a human must check.
    hints = [f["metric_hint"] for f in figures]
    for f in figures:
        f["status"], f["reason"] = "ok", ""
        if f.pop("_negated"):
            f["status"], f["reason"] = "review", "negated wording near a limit: check what the limit means"
        elif not f["metric_hint"]:
            f["status"], f["reason"] = "review", "no metric phrase nearby"
        elif len(figures) > 1 and hints.count(f["metric_hint"]) > 1:
            f["status"], f["reason"] = "review", "several figures share one metric phrase"
    return figures


# ---------- A table for many sentences ----------
def figures_table(sentences):
    """sentences: a DataFrame with the columns sentence_id and sentence."""
    rows = []
    for r in sentences.itertuples():
        for n, fig in enumerate(extract_figures(r.sentence), start=1):
            rows.append({"sentence_id": r.sentence_id, "figure_id": n, **fig})
    return pd.DataFrame(rows, columns=["sentence_id", "figure_id"] + FIGURE_FIELDS)


# ---------- Command line ----------
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Find numbers and their meaning in sentences.")
    parser.add_argument("--text", help="one sentence to test")
    parser.add_argument("--csv", help="a csv with sentence_id and sentence columns")
    parser.add_argument("--labels", help="keep only sentences labeled as assumptions in this labels csv")
    parser.add_argument("--out", default="data/clean/figures.csv")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    if args.text:
        figs = extract_figures(args.text)
        print(pd.DataFrame(figs, columns=FIGURE_FIELDS).T.to_string() if figs else "(no figures found)")
        return 0
    if not args.csv:
        print("ERROR: give --text or --csv.", file=sys.stderr)
        return 1
    try:
        data = pd.read_csv(args.csv, dtype=str).fillna("")
        if not {"sentence_id", "sentence"} <= set(data.columns):
            raise ValueError("the csv needs the columns sentence_id and sentence")
        if args.labels:
            labels = pd.read_csv(args.labels, dtype=str).fillna("")
            data = data[data["sentence_id"].isin(labels.loc[labels["is_assumption"] == "1", "sentence_id"])]
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    table = figures_table(data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    ok = int((table["status"] == "ok").sum())
    print(f"{len(data)} sentences, {len(table)} figures ({ok} ok, {len(table) - ok} need a look) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
