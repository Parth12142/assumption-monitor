"""Core analytics, review persistence, PDF generation, and monitoring engine for the Assumption Monitor demo.

Bridges the existing `assumption_monitor` modules:
- `reader.py` (PDF/DOCX ingestion, cleaning, and audit logs)
- `autolabel.py` (Rules v1 classification, topic detection, audit & Wilson intervals)
- `labeler.py` (Label validation, topics, and status counts)
- `model_a.py` (TF-IDF + LogisticRegression assumption detector & CV evaluation)
- `sources.py` & `series.py` (FRED indicator registry, SHA-256 verification, W-FRI cleaning,
  robust-z outlier detection, and point-in-time `available_at` filtering)
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

for venv_name in (".venv", "venv"):
    site_pkgs = ROOT_DIR / venv_name / "lib" / "python3.14" / "site-packages"
    if site_pkgs.exists() and str(site_pkgs) not in sys.path:
        sys.path.insert(0, str(site_pkgs))

import fitz  # PyMuPDF
import joblib
import numpy as np
import pandas as pd

from assumption_monitor import autolabel, labeler, model_a, reader, series, sources

PASSIVE_ASSUMP_RX = re.compile(
    r"\b(?:is|are)\s+(?:assumed|expected|projected|forecast(?:ed)?)\s+to\b|"
    r"\b(?:management|we)\s+believes?\s+that\b.*\bwill\b",
    re.IGNORECASE,
)

# ---------- Regex patterns for quantitative assumption threshold extraction ----------
RANGE_USD_RX = re.compile(
    r"between\s+(?:USD|\$)\s*(\d+(?:\.\d+)?)\s+and\s+(?:USD|\$)?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
TARGET_USD_RX = re.compile(
    r"(?:near|around|about|close to|at|averages?)\s+(?:USD|\$)\s*(\d+(?:\.\d+)?)\s*(?:per\s+barrel|/bbl)?",
    re.IGNORECASE,
)
ABOVE_USD_RX = re.compile(
    r"(?:above|exceed\w*|over|stay\s+below|below|under)\s+(?:USD|\$)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
INR_RANGE_RX = re.compile(
    r"between\s+(?:INR|Rs\.?)\s*(\d+(?:\.\d+)?)\s+and\s+(?:INR|Rs\.?)?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
INR_TARGET_RX = re.compile(
    r"(?:close to|about|around|near|at)\s+(?:INR|Rs\.?)\s*(\d+(?:\.\d+)?)\s*(?:per\s+(?:US\s+dollar|USD))?",
    re.IGNORECASE,
)
INR_ABOVE_RX = re.compile(
    r"(?:above|exceed\w*|over|stay\s+below|below|under)\s+(?:INR|Rs\.?)\s*(\d+(?:\.\d+)?)\s*(?:per\s+(?:US\s+dollar|USD))?",
    re.IGNORECASE,
)
DIESEL_CAP_RX = re.compile(
    r"(?:stay\s+)?(?:at\s+or\s+below|below|under|exceed\w*|above)\s+(?:Rs\.?|INR)\s*(\d+(?:\.\d+)?)\s*per\s+litre",
    re.IGNORECASE,
)
RATE_TARGET_RX = re.compile(
    r"interest rate.*?(?:assumed to be|carries an interest rate of|near|around|below|of)\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
GROWTH_PCT_RX = re.compile(
    r"(?:grow\w*|rise\w*|increase\w*|stay\s+below|of)\s+(?:by\s+|at\s+around\s+|below\s+)?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

DIESEL_BASE_CRUDE_USD = 72.0
DIESEL_BASE_INR_PER_L = 88.0
DIESEL_PASS_THROUGH_INR_PER_USD_BBL = 0.65

DEMO_REVIEWS_PATH = ROOT_DIR / "data" / "labels" / "demo_reviews.csv"
GENERATED_PDF_DIR = ROOT_DIR / "data" / "raw" / "generated"


class DemoEngine:
    """Stateful helper managing Model A, cleaned weekly series, human reviews, and injected clauses."""

    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root = Path(root_dir)
        self.model = None
        self.registry_df = None
        self.weekly_series: dict[str, pd.DataFrame] = {}
        # doc_id -> list of custom injected sentence dicts
        self.injected_clauses: dict[str, list[dict]] = {}
        # sentence_id -> {"is_assumption": "1"|"0"|"", "topic": str, "review_status": str, "sentence": str, "doc_id": str}
        self.human_reviews: dict[str, dict] = {}
        self._load_resources()
        self._load_human_reviews()

    def _load_resources(self):
        model_path = self.root / model_a.MODEL_PATH
        if model_path.exists():
            self.model = joblib.load(model_path)
        else:
            data = model_a.build_dataset(
                labels_path=str(self.root / model_a.LABELS),
                silver_auto_path=str(self.root / model_a.SILVER_AUTO),
                candidates_paths=[str(self.root / p) for p in model_a.CANDIDATES],
                frozen_docs_path=str(self.root / model_a.FROZEN_DOCS),
            )
            dev = data[data["split"] == "dev"]
            self.model = model_a.make_model().fit(dev["sentence"], dev["y"])

        reg_path = self.root / sources.REGISTRY_PATH
        self.registry_df = sources.load_registry(reg_path)

        for sid in ("DCOILWTICO", "DEXINUS"):
            clean_csv = self.root / "data" / "clean" / f"weekly_{sid}.csv"
            if clean_csv.exists():
                df = pd.read_csv(clean_csv)
            else:
                df = series.clean_series(
                    sid,
                    log_path=self.root / sources.LOG_PATH,
                    registry=self.registry_df,
                )
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            df["available_at"] = pd.to_datetime(df["available_at"])
            self.weekly_series[sid] = df.sort_values("observation_date").reset_index(drop=True)

    def _load_human_reviews(self):
        self.human_reviews = {}
        if DEMO_REVIEWS_PATH.exists():
            try:
                df = pd.read_csv(DEMO_REVIEWS_PATH, dtype=str).fillna("")
                for r in df.to_dict("records"):
                    self.human_reviews[r["sentence_id"]] = r
            except Exception:
                pass

    def _save_human_reviews(self):
        DEMO_REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "sentence_id", "doc_id", "sentence", "is_assumption",
            "topic", "review_status", "updated_at_utc",
        ]
        rows = list(self.human_reviews.values())
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        tmp = DEMO_REVIEWS_PATH.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(DEMO_REVIEWS_PATH)

    # ---------- Human Review Actions ----------
    def save_review_decision(
        self,
        sentence_id: str,
        doc_id: str,
        sentence: str,
        is_assumption: str,
        topic: str = "",
        review_status: str = "reviewed",
    ) -> dict:
        if is_assumption == "needs_review":
            is_assumption = ""
            review_status = "needs_review"
        if is_assumption not in ("1", "0", ""):
            raise ValueError("is_assumption must be '1', '0', '', or 'needs_review'")
        if is_assumption == "1":
            if topic not in labeler.TOPICS:
                topic = autolabel.guess_topic(sentence)
        else:
            topic = ""

        record = {
            "sentence_id": str(sentence_id),
            "doc_id": str(doc_id),
            "sentence": str(sentence),
            "is_assumption": str(is_assumption),
            "topic": str(topic),
            "review_status": str(review_status),
            "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.human_reviews[str(sentence_id)] = record
        self._save_human_reviews()
        return record

    def reset_reviews_and_clauses(self, doc_id: str | None = None):
        if doc_id:
            self.injected_clauses.pop(doc_id, None)
            to_del = [k for k, v in self.human_reviews.items() if v.get("doc_id") == doc_id]
            for k in to_del:
                self.human_reviews.pop(k, None)
        else:
            self.injected_clauses.clear()
            self.human_reviews.clear()
        self._save_human_reviews()

    # ---------- Add & Predict New Assumption Clause ----------
    def add_assumption_clause(
        self,
        doc_source: str,
        sentence: str,
        section: str = "5. Custom Strategic Assumptions",
        as_of: str = "2026-10-16",
    ) -> dict:
        clean_s = " ".join(str(sentence).split())
        if not clean_s:
            raise ValueError("Sentence cannot be empty.")
        if not clean_s.endswith((".", "!", "?")):
            clean_s += "."

        target_doc = doc_source if doc_source in ("plan1", "plan2", "plan3") else "plan1"
        bucket = self.injected_clauses.setdefault(target_doc, [])
        idx = len(bucket) + 1
        sid = f"{target_doc}-custom-{idx:02d}"

        explained = self.explain_sentence(clean_s)
        snapshots = self.get_indicator_snapshot(as_of=as_of)
        r_info = explained["rules_v1"]
        prob = explained["model_a"]["probability"]
        is_assump = (
            r_info["is_assumption"] == "1"
            or prob >= 0.40
            or bool(PASSIVE_ASSUMP_RX.search(clean_s))
            or bool(RANGE_USD_RX.search(clean_s))
            or bool(TARGET_USD_RX.search(clean_s))
            or bool(ABOVE_USD_RX.search(clean_s))
            or bool(INR_TARGET_RX.search(clean_s))
            or bool(INR_ABOVE_RX.search(clean_s))
            or bool(DIESEL_CAP_RX.search(clean_s))
        )
        topic = r_info["topic"] or r_info["guessed_topic"] or "other"
        claim = self.evaluate_claim(clean_s, topic, is_assump, snapshots)

        clause_entry = {
            "sentence_id": sid,
            "doc_id": target_doc,
            "page_start": 99,
            "page_end": 99,
            "pages": "Live",
            "section": section,
            "sentence": clean_s,
            "n_words": len(clean_s.split()),
            "n_occurrences": 1,
            "percents": reader.PERCENT.findall(clean_s),
            "money": reader.MONEY.findall(clean_s),
            "extraction_status": "ok" if len(clean_s.split()) >= 5 else "too_short",
        }
        bucket.append(clause_entry)

        return {
            "sentence_id": sid,
            "injected": clause_entry,
            "explained": explained,
            "predicted_is_assumption": is_assump,
            "predicted_topic": topic,
            "claim": claim,
            "total_injected_for_doc": len(bucket),
        }

    # ---------- Dynamic PDF Business Plan Generator ----------
    def generate_sample_pdf(
        self,
        company_name: str = "AeroVantage Freight & Aviation Pvt. Ltd.",
        plan_horizon: str = "2026-2029",
        crude_low: float = 68.0,
        crude_high: float = 82.0,
        diesel_cap: float = 94.0,
        fx_target: float = 83.5,
        fx_trigger: float = 87.5,
        rate_target: float = 9.8,
        demand_growth: float = 11.0,
    ) -> dict:
        """Create a real 3-page PDF business plan with headers, footers, table fragments, and assumptions."""
        GENERATED_PDF_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = GENERATED_PDF_DIR / "plan3.pdf"

        header_text = f"{company_name} | Strategic Plan {plan_horizon} | CONFIDENTIAL"
        footer_text = "Prepared by Quantitative Strategy Desk - Draft v1"

        pages_content = [
            [
                f"1. Executive Summary",
                f"{company_name} operates an integrated air-and-surface express cargo network across Western and Southern India.",
                f"In 2025 the company generated operating revenue of Rs. 164.0 crore and an operating margin of 11.4%.",
                f"We assume that the average diesel price will stay at or below Rs. {diesel_cap:.0f} per litre throughout the plan period.",
                f"We expect the global crude oil price to remain between USD {crude_low:.0f} and USD {crude_high:.0f} per barrel.",
                f"We also assume that express cargo demand in our primary corridors will grow by {demand_growth:.0f}% per year.",
                f"Management plans to commission four new automated sorting hubs by the end of 2028.",
            ],
            [
                f"2.1 Fuel and Currency Exposure",
                f"Aviation turbine fuel and road diesel together represent 41% of direct operating costs.",
                f"Management believes that crude oil will trade near USD {(crude_low + crude_high) / 2:.0f} per barrel on average during 2026.",
                f"Prices above USD {crude_high + 10:.0f} per barrel would reduce the operating margin by about 3.5 percentage points.",
                f"We assume the exchange rate will remain close to INR {fx_target:.1f} per USD during the planning period.",
                f"A move above INR {fx_trigger:.1f} per USD would raise aircraft lease and import costs by roughly 7%.",
                f"Metric",
                f"2026E",
                f"2027E",
                f"Revenue",
                f"192.5",
                f"Pilot and ground crew wages are expected to rise by 6.5% per year.",
            ],
            [
                f"3. Financing, Regulation and Downside Sensitivities",
                f"The interest rate on the proposed fleet expansion loan is assumed to be {rate_target:.1f}%, and we expect borrowing costs to remain stable.",
                f"In a downside scenario where crude oil averages USD {crude_high + 15:.0f} per barrel, we estimate that the 2027 operating margin would fall from 11.4% to about 6.8%.",
                f"We expect regulatory changes on cargo emissions to have only a minor impact before 2028.",
                f"The board of directors reviews macro assumptions every quarter.",
            ],
        ]

        doc = fitz.open()
        for page_no, lines in enumerate(pages_content, start=1):
            page = doc.new_page(width=595, height=842)
            page.insert_text((48, 36), header_text, fontsize=9)
            y = 80
            for ln in lines:
                page.insert_text((48, y), ln, fontsize=11)
                y += 22
            page.insert_text((48, 790), footer_text, fontsize=9)
            page.insert_text((48, 810), f"Page {page_no}", fontsize=9)
        doc.save(str(pdf_path))
        doc.close()

        return {
            "doc_id": "plan3",
            "pdf_path": str(pdf_path),
            "file_name": "plan3.pdf",
            "company_name": company_name,
            "pages": 3,
            "size_bytes": pdf_path.stat().st_size,
        }

    # ---------- Point-in-time series lookup ----------
    def get_indicator_snapshot(
        self,
        as_of: str | None = None,
        override_oil: float | None = None,
        override_fx: float | None = None,
    ) -> dict[str, dict]:
        """Return point-in-time indicator snapshots where `available_at <= as_of`."""
        as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp("2026-10-22")

        snapshots = {}
        for sid in ("DCOILWTICO", "DEXINUS"):
            df = self.weekly_series[sid]
            known = df[df["available_at"] <= as_of_ts].dropna(subset=["value"])
            if known.empty:
                known = df.dropna(subset=["value"]).head(1)
            latest = known.iloc[-1]
            prev_4w = known.iloc[-5] if len(known) >= 5 else known.iloc[0]
            prev_26w = known.iloc[-26:] if len(known) >= 26 else known

            entry = sources.get_entry(self.registry_df, sid)
            live_val = float(latest["value"])
            overridden = False
            if sid == "DCOILWTICO" and override_oil is not None:
                live_val = float(override_oil)
                overridden = True
            elif sid == "DEXINUS" and override_fx is not None:
                live_val = float(override_fx)
                overridden = True

            change_4w_pct = (
                ((live_val - float(prev_4w["value"])) / float(prev_4w["value"])) * 100.0
                if float(prev_4w["value"]) != 0
                else 0.0
            )

            snapshots[sid] = {
                "series_id": sid,
                "title": entry["title"],
                "units": entry["units"],
                "currency": entry["currency"],
                "topic": entry["topic"],
                "mapping_type": entry["mapping_type"],
                "publication_pattern": entry["publication_pattern"],
                "as_of_requested": str(as_of_ts.date()),
                "observation_date": str(latest["observation_date"].date()),
                "available_at": str(latest["available_at"].date()),
                "value": round(live_val, 3),
                "recorded_value": round(float(latest["value"]), 3),
                "is_overridden": overridden,
                "is_filled": bool(latest["is_filled"]),
                "is_outlier": bool(latest["is_outlier"]),
                "robust_z": None if pd.isna(latest["robust_z"]) else round(float(latest["robust_z"]), 2),
                "change_4w_pct": round(change_4w_pct, 2),
                "min_26w": round(float(prev_26w["value"].min()), 2),
                "max_26w": round(float(prev_26w["value"].max()), 2),
                "mean_26w": round(float(prev_26w["value"].mean()), 2),
                "sparkline": [
                    {
                        "obs_date": str(r.observation_date.date()),
                        "avail_date": str(r.available_at.date()),
                        "value": round(float(r.value), 2),
                    }
                    for r in prev_26w.itertuples()
                ],
            }
        return snapshots

    # ---------- Explainable Model A + Rules v1 on one sentence ----------
    def explain_sentence(self, sentence: str) -> dict:
        """Run both Model A and Rules v1 on a single sentence and return feature attributions."""
        clean_s = " ".join(str(sentence).split())
        tfidf = self.model.named_steps["tfidf"]
        clf = self.model.named_steps["clf"]

        vec = tfidf.transform([clean_s])
        prob = float(self.model.predict_proba([clean_s])[0, 1])
        intercept = float(clf.intercept_[0])

        feature_names = tfidf.get_feature_names_out()
        weights = clf.coef_[0]
        nonzero_idx = vec.nonzero()[1]
        token_impacts = []
        for idx in nonzero_idx:
            tfidf_val = float(vec[0, idx])
            coef_val = float(weights[idx])
            contrib = tfidf_val * coef_val
            token_impacts.append({
                "ngram": str(feature_names[idx]),
                "tfidf": round(tfidf_val, 3),
                "coef": round(coef_val, 3),
                "contribution": round(contrib, 4),
            })
        token_impacts.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        rule_res = autolabel.classify(clean_s)
        guessed_topic = autolabel.guess_topic(clean_s)

        rule_signals = {
            "EXPLICIT": bool(autolabel.EXPLICIT.search(clean_s)),
            "EXPECT": bool(autolabel.EXPECT.search(clean_s)),
            "FUTURE": bool(autolabel.FUTURE.search(clean_s)),
            "HEDGE": bool(autolabel.HEDGE.search(clean_s)),
            "BOILERPLATE": bool(autolabel.BOILERPLATE.search(clean_s)),
            "ASSUMED_THING": bool(autolabel.ASSUMED_THING.search(clean_s)),
            "VALUATION": bool(autolabel.VALUATION.search(clean_s)),
            "OWN_RESULT": bool(autolabel.OWN_RESULT.search(clean_s)),
            "OWN_PLAN": bool(autolabel.OWN_PLAN.search(clean_s)),
            "HAS_CUE": bool(
                re.search(
                    r"\b(?:assum\w*|expect\w*|project\w*|forecast\w*|estimat\w*|anticipat\w*|believ\w*)\b",
                    clean_s,
                    re.IGNORECASE,
                )
            ),
        }
        matched_topics = [
            t for t, rx in autolabel.TOPIC_WORDS.items() if rx.search(clean_s)
        ]

        percents = reader.PERCENT.findall(clean_s)
        money = reader.MONEY.findall(clean_s)

        return {
            "sentence": clean_s,
            "n_words": len(clean_s.split()),
            "model_a": {
                "probability": round(prob, 4),
                "intercept": round(intercept, 4),
                "top_positive": [t for t in token_impacts if t["contribution"] > 0][:6],
                "top_negative": [t for t in token_impacts if t["contribution"] < 0][:6],
                "all_Matched_ngrams": len(token_impacts),
            },
            "rules_v1": {
                "is_assumption": rule_res["is_assumption"],
                "topic": rule_res["topic"] or guessed_topic,
                "raw_topic": rule_res["topic"],
                "guessed_topic": guessed_topic,
                "confidence": float(rule_res["confidence"]),
                "is_confident": float(rule_res["confidence"]) >= autolabel.CONFIDENT,
                "reason": rule_res["reason"],
                "signals": rule_signals,
                "matched_topics": matched_topics,
            },
            "entities": {
                "percents": percents,
                "money": money,
            },
        }

    # ---------- Quantitative claim matching against live FRED indicators ----------
    def evaluate_claim(
        self,
        sentence: str,
        topic: str,
        is_assumption: bool,
        snapshots: dict[str, dict],
    ) -> dict:
        """Compare a sentence's stated quantitative expectation against point-in-time indicator data."""
        s = " ".join(str(sentence).split())
        oil = snapshots["DCOILWTICO"]
        fx = snapshots["DEXINUS"]

        # 1. Direct WTI / Crude oil range in USD/barrel
        m_range = RANGE_USD_RX.search(s)
        if m_range and ("oil" in s.lower() or "crude" in s.lower() or "wti" in s.lower() or topic == "fuel"):
            low, high = float(m_range.group(1)), float(m_range.group(2))
            actual = oil["value"]
            if actual > high:
                drift_pct = ((actual - high) / high) * 100.0
                status = "breached" if drift_pct > 5.0 else "warning"
                summary = (
                    f"WTI Crude is ${actual:.2f}/bbl (as of {oil['available_at']}), "
                    f"+${actual - high:.2f}/bbl (+{drift_pct:.1f}%) above the assumed USD {low:.0f}–{high:.0f}/bbl corridor."
                )
            elif actual < low:
                drift_pct = ((actual - low) / low) * 100.0
                status = "warning"
                summary = (
                    f"WTI Crude is ${actual:.2f}/bbl, ${low - actual:.2f}/bbl below the assumed USD {low:.0f}–{high:.0f}/bbl corridor."
                )
            else:
                drift_pct = 0.0
                status = "holding"
                summary = (
                    f"WTI Crude is ${actual:.2f}/bbl, inside the assumed USD {low:.0f}–{high:.0f}/bbl range."
                )
            return {
                "monitored": True,
                "series_id": "DCOILWTICO",
                "mapping_kind": "direct",
                "claim_type": "range",
                "assumed_label": f"USD {low:.0f} – {high:.0f} / bbl",
                "low": low,
                "high": high,
                "target": round((low + high) / 2, 2),
                "actual_value": actual,
                "actual_label": f"USD {actual:.2f} / bbl",
                "observation_date": oil["observation_date"],
                "available_at": oil["available_at"],
                "drift_pct": round(drift_pct, 1),
                "status": status,
                "explanation": summary,
            }

        # 2. Direct Crude oil target ("near USD 72 per barrel")
        m_target_oil = TARGET_USD_RX.search(s)
        if m_target_oil and ("oil" in s.lower() or "crude" in s.lower() or "wti" in s.lower()):
            target = float(m_target_oil.group(1))
            is_downside_scenario = "downside" in s.lower() or "if " in s.lower()
            actual = oil["value"]
            drift_pct = ((actual - target) / target) * 100.0
            if is_downside_scenario:
                status = "breached" if actual >= target else ("warning" if actual >= target * 0.92 else "holding")
                summary = (
                    f"Downside trigger at USD {target:.0f}/bbl: live WTI Crude is ${actual:.2f}/bbl "
                    f"({'TRIGGERED' if actual >= target else 'below trigger'})."
                )
                low, high = 0.0, target
            else:
                low, high = round(target * 0.90, 2), round(target * 1.10, 2)
                if abs(drift_pct) > 15.0:
                    status = "breached"
                elif abs(drift_pct) > 8.0:
                    status = "warning"
                else:
                    status = "holding"
                summary = (
                    f"WTI Crude is ${actual:.2f}/bbl vs. assumed ~USD {target:.0f}/bbl "
                    f"({drift_pct:+.1f}% deviation)."
                )
            return {
                "monitored": True,
                "series_id": "DCOILWTICO",
                "mapping_kind": "direct",
                "claim_type": "scenario_threshold" if is_downside_scenario else "target",
                "assumed_label": f"{'>= ' if is_downside_scenario else '~'}USD {target:.0f} / bbl",
                "low": low,
                "high": high,
                "target": target,
                "actual_value": actual,
                "actual_label": f"USD {actual:.2f} / bbl",
                "observation_date": oil["observation_date"],
                "available_at": oil["available_at"],
                "drift_pct": round(drift_pct, 1),
                "status": status,
                "explanation": summary,
            }

        # 3. Direct Crude oil upper/lower threshold ("Prices above USD 90 per barrel...", "stay below USD 80")
        m_above_oil = ABOVE_USD_RX.search(s)
        if m_above_oil and ("barrel" in s.lower() or "oil" in s.lower() or "crude" in s.lower() or "wti" in s.lower()):
            cap = float(m_above_oil.group(1))
            actual = oil["value"]
            drift_pct = ((actual - cap) / cap) * 100.0
            if actual >= cap:
                status = "breached"
                summary = (
                    f"Threshold USD {cap:.0f}/bbl is EXCEEDED: WTI Crude is ${actual:.2f}/bbl ({drift_pct:+.1f}%)."
                )
            elif actual >= cap * 0.92:
                status = "warning"
                summary = (
                    f"WTI Crude (${actual:.2f}/bbl) is within ${(cap - actual):.2f} of the USD {cap:.0f}/bbl threshold."
                )
            else:
                status = "holding"
                summary = (
                    f"WTI Crude (${actual:.2f}/bbl) remains safely below the USD {cap:.0f}/bbl threshold."
                )
            return {
                "monitored": True,
                "series_id": "DCOILWTICO",
                "mapping_kind": "direct",
                "claim_type": "upper_threshold",
                "assumed_label": f"<= USD {cap:.0f} / bbl",
                "low": 0.0,
                "high": cap,
                "target": cap,
                "actual_value": actual,
                "actual_label": f"USD {actual:.2f} / bbl",
                "observation_date": oil["observation_date"],
                "available_at": oil["available_at"],
                "drift_pct": round(drift_pct, 1),
                "status": status,
                "explanation": summary,
            }

        # 4. Proxy Diesel price cap in Rs./litre
        m_diesel = DIESEL_CAP_RX.search(s)
        if m_diesel and "diesel" in s.lower():
            cap_inr = float(m_diesel.group(1))
            actual_oil = oil["value"]
            fx_factor = (fx["value"] - 84.0) * 0.35
            implied_diesel = round(
                DIESEL_BASE_INR_PER_L
                + (actual_oil - DIESEL_BASE_CRUDE_USD) * DIESEL_PASS_THROUGH_INR_PER_USD_BBL
                + fx_factor,
                1,
            )
            drift_pct = ((implied_diesel - cap_inr) / cap_inr) * 100.0
            if implied_diesel > cap_inr:
                status = "breached"
                summary = (
                    f"Proxy Alert (via DCOILWTICO @ ${actual_oil:.2f}/bbl & DEXINUS @ {fx['value']:.2f}): "
                    f"Implied unhedged diesel cost is ~Rs. {implied_diesel:.1f}/L, breaching the Rs. {cap_inr:.0f}/L cap ({drift_pct:+.1f}%)."
                )
            elif implied_diesel >= cap_inr - 3.0:
                status = "warning"
                summary = (
                    f"Proxy Watch (via DCOILWTICO @ ${actual_oil:.2f}/bbl): "
                    f"Implied diesel pressure (~Rs. {implied_diesel:.1f}/L) is approaching the Rs. {cap_inr:.0f}/L limit."
                )
            else:
                status = "holding"
                summary = (
                    f"Proxy OK (via DCOILWTICO @ ${actual_oil:.2f}/bbl): "
                    f"Implied diesel cost (~Rs. {implied_diesel:.1f}/L) is below the Rs. {cap_inr:.0f}/L cap."
                )
            return {
                "monitored": True,
                "series_id": "DCOILWTICO",
                "mapping_kind": "proxy",
                "claim_type": "proxy_cap",
                "assumed_label": f"<= Rs. {cap_inr:.0f} / L (proxy)",
                "low": 70.0,
                "high": cap_inr,
                "target": cap_inr,
                "actual_value": implied_diesel,
                "actual_label": f"~Rs. {implied_diesel:.1f} / L (WTI ${actual_oil:.1f})",
                "observation_date": oil["observation_date"],
                "available_at": oil["available_at"],
                "drift_pct": round(drift_pct, 1),
                "status": status,
                "explanation": summary,
            }

        # 5. Direct Currency INR/USD range
        m_inr_range = INR_RANGE_RX.search(s)
        if m_inr_range and (
            "rupee" in s.lower() or "exchange rate" in s.lower() or "usd" in s.lower() or "dollar" in s.lower() or "inr" in s.lower()
        ):
            low, high = float(m_inr_range.group(1)), float(m_inr_range.group(2))
            actual = fx["value"]
            if actual > high:
                drift_pct = ((actual - high) / high) * 100.0
                status = "breached" if drift_pct > 3.0 else "warning"
            elif actual < low:
                drift_pct = ((actual - low) / low) * 100.0
                status = "warning"
            else:
                drift_pct = 0.0
                status = "holding"
            return {
                "monitored": True,
                "series_id": "DEXINUS",
                "mapping_kind": "direct",
                "claim_type": "range",
                "assumed_label": f"INR {low:.1f} – {high:.1f} / USD",
                "low": low,
                "high": high,
                "target": round((low + high) / 2, 2),
                "actual_value": actual,
                "actual_label": f"INR {actual:.2f} / USD",
                "observation_date": fx["observation_date"],
                "available_at": fx["available_at"],
                "drift_pct": round(drift_pct, 1),
                "status": status,
                "explanation": f"DEXINUS spot rate is {actual:.2f} INR/USD vs. assumed INR {low:.1f}–{high:.1f}/USD range.",
            }

        # 6. Direct Currency INR/USD target
        m_inr_target = INR_TARGET_RX.search(s)
        if m_inr_target and (
            "rupee" in s.lower() or "exchange rate" in s.lower() or "usd" in s.lower() or "dollar" in s.lower()
        ):
            target = float(m_inr_target.group(1))
            actual = fx["value"]
            drift_pct = ((actual - target) / target) * 100.0
            low, high = round(target * 0.96, 2), round(target * 1.04, 2)
            if abs(drift_pct) > 6.0:
                status = "breached"
            elif abs(drift_pct) > 3.0:
                status = "warning"
            else:
                status = "holding"
            summary = (
                f"DEXINUS spot rate is {actual:.2f} INR/USD (as of {fx['available_at']}) vs. assumed "
                f"INR {target:.1f}/USD ({drift_pct:+.1f}% weaker rupee)."
            )
            return {
                "monitored": True,
                "series_id": "DEXINUS",
                "mapping_kind": "direct",
                "claim_type": "target",
                "assumed_label": f"~INR {target:.1f} / USD",
                "low": low,
                "high": high,
                "target": target,
                "actual_value": actual,
                "actual_label": f"INR {actual:.2f} / USD",
                "observation_date": fx["observation_date"],
                "available_at": fx["available_at"],
                "drift_pct": round(drift_pct, 1),
                "status": status,
                "explanation": summary,
            }

        # 7. Direct Currency INR/USD upper threshold
        m_inr_above = INR_ABOVE_RX.search(s)
        if m_inr_above and (
            "usd" in s.lower() or "dollar" in s.lower() or "import" in s.lower() or "rupee" in s.lower() or "exchange" in s.lower()
        ):
            cap = float(m_inr_above.group(1))
            actual = fx["value"]
            drift_pct = ((actual - cap) / cap) * 100.0
            if actual >= cap:
                status = "breached"
                summary = (
                    f"FX Threshold INR {cap:.1f}/USD is BREACHED: DEXINUS is at {actual:.2f} INR/USD ({drift_pct:+.1f}%)."
                )
            elif actual >= cap * 0.97:
                status = "warning"
                summary = (
                    f"DEXINUS ({actual:.2f} INR/USD) is approaching the INR {cap:.1f}/USD threshold."
                )
            else:
                status = "holding"
                summary = (
                    f"DEXINUS ({actual:.2f} INR/USD) is below the INR {cap:.1f}/USD threshold."
                )
            return {
                "monitored": True,
                "series_id": "DEXINUS",
                "mapping_kind": "direct",
                "claim_type": "upper_threshold",
                "assumed_label": f"<= INR {cap:.1f} / USD",
                "low": 70.0,
                "high": cap,
                "target": cap,
                "actual_value": actual,
                "actual_label": f"INR {actual:.2f} / USD",
                "observation_date": fx["observation_date"],
                "available_at": fx["available_at"],
                "drift_pct": round(drift_pct, 1),
                "status": status,
                "explanation": summary,
            }

        # 8. Unmapped topic assumptions (e.g., demand, labor, rates, supplier, regulation)
        if is_assumption:
            m_rate = RATE_TARGET_RX.search(s)
            m_growth = GROWTH_PCT_RX.search(s)
            extracted_val = (
                f"{m_rate.group(1)}% interest rate"
                if m_rate
                else (f"{m_growth.group(1)}% rate/growth" if m_growth else "Qualitative / internal metric")
            )
            return {
                "monitored": False,
                "series_id": None,
                "mapping_kind": "unmapped",
                "claim_type": "unmapped_assumption",
                "assumed_label": extracted_val,
                "low": None,
                "high": None,
                "target": None,
                "actual_value": None,
                "actual_label": "No live FRED series registered",
                "observation_date": None,
                "available_at": None,
                "drift_pct": None,
                "status": "unmapped",
                "explanation": (
                    f"Detected '{topic}' assumption ({extracted_val}). "
                    f"Add a '{topic}' indicator series to config/source_registry.json to enable live monitoring."
                ),
            }

        return {
            "monitored": False,
            "series_id": None,
            "mapping_kind": "none",
            "claim_type": "none",
            "assumed_label": "",
            "low": None,
            "high": None,
            "target": None,
            "actual_value": None,
            "actual_label": "",
            "observation_date": None,
            "available_at": None,
            "drift_pct": None,
            "status": "not_assumption",
            "explanation": "",
        }

    # ---------- Full Document Analysis Pipeline ----------
    def analyze_document(
        self,
        doc_source: str = "plan1",
        as_of: str | None = "2026-10-16",
        mode: str = "hybrid",
        threshold: float = 0.45,
        override_oil: float | None = None,
        override_fx: float | None = None,
        custom_text: str | None = None,
        uploaded_bytes: bytes | None = None,
        uploaded_filename: str | None = None,
    ) -> dict:
        """Run reader -> detector (Model A + Rules v1 + Human Reviews) -> indicator monitor on a plan."""
        warnings_list = []
        if uploaded_bytes is not None and uploaded_filename:
            suffix = Path(uploaded_filename).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_bytes)
                tmp_path = Path(tmp.name)
            try:
                res = reader.read_document(tmp_path, doc_id=Path(uploaded_filename).stem)
            finally:
                tmp_path.unlink(missing_ok=True)
            sentences_df, audit_df, warnings_list = res.sentences, res.audit, res.warnings
            doc_title = f"{uploaded_filename} (Uploaded Document)"
        elif custom_text is not None and custom_text.strip():
            raw_lines = [
                reader.Line(line, page=1, heading=False, para_end=True)
                for line in custom_text.splitlines()
                if line.strip()
            ]
            sentences_df, audit_df = reader.clean_lines(raw_lines, n_pages=1, doc_id="custom")
            warnings_list = ["Custom text mode: lines treated as single-page paragraphs."]
            doc_title = "Custom Business Plan Text"
        elif doc_source == "plan3":
            gen_path = GENERATED_PDF_DIR / "plan3.pdf"
            if not gen_path.exists():
                self.generate_sample_pdf()
            res = reader.read_document(gen_path, doc_id="plan3")
            sentences_df, audit_df, warnings_list = res.sentences, res.audit, res.warnings
            doc_title = "AeroVantage Freight & Aviation Pvt. Ltd. — Strategic Plan 2026–2029 (plan3.pdf)"
        elif doc_source == "plan2":
            raw_path = self.root / "data" / "raw" / "plan2.docx"
            doc_title = "GreenCart Retail Pvt. Ltd. — Business Plan 2026–2028 (plan2.docx)"
            res = reader.read_document(raw_path)
            sentences_df, audit_df, warnings_list = res.sentences, res.audit, res.warnings
        else:
            raw_path = self.root / "data" / "raw" / "plan1.pdf"
            doc_title = "NorthStar Freight Pvt. Ltd. — Business Plan 2026–2029 (plan1.pdf)"
            res = reader.read_document(raw_path)
            sentences_df, audit_df, warnings_list = res.sentences, res.audit, res.warnings

        doc_id_str = str(sentences_df["doc_id"].iloc[0])
        records_list = sentences_df.to_dict("records")

        # Append any user-injected custom clauses for this document
        extra_clauses = self.injected_clauses.get(doc_id_str, [])
        if extra_clauses:
            records_list.extend(extra_clauses)

        snapshots = self.get_indicator_snapshot(as_of=as_of, override_oil=override_oil, override_fx=override_fx)

        rows = []
        monitored_claims = []
        for row in records_list:
            sent_id = str(row["sentence_id"])
            sent_text = row["sentence"]
            explained = self.explain_sentence(sent_text)
            prob = explained["model_a"]["probability"]
            r_info = explained["rules_v1"]

            # Check if a human review override exists for this sentence_id
            human_rev = self.human_reviews.get(sent_id)
            if human_rev is not None:
                if human_rev["review_status"] == "needs_review" or human_rev["is_assumption"] == "":
                    is_assump = False
                    needs_review = True
                    decision_source = "human_flagged_review"
                    topic = human_rev.get("topic") or r_info["guessed_topic"]
                else:
                    is_assump = human_rev["is_assumption"] == "1"
                    needs_review = False
                    decision_source = "human_reviewed ✓"
                    topic = human_rev["topic"] if is_assump else ""
            elif mode == "model_only":
                is_assump = prob >= threshold
                decision_source = "model_a"
                needs_review = abs(prob - threshold) <= 0.07
                topic = r_info["topic"] if is_assump else ""
            elif mode == "rules_only":
                is_assump = r_info["is_assumption"] == "1" and r_info["is_confident"]
                decision_source = "rules_v1"
                needs_review = not r_info["is_confident"]
                topic = r_info["topic"] if is_assump else ""
            else:
                is_own_metric = bool(
                    re.search(
                        r"^(?:if these assumptions hold,\s*)?(?:revenue|sales|online orders)\s+(?:is|are)\s+projected\b",
                        sent_text,
                        re.IGNORECASE,
                    )
                )
                has_quant_bound = bool(
                    ABOVE_USD_RX.search(sent_text)
                    or INR_ABOVE_RX.search(sent_text)
                    or RANGE_USD_RX.search(sent_text)
                    or INR_RANGE_RX.search(sent_text)
                    or ("downside scenario" in sent_text.lower() and TARGET_USD_RX.search(sent_text))
                )
                has_passive_external = bool(PASSIVE_ASSUMP_RX.search(sent_text)) and not is_own_metric

                if r_info["is_confident"] and r_info["is_assumption"] in ("0", "1"):
                    if r_info["is_assumption"] == "1":
                        is_assump = True
                        decision_source = "rules_v1 (confident)"
                    elif has_quant_bound:
                        is_assump = True
                        decision_source = "sensitivity_rule + model_a"
                    elif has_passive_external:
                        is_assump = True
                        decision_source = "hybrid (passive external)"
                    elif prob >= max(threshold + 0.12, 0.55):
                        is_assump = True
                        decision_source = "model_a (override)"
                    else:
                        is_assump = False
                        decision_source = "rules_v1 (confident)"
                    needs_review = False
                else:
                    if has_passive_external or r_info["is_assumption"] == "1" or prob >= threshold or has_quant_bound:
                        is_assump = not is_own_metric
                        decision_source = "hybrid (model_a + rules_v1)"
                    else:
                        is_assump = False
                        decision_source = "hybrid (model_a + rules_v1)"
                    needs_review = not is_assump and (prob >= (threshold - 0.08) or not r_info["is_confident"])

                topic = r_info["topic"] if is_assump else (r_info["guessed_topic"] if r_info["matched_topics"] else "")
                if is_assump and (not topic or topic == "other"):
                    low_s = sent_text.lower()
                    if any(w in low_s for w in ("market", "freight rates", "household spending", "churn", "volumes", "cargo demand")):
                        topic = "demand"
                    elif any(w in low_s for w in ("tyre", "maintenance", "freight and store rents")):
                        topic = "supplier"

            claim = self.evaluate_claim(sent_text, topic or "other", is_assump, snapshots)

            percents = row["percents"] if isinstance(row["percents"], list) else []
            money = row["money"] if isinstance(row["money"], list) else []

            enriched = {
                "sentence_id": sent_id,
                "doc_id": str(row["doc_id"]),
                "page_start": None if pd.isna(row["page_start"]) else int(row["page_start"]),
                "page_end": None if pd.isna(row["page_end"]) else int(row["page_end"]),
                "pages": str(row["pages"] or ""),
                "section": str(row["section"] or ""),
                "sentence": sent_text,
                "n_words": int(row["n_words"]),
                "n_occurrences": int(row["n_occurrences"]),
                "percents": percents,
                "money": money,
                "extraction_status": row["extraction_status"],
                "is_assumption": bool(is_assump),
                "needs_review": bool(needs_review),
                "is_human_reviewed": human_rev is not None,
                "is_injected": sent_id.startswith(f"{doc_id_str}-custom-"),
                "topic": topic if is_assump else "",
                "guessed_topic": r_info["guessed_topic"],
                "decision_source": decision_source,
                "model_prob": prob,
                "model_top_positive": explained["model_a"]["top_positive"],
                "model_top_negative": explained["model_a"]["top_negative"],
                "rule_is_assumption": r_info["is_assumption"],
                "rule_confidence": r_info["confidence"],
                "rule_reason": r_info["reason"],
                "rule_signals": r_info["signals"],
                "claim": claim,
            }
            rows.append(enriched)
            if claim["monitored"]:
                monitored_claims.append({
                    "sentence_id": sent_id,
                    "section": str(row["section"] or ""),
                    "page": str(row["pages"] or "—"),
                    "sentence": sent_text,
                    "topic": topic or r_info["guessed_topic"],
                    "model_prob": prob,
                    "rule_reason": r_info["reason"],
                    **claim,
                })

        n_assumptions = sum(1 for r in rows if r["is_assumption"])
        n_breached = sum(1 for c in monitored_claims if c["status"] == "breached")
        n_warning = sum(1 for c in monitored_claims if c["status"] == "warning")
        n_holding = sum(1 for c in monitored_claims if c["status"] == "holding")
        n_human_reviewed = sum(1 for r in rows if r["is_human_reviewed"])

        topic_counts = {}
        for r in rows:
            if r["is_assumption"]:
                t = r["topic"] or "other"
                topic_counts[t] = topic_counts.get(t, 0) + 1

        audit_records = audit_df.fillna("").to_dict("records")
        audit_summary = audit_df["reason"].value_counts().to_dict() if not audit_df.empty else {}

        oil_val = snapshots["DCOILWTICO"]["value"]
        fx_val = snapshots["DEXINUS"]["value"]
        doc_key = doc_id_str.lower()

        if len(monitored_claims) > 0:
            avg_abs_drift = sum(abs(c.get("drift_pct") or 0.0) for c in monitored_claims) / len(monitored_claims)
            raw_score = 100.0 - (n_breached * 12.5) - (n_warning * 5.0) - (avg_abs_drift * 0.85)
            viability_index = int(max(8, min(100, round(raw_score))))
        else:
            viability_index = 92

        if n_breached >= 2 or viability_index < 55:
            regime_label = "CRITICAL REGIME BREACH"
        elif n_breached == 1 or n_warning >= 1 or viability_index < 80:
            regime_label = "ELEVATED MACRO DRIFT"
        else:
            regime_label = "NOMINAL EQUILIBRIUM"

        if "plan2" in doc_key:
            base_margin_pct = 3.4
            fx_excess = max(0.0, fx_val - 83.0)
            diesel_excess = max(0.0, oil_val - 75.0)
            margin_delta_pct = -(fx_excess * 0.085 + diesel_excess * 0.025)
            projected_margin_pct = round(max(0.2, base_margin_pct + margin_delta_pct), 2)
            sensitivity_narrative = (
                f"Base 2025 net margin: {base_margin_pct:.1f}% on Rs. 118 Cr sales (18% imported goods). "
                f"At INR {fx_val:.2f}/USD & WTI ${oil_val:.2f}/bbl, estimated net margin is {projected_margin_pct:.2f}% "
                f"({margin_delta_pct * 100:+.0f} bps impact)."
            )
        elif "plan3" in doc_key:
            base_margin_pct = 11.4
            oil_excess = max(0.0, oil_val - 75.0)
            fx_excess = max(0.0, fx_val - 83.5)
            margin_delta_pct = -(oil_excess * 0.16 + fx_excess * 0.06)
            projected_margin_pct = round(max(1.8, base_margin_pct + margin_delta_pct), 2)
            sensitivity_narrative = (
                f"Base aviation operating margin: {base_margin_pct:.1f}% on Rs. 164 Cr revenue (fuel = 41% opex). "
                f"At WTI ${oil_val:.2f}/bbl & INR {fx_val:.2f}/USD, modeled margin shifts to {projected_margin_pct:.2f}% "
                f"({margin_delta_pct * 100:+.0f} bps)."
            )
        else:
            base_margin_pct = 9.2
            if oil_val >= 95.0:
                margin_delta_pct = -3.9 - max(0.0, (oil_val - 95.0) * 0.08)
            elif oil_val >= 90.0:
                margin_delta_pct = -3.0 - (oil_val - 90.0) * 0.18
            elif oil_val > 72.0:
                margin_delta_pct = -(oil_val - 72.0) * 0.14
            else:
                margin_delta_pct = (72.0 - oil_val) * 0.06
            fx_drag = max(0.0, (fx_val - 84.0) * 0.03)
            margin_delta_pct -= fx_drag
            projected_margin_pct = round(max(1.5, base_margin_pct + margin_delta_pct), 2)
            sensitivity_narrative = (
                f"Base operating margin: {base_margin_pct:.1f}% (fuel = 38% of opex). "
                f"At WTI ${oil_val:.2f}/bbl & INR {fx_val:.2f}/USD, modeled margin shifts to {projected_margin_pct:.2f}% "
                f"({margin_delta_pct * 100:+.0f} bps)."
            )

        chronograph_dates = [
            "2026-07-08",
            "2026-07-15",
            "2026-07-22",
            "2026-07-29",
            "2026-08-05",
            "2026-08-12",
            "2026-08-19",
            "2026-08-26",
            "2026-09-02",
            "2026-09-09",
            "2026-09-16",
            "2026-09-23",
            "2026-09-30",
            "2026-10-07",
            "2026-10-16",
        ]

        return {
            "doc_id": doc_id_str,
            "doc_title": doc_title,
            "as_of": snapshots["DCOILWTICO"]["as_of_requested"],
            "mode": mode,
            "threshold": threshold,
            "summary": {
                "total_sentences": len(rows),
                "ok_sentences": sum(1 for r in rows if r["extraction_status"] == "ok"),
                "assumptions_detected": n_assumptions,
                "needs_review": sum(1 for r in rows if r["needs_review"]),
                "human_reviewed": n_human_reviewed,
                "injected_clauses": len(extra_clauses),
                "monitored_claims": len(monitored_claims),
                "breached_claims": n_breached,
                "warning_claims": n_warning,
                "holding_claims": n_holding,
                "removed_lines": len(audit_records),
                "topic_counts": topic_counts,
                "audit_counts": audit_summary,
            },
            "financial_telemetry": {
                "viability_index": viability_index,
                "regime_label": regime_label,
                "base_margin_pct": base_margin_pct,
                "projected_margin_pct": projected_margin_pct,
                "margin_delta_bps": int(round(margin_delta_pct * 100)),
                "narrative": sensitivity_narrative,
                "chronograph_dates": chronograph_dates,
            },
            "warnings": warnings_list,
            "indicators": snapshots,
            "monitored_claims": monitored_claims,
            "sentences": rows,
            "audit_log": audit_records,
        }

    # ---------- Time-Series Explorer Payload ----------
    def get_series_explorer_data(self, years: int = 5) -> dict:
        checksum_problems = sources.verify_log(self.root / sources.LOG_PATH)
        log_df = sources.read_log(self.root / sources.LOG_PATH)
        registry_records = self.registry_df.fillna("").to_dict("records")

        series_payload = {}
        for sid, df in self.weekly_series.items():
            if years > 0:
                cutoff = df["observation_date"].max() - pd.Timedelta(days=int(years * 365.25))
                sub = df[df["observation_date"] >= cutoff].copy()
            else:
                sub = df.copy()

            outliers = df[df["is_outlier"]].tail(15)
            filled = df[df["is_filled"]].tail(15)

            series_payload[sid] = {
                "series_id": sid,
                "summary": series.summary(df),
                "points": [
                    {
                        "observation_date": str(r.observation_date.date()),
                        "available_at": str(r.available_at.date()),
                        "value": None if pd.isna(r.value) else round(float(r.value), 3),
                        "n_daily_obs": int(r.n_daily_obs),
                        "is_missing": bool(r.is_missing),
                        "is_filled": bool(r.is_filled),
                        "is_outlier": bool(r.is_outlier),
                        "robust_z": None if pd.isna(r.robust_z) else round(float(r.robust_z), 2),
                    }
                    for r in sub.itertuples()
                ],
                "recent_outliers": [
                    {
                        "observation_date": str(r.observation_date.date()),
                        "available_at": str(r.available_at.date()),
                        "value": round(float(r.value), 3),
                        "robust_z": round(float(r.robust_z), 2),
                    }
                    for r in outliers.itertuples()
                ],
                "recent_filled": [
                    {
                        "observation_date": str(r.observation_date.date()),
                        "available_at": str(r.available_at.date()),
                        "value": round(float(r.value), 3),
                    }
                    for r in filled.itertuples()
                ],
            }

        return {
            "registry": registry_records,
            "download_log": log_df.to_dict("records"),
            "checksum_verified": len(checksum_problems) == 0,
            "checksum_problems": checksum_problems,
            "series": series_payload,
        }

    # ---------- Model A, Labeling & Benchmark Lab Payload ----------
    def get_model_lab_data(self, retrain: bool = False) -> dict:
        data = model_a.build_dataset(
            labels_path=str(self.root / model_a.LABELS),
            silver_auto_path=str(self.root / model_a.SILVER_AUTO),
            candidates_paths=[str(self.root / p) for p in model_a.CANDIDATES],
            frozen_docs_path=str(self.root / model_a.FROZEN_DOCS),
        )

        # Incorporate human reviews from UI if present
        if self.human_reviews:
            extra_rows = []
            for sid, rev in self.human_reviews.items():
                if rev.get("is_assumption") in ("0", "1") and rev.get("sentence"):
                    extra_rows.append({
                        "sentence_id": sid,
                        "doc_id": rev.get("doc_id", "ui_doc"),
                        "y": int(rev["is_assumption"]),
                        "source": "hand",
                        "sentence": rev["sentence"],
                        "split": "dev",
                    })
            if extra_rows:
                extra_df = pd.DataFrame(extra_rows)
                data = pd.concat([data[~data["sentence_id"].isin(extra_df["sentence_id"])], extra_df], ignore_index=True)

        if retrain:
            dev = data[data["split"] == "dev"]
            self.model = model_a.make_model().fit(dev["sentence"], dev["y"])

        comp_table, _ = model_a.compare_systems(data)
        systems = []
        for sys_name, row in comp_table.iterrows():
            systems.append({
                "system": str(sys_name),
                "n": int(row["n"]),
                "precision": round(float(row["precision"]), 3),
                "recall": round(float(row["recall"]), 3),
                "f1": round(float(row["f1"]), 3),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tn": int(row["tn"]),
            })

        split_breakdown = (
            data.groupby(["split", "source"])["y"]
            .agg(total="count", assumptions="sum")
            .reset_index()
            .to_dict("records")
        )
        for r in split_breakdown:
            r["total"] = int(r["total"])
            r["assumptions"] = int(r["assumptions"])
            r["not_assumptions"] = r["total"] - r["assumptions"]

        words = model_a.top_words(self.model, k=15)

        test_metrics_path = self.root / model_a.TEST_REPORT
        test_metrics = (
            json.loads(test_metrics_path.read_text())
            if test_metrics_path.exists()
            else {}
        )

        audit_df = autolabel.load_audit(self.root / "data" / "labels" / "audit_log.csv")
        n_audit = len(audit_df)
        agreed = int((audit_df["agreed"] == "True").sum()) if n_audit else 0
        w_low, w_high = autolabel.wilson_interval(agreed, n_audit)

        labels_df = labeler.load_labels(self.root / model_a.LABELS)
        label_stats = labeler.status(labels_df)

        err_hand_path = self.root / "reports" / "model_a_errors_hand_only.csv"
        err_rules_path = self.root / "reports" / "model_a_errors_with_rules.csv"
        errors_hand = (
            pd.read_csv(err_hand_path).fillna("").to_dict("records")
            if err_hand_path.exists()
            else []
        )
        errors_rules = (
            pd.read_csv(err_rules_path).fillna("").to_dict("records")
            if err_rules_path.exists()
            else []
        )

        bench_path = self.root / "results" / "benchmark.csv"
        pc_path = self.root / "results" / "pc_info.json"
        benchmark_rows = (
            pd.read_csv(bench_path).round(2).to_dict("records")
            if bench_path.exists()
            else []
        )
        pc_info = json.loads(pc_path.read_text()) if pc_path.exists() else {}

        return {
            "cv_comparison": systems,
            "dataset_breakdown": split_breakdown,
            "top_words": words,
            "test_metrics": test_metrics,
            "audit": {
                "checked": n_audit,
                "agreed": agreed,
                "agreement_rate": round(agreed / n_audit, 3) if n_audit else 0.0,
                "wilson_low": round(w_low, 3),
                "wilson_high": round(w_high, 3),
            },
            "label_status": label_stats,
            "ui_reviews_count": len(self.human_reviews),
            "ui_reviews": list(self.human_reviews.values()),
            "errors_hand_only": errors_hand,
            "errors_with_rules": errors_rules,
            "benchmark": {
                "pc_info": pc_info,
                "measurements": benchmark_rows,
            },
        }
