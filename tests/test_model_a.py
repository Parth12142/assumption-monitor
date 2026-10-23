import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from assumption_monitor import model_a

POSITIVE = [
    "We assume that fuel prices will remain stable in case {i}.",
    "Management expects interest rates to remain low through fiscal 2021 in case {i}.",
    "Customer demand is projected to grow by 8% next year in case {i}.",
]
NEGATIVE = [
    "The company operates {i} stores in the state of Ohio today.",
    "Our main office is located in a leased building in Texas number {i}.",
    "The company was founded in 1985 and employs {i} people in total.",
]


def make_files(tmp_path):
    """Ten hand-labeled documents (two of them frozen as test), plus 4 documents with rule labels only."""
    cand, labels, auto = [], [], []
    for d in range(10):
        for k in range(6):
            positive = k < 3
            template = (POSITIVE if positive else NEGATIVE)[k % 3]
            sid = f"d{d}-{k:05d}"
            cand.append({"sentence_id": sid, "doc_id": f"d{d}", "section": "s", "sentence": template.format(i=d * 10 + k)})
            labels.append({"sentence_id": sid, "doc_id": f"d{d}", "is_assumption": "1" if positive else "0",
                           "topic": "other" if positive else "", "labeler": "me", "label_version": "v1",
                           "review_status": "reviewed" if k == 0 and d < 2 else "single", "is_synthetic": "False"})
    for d in range(4):
        for k in range(6):
            positive = k < 3
            template = (POSITIVE if positive else NEGATIVE)[k % 3]
            sid = f"s{d}-{k:05d}"
            cand.append({"sentence_id": sid, "doc_id": f"s{d}", "section": "s", "sentence": template.format(i=500 + d * 10 + k)})
            auto.append({"sentence_id": sid, "doc_id": f"s{d}", "is_assumption": "1" if positive else "0",
                         "topic": "", "labeler": "rules_v1", "label_version": "v1", "review_status": "auto",
                         "is_synthetic": "False", "confidence": "0.90", "reason": "x"})
    paths = {
        "labels": tmp_path / "labels.csv", "auto": tmp_path / "auto.csv",
        "cand": tmp_path / "cand.csv", "frozen": tmp_path / "frozen.txt",
    }
    pd.DataFrame(labels).to_csv(paths["labels"], index=False)
    pd.DataFrame(auto).to_csv(paths["auto"], index=False)
    pd.DataFrame(cand).to_csv(paths["cand"], index=False)
    paths["frozen"].write_text("d8\nd9\n")
    return paths


def dataset(tmp_path):
    p = make_files(tmp_path)
    return model_a.build_dataset(p["labels"], p["auto"], [p["cand"]], p["frozen"])


# ---------- The data table ----------
def test_dataset_has_sources_and_a_document_level_test_split(tmp_path):
    data = dataset(tmp_path)
    assert set(data["source"]) == {"hand", "audited", "rules"}
    assert set(data[data["split"] == "test"]["doc_id"]) == {"d8", "d9"}
    assert not ((data["source"] == "rules") & (data["split"] == "test")).any()
    assert data["sentence"].notna().all() and set(data["y"]) == {0, 1}


def test_too_few_labels_give_a_clear_error(tmp_path):
    p = make_files(tmp_path)
    pd.read_csv(p["labels"], dtype=str).head(10).to_csv(p["labels"], index=False)
    data = model_a.build_dataset(p["labels"], p["auto"], [p["cand"]], p["frozen"])
    with pytest.raises(model_a.ModelError):
        model_a.check_enough(data)


def test_missing_files_give_clear_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert model_a.main(["cv"]) == 1
    assert "ERROR" in capsys.readouterr().err


# ---------- Scores ----------
def test_scores_for_a_known_case():
    s = model_a.score_predictions([1, 1, 0, 0], [1, 0, 1, 0])
    assert (s["precision"], s["recall"], s["f1"]) == (0.5, 0.5, 0.5)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 1)


# ---------- Cross-validation ----------
def test_no_document_is_in_two_folds_and_only_hand_labels_are_judged(tmp_path):
    data = dataset(tmp_path)
    metrics, preds = model_a.cross_validate(data, use_rule_labels=True)
    assert preds.groupby("doc_id")["fold"].nunique().max() == 1
    hand_ids = set(data[data["source"] == "hand"]["sentence_id"])
    assert set(preds["sentence_id"]) <= hand_ids
    assert not set(preds["doc_id"]) & {"d8", "d9"}                  # test documents are never used here
    assert metrics["f1"] > 0.9


def test_rules_baseline_gives_scores(tmp_path):
    result = model_a.rules_baseline(dataset(tmp_path))
    assert result["n"] > 0 and result["precision"] > 0.9 and 0 < result["recall"] <= 1


def test_comparison_table_has_three_systems(tmp_path):
    table, _ = model_a.compare_systems(dataset(tmp_path))
    assert list(table.index) == ["rules only", "model, your labels", "model, your labels + rule labels"]


# ---------- Final model, test documents looked at once ----------
def test_final_model_is_saved_and_the_test_can_only_be_read_once(tmp_path):
    data = dataset(tmp_path)
    model_file, report_file = tmp_path / "m" / "model.joblib", tmp_path / "r" / "test.json"
    model, result = model_a.final_model(data, False, model_file, report_file)
    assert result["test_sentences"] == 12 and result["f1"] > 0.9
    assert json.loads(report_file.read_text())["test_sentences"] == 12
    assert joblib.load(model_file).predict(["We assume that diesel prices will remain stable."]).shape == (1,)

    with pytest.raises(model_a.ModelError):
        model_a.final_model(data, False, model_file, report_file)
    model_a.final_model(data, False, model_file, report_file, force=True)


def test_top_words_show_what_the_model_learned(tmp_path):
    data = dataset(tmp_path)
    model, _ = model_a.final_model(data, False, tmp_path / "m.joblib", tmp_path / "t.json")
    words = model_a.top_words(model, k=15)
    assert any(w in words["assumption"] for w in ["assume", "expects", "projected", "demand", "prices", "rates"])
    assert any(w in words["not_assumption"] for w in ["stores", "company", "office", "founded", "ohio", "texas"])


# ---------- Calibration and Thresholds (Day 13) ----------
def test_route_sentence_respects_fixed_thresholds():
    assert model_a.route_sentence(0.85) == "accept"
    assert model_a.route_sentence(0.70) == "accept"
    assert model_a.route_sentence(0.69) == "review"
    assert model_a.route_sentence(0.40) == "review"
    assert model_a.route_sentence(0.39) == "ignore"
    assert model_a.route_sentence(0.10) == "ignore"


def test_calculate_reliability_curve_and_brier_score():
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    # Perfect probabilities
    res_perfect = model_a.calculate_reliability_curve(y_true, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], n_bins=2)
    assert res_perfect["brier_score"] == 0.0
    assert len(res_perfect["reliability_curve"]) > 0

    # Overconfident wrong probabilities
    res_bad = model_a.calculate_reliability_curve(y_true, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], n_bins=2)
    assert res_bad["brier_score"] == 1.0


def test_calibrate_model_saves_artifacts_and_computes_metrics(tmp_path):
    data = dataset(tmp_path)
    model_file = tmp_path / "m" / "calibrated.joblib"
    report_file = tmp_path / "r" / "calibration.json"
    calibrated, report = model_a.calibrate_model(data, False, model_file, report_file, n_splits=3)
    assert Path(model_file).exists()
    assert Path(report_file).exists()
    assert report["method"] == "sigmoid"
    assert "calibrated_brier_score" in report
    assert "reliability_curve" in report
    assert set(report["thresholds"].keys()) == {"accept", "review"}
    assert set(report["routing_summary"].keys()) == {"accept", "review", "ignore"}


def test_calibrate_and_route_cli_commands(tmp_path, monkeypatch, capsys):
    p = make_files(tmp_path)
    # Set up repo directory structure in tmp_path
    (tmp_path / "data" / "labels").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "labels" / "labels.csv").write_text(p["labels"].read_text())
    (tmp_path / "data" / "labels" / "auto_day9.csv").write_text(p["auto"].read_text())
    (tmp_path / "data" / "labels" / "candidates.csv").write_text(p["cand"].read_text())
    (tmp_path / "data" / "labels" / "frozen_test_docs.txt").write_text(p["frozen"].read_text())
    monkeypatch.chdir(tmp_path)

    # Calibrate CLI
    assert model_a.main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "calibrated_brier_score" in out
    assert "reliability_curve" in out

    # Route CLI on assumption sentence
    assert model_a.main(["route", "--text", "We assume that fuel prices will remain stable."]) == 0
    route_out = capsys.readouterr().out
    assert "Calibrated probability:" in route_out
    assert "Action:" in route_out
