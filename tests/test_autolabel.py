import pandas as pd
import pytest

from assumption_monitor import autolabel, labeler

QUIET = lambda *a, **k: None


def write_candidates(path, sentences, doc="d1"):
    rows = [{"sentence_id": f"{doc}-{i:05d}", "doc_id": doc, "section": "section_7", "sentence": s, "has_cue": "True"}
            for i, s in enumerate(sentences, start=1)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------- The rules ----------
@pytest.mark.parametrize("sentence, label, topic", [
    ("We assume that diesel prices will remain stable throughout 2020.", "1", "fuel"),
    ("Management expects interest rates to remain low through fiscal 2021.", "1", "rates"),
    ("Customer demand is projected to grow by 8% next year.", "1", "demand"),
    ("This report contains forward-looking statements within the meaning of the Securities Litigation Reform Act.", "0", ""),
    ("Fuel prices could rise and reduce our margins.", "0", ""),
    ("Oil prices may remain volatile.", "0", ""),
    ("Net sales increased 5% to 120 million dollars in fiscal 2018.", "0", ""),
    ("We operate 22 stores in Gujarat and Maharashtra.", "0", ""),
    ("We plan to open 12 new stores during the next year.", "0", ""),
    ("We believe our facilities are adequate for our current needs.", "0", ""),
    ("The company assumed certain liabilities in connection with the acquisition.", "0", ""),
])
def test_clear_cases_get_a_confident_label(sentence, label, topic):
    result = autolabel.classify(sentence)
    assert result["is_assumption"] == label
    assert result["topic"] == topic
    assert result["confidence"] >= autolabel.CONFIDENT


@pytest.mark.parametrize("sentence", [
    "We expect that higher wages will reduce our operating margins in the coming year.",
    "In estimating fair value, we assumed an expected volatility of 40%.",
    "We expect to complete the acquisition in the third quarter.",
    "We expect revenue to increase by 5% next year.",
])
def test_mixed_cases_are_left_for_a_human(sentence):
    assert autolabel.classify(sentence)["confidence"] < autolabel.CONFIDENT


def test_topic_is_the_first_one_mentioned():
    assert autolabel.guess_topic("Higher diesel costs and new tax rules will matter.") == "fuel"
    assert autolabel.guess_topic("New tax rules and higher diesel costs will matter.") == "regulation"
    assert autolabel.guess_topic("The office is in Ohio.") == "other"


# ---------- Labeling a whole file ----------
def test_autolabel_writes_the_expected_columns_and_statuses(tmp_path):
    cand = write_candidates(tmp_path / "c.csv", [
        "We assume that diesel prices will remain stable throughout 2020.",
        "We operate 22 stores in Gujarat and Maharashtra.",
        "We expect revenue to increase by 5% next year.",
    ])
    table = autolabel.autolabel(cand, tmp_path / "auto.csv")
    assert list(table.columns) == autolabel.AUTO_COLUMNS
    assert table["review_status"].tolist() == ["auto", "auto", "needs_review"]
    assert table["is_assumption"].tolist() == ["1", "0", ""]
    assert table["labeler"].eq("rules_v1").all()
    assert autolabel.summarize(table) == {
        "sentences": 3, "auto_assumption": 1, "auto_not_assumption": 1, "needs_review": 1,
    }


# ---------- The audit ----------
def audit_setup(tmp_path):
    sentences = (
        [f"We assume that diesel prices will remain stable in case {i}." for i in range(6)]
        + [f"We operate {i} stores in Gujarat and Maharashtra." for i in range(6)]
    )
    cand = write_candidates(tmp_path / "c.csv", sentences)
    autolabel.autolabel(cand, tmp_path / "auto.csv")
    return cand


def run_audit(tmp_path, answers, n_per_class=2):
    cand = audit_setup(tmp_path)
    script = iter(answers)
    return autolabel.run_audit(cand, tmp_path / "auto.csv", tmp_path / "labels.csv", tmp_path / "audit.csv",
                               n_per_class=n_per_class, input_fn=lambda p: next(script), print_fn=QUIET)


def test_audit_saves_your_answers_as_real_labels(tmp_path):
    audit = run_audit(tmp_path, ["y", "y", "y", "y"])
    assert len(audit) == 4 and audit["agreed"].eq("True").all()
    labels = labeler.load_labels(tmp_path / "labels.csv")
    assert len(labels) == 4
    assert labels["labeler"].eq("me").all() and labels["review_status"].eq("reviewed").all()
    assert labeler.validate_labels(labels) == []


def test_changing_a_label_is_recorded_as_a_disagreement(tmp_path):
    sentences = [f"We assume that diesel prices will remain stable in case {i}." for i in range(4)]
    cand = write_candidates(tmp_path / "c.csv", sentences)       # the rules call all of them assumptions
    autolabel.autolabel(cand, tmp_path / "auto.csv")
    script = iter(["c", "n", "y"])
    audit = autolabel.run_audit(cand, tmp_path / "auto.csv", tmp_path / "labels.csv", tmp_path / "audit.csv",
                                n_per_class=2, input_fn=lambda p: next(script), print_fn=QUIET)
    assert audit["agreed"].tolist() == ["False", "True"]
    assert audit["human_is_assumption"].tolist() == ["0", "1"]
    labels = labeler.load_labels(tmp_path / "labels.csv")
    assert labels["is_assumption"].tolist() == ["0", "1"]
    assert labeler.validate_labels(labels) == []


def test_audit_can_be_stopped_and_resumed_without_repeating(tmp_path):
    run_audit(tmp_path, ["y", "q"])
    cand = tmp_path / "c.csv"
    script = iter(["y", "y", "y", "y"])
    audit = autolabel.run_audit(cand, tmp_path / "auto.csv", tmp_path / "labels.csv", tmp_path / "audit.csv",
                                n_per_class=2, input_fn=lambda p: next(script), print_fn=QUIET)
    assert len(audit) == 4
    assert audit["sentence_id"].is_unique


def test_wilson_interval_gives_a_sensible_range():
    low, high = autolabel.wilson_interval(18, 20)
    assert 0.65 < low < 0.75 and 0.95 < high < 0.99
    assert autolabel.wilson_interval(0, 0) == (0.0, 0.0)
    low, high = autolabel.wilson_interval(20, 20)
    assert high == 1.0 and low > 0.8


def test_report_text_mentions_agreement_and_the_changed_examples(tmp_path):
    sentences = [f"We assume that diesel prices will remain stable in case {i}." for i in range(4)]
    cand = write_candidates(tmp_path / "c.csv", sentences)
    auto = autolabel.autolabel(cand, tmp_path / "auto.csv")
    script = iter(["c", "n", "y"])
    audit = autolabel.run_audit(cand, tmp_path / "auto.csv", tmp_path / "labels.csv", tmp_path / "audit.csv",
                                n_per_class=2, input_fn=lambda p: next(script), print_fn=QUIET)
    text = autolabel.audit_report(audit, auto, labeler.load_candidates(cand))
    assert "Checked 2 rule labels: you agreed with 1 (50%)" in text
    assert "Plausible true agreement" in text
    assert "Examples you changed" in text


# ---------- Frozen test documents ----------
def test_test_documents_are_frozen_and_cannot_be_changed(tmp_path):
    rows = [{"sentence_id": f"d{i}-00001", "doc_id": f"d{i}", "section": "s", "sentence": "x", "has_cue": "True"} for i in range(10)]
    cand = tmp_path / "c.csv"
    pd.DataFrame(rows).to_csv(cand, index=False)
    out = tmp_path / "frozen.txt"
    chosen = autolabel.freeze_test_docs(cand, out, share=0.2)
    assert len(chosen) == 2 and autolabel.read_doc_list(out) == set(chosen)
    with pytest.raises(labeler.LabelError):
        autolabel.freeze_test_docs(cand, out)


# ---------- Comparing the rules with your own labels ----------
def test_comparison_with_manual_labels(tmp_path):
    sentences = [
        "We assume that diesel prices will remain stable throughout 2020.",      # rules: 1 fuel
        "We operate 22 stores in Gujarat and Maharashtra.",                       # rules: 0
        "We expect revenue to increase by 5% next year.",                         # rules: unsure
        "Customer demand is projected to grow by 8% next year.",                  # rules: 1 demand
    ]
    cand = write_candidates(tmp_path / "c.csv", sentences)
    auto = autolabel.autolabel(cand, tmp_path / "auto.csv")
    hand = pd.DataFrame([
        {"sentence_id": "d1-00001", "doc_id": "d1", "is_assumption": "1", "topic": "fuel", "labeler": "me", "label_version": "v1", "review_status": "single", "is_synthetic": "False"},
        {"sentence_id": "d1-00002", "doc_id": "d1", "is_assumption": "0", "topic": "", "labeler": "me", "label_version": "v1", "review_status": "single", "is_synthetic": "False"},
        {"sentence_id": "d1-00003", "doc_id": "d1", "is_assumption": "1", "topic": "demand", "labeler": "me", "label_version": "v1", "review_status": "single", "is_synthetic": "False"},
        {"sentence_id": "d1-00004", "doc_id": "d1", "is_assumption": "1", "topic": "other", "labeler": "me", "label_version": "v1", "review_status": "single", "is_synthetic": "False"},
    ])
    result = autolabel.compare_with_manual(auto, hand)
    assert result["n"] == 4
    assert result["coverage"] == 0.75                      # sure about 3 of 4
    assert result["accuracy_when_sure"] == 1.0              # all 3 confident labels match yours
    assert result["precision"] == 1.0                      # both "assumption" calls were right
    assert result["recall"] == pytest.approx(2 / 3)        # found 2 of the 3 real assumptions
    assert result["topic_accuracy"] == 0.5                 # fuel right, demand vs other wrong
    assert autolabel.compare_with_manual(auto, hand, exclude_docs={"d1"})["n"] == 0
    assert "Compared with 4 sentences" in autolabel.compare_report(result, labeler.load_candidates(cand))


def test_missing_auto_file_gives_a_clear_error(tmp_path, capsys):
    code = autolabel.main(["report", "--auto", str(tmp_path / "none.csv"), "--audit-log", str(tmp_path / "a.csv"),
                           "--candidates", str(tmp_path / "c.csv")])
    assert code == 1 and "not found" in capsys.readouterr().err
