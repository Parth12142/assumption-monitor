import pandas as pd
import pytest

from assumption_monitor import labeler


def make_candidates(path, n=3):
    rows = [
        {"sentence_id": f"d1-{i:05d}", "doc_id": "d1", "section": "section_7",
         "sentence": f"We expect that prices will stay stable in case number {i}.", "has_cue": "True"}
        for i in range(1, n + 1)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run(tmp_path, answers, n=3):
    cand = make_candidates(tmp_path / "cand.csv", n)
    labels = tmp_path / "labels.csv"
    script = iter(answers)
    quiet = lambda *a, **k: None
    out = labeler.run_labeler(cand, labels, input_fn=lambda prompt: next(script), print_fn=quiet)
    return out, labels


# ---------- Labeling ----------
def test_answers_are_saved_and_marked_correctly(tmp_path):
    out, path = run(tmp_path, ["n", "a", "4", "s"])
    assert out["is_assumption"].tolist() == ["0", "1", ""]
    assert out["topic"].tolist() == ["", "rates", ""]
    assert out["review_status"].tolist() == ["single", "single", "needs_review"]
    assert labeler.load_labels(path).to_dict("records") == out.to_dict("records")
    assert labeler.validate_labels(out) == []


def test_work_is_saved_after_every_answer_and_resumes_later(tmp_path):
    run(tmp_path, ["n", "q"])
    assert len(labeler.load_labels(tmp_path / "labels.csv")) == 1

    cand = tmp_path / "cand.csv"
    script = iter(["a", "1", "q"])
    out = labeler.run_labeler(cand, tmp_path / "labels.csv", input_fn=lambda p: next(script), print_fn=lambda *a, **k: None)
    assert out["sentence_id"].tolist() == ["d1-00001", "d1-00002"]      # did not ask about d1-00001 again
    assert out["topic"].tolist() == ["", "fuel"]


def test_undo_removes_the_last_label_and_asks_again(tmp_path):
    out, _ = run(tmp_path, ["n", "u", "a", "1", "q"])
    assert out["sentence_id"].tolist() == ["d1-00001"]
    assert out["is_assumption"].tolist() == ["1"]
    assert out["topic"].tolist() == ["fuel"]


def test_back_from_topic_menu_and_wrong_keys_do_not_save(tmp_path):
    out, _ = run(tmp_path, ["x", "a", "b", "a", "99", "8", "q"])
    assert out["topic"].tolist() == ["other"]
    assert len(out) == 1


def test_undo_with_nothing_to_undo_is_harmless(tmp_path):
    out, _ = run(tmp_path, ["u", "q"])
    assert len(out) == 0


# ---------- Checks ----------
def test_validation_finds_problems():
    bad = pd.DataFrame([
        {"sentence_id": "a", "doc_id": "d", "is_assumption": "1", "topic": "", "labeler": "me", "label_version": "v1", "review_status": "single", "is_synthetic": "False"},
        {"sentence_id": "b", "doc_id": "d", "is_assumption": "0", "topic": "fuel", "labeler": "me", "label_version": "v1", "review_status": "single", "is_synthetic": "False"},
        {"sentence_id": "c", "doc_id": "d", "is_assumption": "", "topic": "", "labeler": "me", "label_version": "v1", "review_status": "single", "is_synthetic": "False"},
        {"sentence_id": "a", "doc_id": "d", "is_assumption": "0", "topic": "", "labeler": "me", "label_version": "v1", "review_status": "weird", "is_synthetic": "maybe"},
    ])
    text = " | ".join(labeler.validate_labels(bad))
    for expected in ["duplicate sentence_id", "needs a topic", "topic must be empty",
                     "only allowed for needs_review", "unknown review_status", "is_synthetic"]:
        assert expected in text


def test_status_counts(tmp_path):
    out, _ = run(tmp_path, ["a", "2", "n", "s"])
    s = labeler.status(out)
    assert s["labeled"] == 2 and s["assumptions"] == 1 and s["not_assumptions"] == 1
    assert s["needs_review"] == 1 and s["topics"] == {"supplier": 1}


def test_missing_files_give_clear_errors(tmp_path, capsys):
    with pytest.raises(labeler.LabelError):
        labeler.load_candidates(tmp_path / "nothing.csv")
    code = labeler.main(["status", "--labels", str(tmp_path / "none.csv")])
    assert code == 0                                  # no labels yet is fine
    code = labeler.main(["check", "--candidates", str(tmp_path / "nothing.csv"), "--labels", str(tmp_path / "none.csv")])
    assert code == 1 and "not found" in capsys.readouterr().err


def test_only_the_chosen_sentences_are_asked(tmp_path):
    cand = make_candidates(tmp_path / "cand.csv", 3)
    script = iter(["n", "q"])
    out = labeler.run_labeler(cand, tmp_path / "labels.csv", input_fn=lambda p: next(script),
                              print_fn=lambda *a, **k: None, only_ids={"d1-00002"})
    assert out["sentence_id"].tolist() == ["d1-00002"]
