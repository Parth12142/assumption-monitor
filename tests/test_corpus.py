import json
import zipfile

import pytest

from assumption_monitor import corpus

CUE_SENTENCE = "We expect that interest rates will remain low during the coming year."
OTHER_SENTENCE = "The company was founded in 1985 and is based in the state of Ohio."


def make_filing(seed):
    """A fake annual report with 6 different cue sentences and 6 different plain sentences."""
    cue = " ".join(f"We expect that interest rates will remain low during period {seed}{i} of the plan." for i in range(6))
    other = " ".join(f"The company was founded in 19{seed}{i} and is based in the state of Ohio." for i in range(6))
    return {"filename": f"f{seed}", "cik": "1", "year": "2019", "section_7": cue + " " + other, "section_1A": "", "section_1": ""}


def make_zip(path, n=6, extra_bad=False):
    with zipfile.ZipFile(path, "w") as zf:
        for seed in range(n):
            zf.writestr(f"2019/{seed}_10K_2019_abc.json", json.dumps(make_filing(seed)))
        if extra_bad:
            zf.writestr("2019/broken_10K.json", "this is not json")
    return path


# ---------- Cleaning sentences ----------
def test_only_real_sentences_are_kept():
    text = (
        "this fragment came from the middle of a table row. "
        "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION. "
        "We expect revenue growth to continue in the next fiscal year. "
        "Total 12,345 6,789 4,321 3,210 2,100 1,000. "
        "Short one. "
        "The company paid Rs. 5 crore to its suppliers during the period."
    )
    kept = corpus.clean_sentences(text)
    assert kept == [
        "We expect revenue growth to continue in the next fiscal year.",
        "The company paid Rs. 5 crore to its suppliers during the period.",
    ]


def test_cue_words_are_found():
    assert corpus.CUE.search("We assume that prices will fall.")
    assert corpus.CUE.search("Management EXPECTS growth.")
    assert not corpus.CUE.search("The office is located in Ohio.")


def test_duplicate_sentences_inside_a_filing_are_removed_and_ids_are_stable():
    data = {"section_7": CUE_SENTENCE + " " + CUE_SENTENCE + " " + OTHER_SENTENCE}
    rows = corpus.filing_sentences(data, "abc")
    assert [r["sentence_id"] for r in rows] == ["abc-00001", "abc-00002"]
    assert rows[0]["has_cue"] and not rows[1]["has_cue"]


# ---------- Sampling ----------
def test_candidates_are_balanced_capped_and_reproducible(tmp_path):
    zip_path = make_zip(tmp_path / "2019.zip")
    first, stats = corpus.build_candidates(zip_path, n_filings=6, n_cue=8, n_random=8, max_per_doc=2, seed=7)
    again, _ = corpus.build_candidates(zip_path, n_filings=6, n_cue=8, n_random=8, max_per_doc=2, seed=7)

    assert len(first) == 16
    assert int(first["has_cue"].sum()) == 8
    assert first["sentence_id"].is_unique
    assert first.groupby(["doc_id", "has_cue"]).size().max() <= 2
    assert first.equals(again)
    assert stats["filings_read"] == 6


def test_broken_filings_are_skipped_and_counted(tmp_path):
    zip_path = make_zip(tmp_path / "2019.zip", n=3, extra_bad=True)
    table, stats = corpus.build_candidates(zip_path, n_filings=10, n_cue=3, n_random=3, max_per_doc=2, seed=1)
    assert stats["filings_skipped"] == 1
    assert "broken" not in " ".join(table["doc_id"])


def test_not_enough_sentences_gives_a_clear_error(tmp_path):
    zip_path = make_zip(tmp_path / "2019.zip", n=2)
    with pytest.raises(corpus.CorpusError) as err:
        corpus.build_candidates(zip_path, n_filings=2, n_cue=500, n_random=5, max_per_doc=2)
    assert "Not enough sentences" in str(err.value)


def test_bad_zip_and_existing_output_are_handled(tmp_path, capsys):
    bad = tmp_path / "bad.zip"
    bad.write_text("not a zip")
    with pytest.raises(corpus.CorpusError):
        corpus.build_candidates(bad)

    good = make_zip(tmp_path / "2019.zip")
    out = tmp_path / "cand.csv"
    args = ["--zip", str(good), "--out", str(out), "--n-filings", "6", "--n-cue", "6",
            "--n-random", "6", "--max-per-doc", "2"]
    assert corpus.main(args) == 0
    assert out.exists()
    assert corpus.main(args) == 1                     # refuses to overwrite your labeling base
    assert "already exists" in capsys.readouterr().err
    assert corpus.main(args + ["--force"]) == 0


def test_excluded_documents_and_sentences_are_left_out(tmp_path):
    zip_path = make_zip(tmp_path / "2019.zip", n=6)
    everything, _ = corpus.build_candidates(zip_path, n_filings=6, n_cue=8, n_random=8, max_per_doc=2, seed=3)
    banned_doc = everything["doc_id"].iloc[0]
    banned_ids = set(everything["sentence_id"].iloc[1:4])

    table, _ = corpus.build_candidates(zip_path, n_filings=6, n_cue=8, n_random=8, max_per_doc=2, seed=3,
                                       exclude_docs={banned_doc}, exclude_ids=banned_ids)
    assert banned_doc not in set(table["doc_id"])
    assert not (banned_ids & set(table["sentence_id"]))


def test_missing_exclude_file_gives_a_clear_error(tmp_path, capsys):
    good = make_zip(tmp_path / "2019.zip")
    code = corpus.main(["--zip", str(good), "--out", str(tmp_path / "c.csv"), "--exclude-docs", str(tmp_path / "nope.txt")])
    assert code == 1 and "exclude file" in capsys.readouterr().err
