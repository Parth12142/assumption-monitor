from pathlib import Path

import pandas as pd
import pytest

from assumption_monitor import sources

REGISTRY_FILE = Path(__file__).resolve().parent.parent / "config" / "source_registry.json"

CSV_TEXT = (
    "observation_date,DCOILWTICO\n"
    "2026-01-05,60.10\n"
    "2026-01-06,.\n"
    "2026-01-07,61.50\n"
    "2026-01-07,61.50\n"
    "2026-01-08,abc\n"
)


def make_csv(path, text=CSV_TEXT):
    path.write_text(text)
    return path


def good_registry():
    return sources.load_registry(REGISTRY_FILE)


# ---------- Registry ----------
def test_real_registry_file_is_valid_and_complete():
    registry = good_registry()
    assert {"DCOILWTICO", "DEXINUS"} <= set(registry["series_id"])
    row = sources.get_entry(registry, "DEXINUS")
    assert row["units"] == "Indian Rupees to One U.S. Dollar"
    assert row["frequency"].startswith("Daily")


def test_registry_validation_finds_problems():
    registry = good_registry()
    broken = registry.copy()
    broken.loc[0, "units"] = ""
    assert any("empty units" in p for p in sources.validate_registry(broken))

    duplicated = pd.concat([registry, registry.iloc[[0]]], ignore_index=True)
    assert "duplicate series_id" in sources.validate_registry(duplicated)

    approved = registry.copy()
    approved.loc[0, "approved"] = "yes"
    approved.loc[0, "terms_checked_on"] = ""
    assert any("terms_checked_on" in p for p in sources.validate_registry(approved))


def test_unknown_series_is_reported():
    with pytest.raises(sources.SourceError):
        sources.get_entry(good_registry(), "NOPE")


# ---------- URLs and approved hosts ----------
def test_fred_url_and_host_checks():
    assert sources.fred_csv_url("DEXINUS") == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS"
    with pytest.raises(sources.SourceError):
        sources.fred_csv_url("../etc/passwd")
    sources.check_allowed("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS")
    with pytest.raises(sources.SourceError):
        sources.check_allowed("https://example.com/data.csv")
    with pytest.raises(sources.SourceError):
        sources.check_allowed("http://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS")


# ---------- Checksums ----------
def test_checksum_is_stable_and_detects_changes(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("hello")
    first = sources.sha256_file(f)
    assert first == sources.sha256_file(f)
    f.write_text("hello!")
    assert sources.sha256_file(f) != first


# ---------- Profiling ----------
def test_profile_counts_missing_duplicates_and_bad_values(tmp_path):
    prof = sources.profile_series(make_csv(tmp_path / "s.csv"))
    assert prof["n_rows"] == 5
    assert prof["n_missing"] == 1
    assert prof["n_bad_values"] == 1
    assert prof["n_duplicate_dates"] == 1
    assert prof["first_date"] == "2026-01-05"
    assert prof["last_date"] == "2026-01-08"
    assert prof["is_sorted"] is True
    assert prof["max_value"] == 61.5


# ---------- Storing files ----------
def test_register_file_copies_logs_and_verifies(tmp_path):
    original = make_csv(tmp_path / "manual.csv")
    raw_dir, log = tmp_path / "raw", tmp_path / "log.csv"
    record = sources.register_file("DCOILWTICO", original, good_registry(), raw_dir, log)

    stored = Path(record["file"])
    assert stored.exists() and stored != original
    assert record["sha256"] == sources.sha256_file(original)
    assert sources.verify_log(log) == []

    stored.write_text("tampered")
    problems = sources.verify_log(log)
    assert any("checksum changed" in p for p in problems)


def test_load_series_turns_missing_markers_into_nan(tmp_path):
    original = make_csv(tmp_path / "manual.csv", "observation_date,DCOILWTICO\n2026-01-06,.\n2026-01-05,60.1\n")
    log = tmp_path / "log.csv"
    sources.register_file("DCOILWTICO", original, good_registry(), tmp_path / "raw", log)
    s = sources.load_series("DCOILWTICO", log)
    assert s.index.is_monotonic_increasing
    assert s.iloc[0] == 60.1
    assert pd.isna(s.iloc[1])


# ---------- Downloading (network is faked) ----------
def test_download_refuses_unapproved_series(tmp_path):
    registry = good_registry().copy()

    registry.loc[registry["series_id"] == "DEXINUS", "approved"] = "no"

    with pytest.raises(sources.SourceError) as err:
        sources.download_fred(
            "DEXINUS",
            registry,
            tmp_path / "raw",
            tmp_path / "log.csv"
        )

    assert "not approved" in str(err.value)


def approved_registry():
    registry = good_registry().copy()
    registry["approved"] = "yes"
    registry["terms_checked_on"] = "2026-10-16"
    return registry


def test_download_saves_file_and_logs_it(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_fetch", lambda url, timeout, retries, pause: CSV_TEXT.encode())
    record = sources.download_fred("DCOILWTICO", approved_registry(), tmp_path / "raw", tmp_path / "log.csv")
    assert Path(record["file"]).exists()
    assert record["n_rows"] == 5
    assert sources.verify_log(tmp_path / "log.csv") == []


def test_download_rejects_a_web_page(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_fetch", lambda url, timeout, retries, pause: b"<!DOCTYPE html><html></html>")
    with pytest.raises(sources.SourceError) as err:
        sources.download_fred("DCOILWTICO", approved_registry(), tmp_path / "raw", tmp_path / "log.csv")
    assert "web page" in str(err.value)
    assert not (tmp_path / "log.csv").exists()


def test_command_line_reports_errors(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert sources.main(["list"]) == 1
    assert "ERROR" in capsys.readouterr().err
