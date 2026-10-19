import numpy as np
import pandas as pd
import pytest

from assumption_monitor import series, sources


def daily(values, start="2026-01-05"):
    """Business-day series starting on a Monday."""
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, dtype=float)


def weekly_values(values):
    """A plain weekly Series (one value per week), used to test the gap and outlier rules."""
    return pd.Series(values, index=pd.date_range("2026-01-02", periods=len(values), freq="W-FRI"), dtype=float)


# ---------- Week boundaries ----------
def test_weeks_end_on_friday_and_use_the_mean():
    d = daily(list(range(1, 16)))                 # 3 full weeks: 1-5, 6-10, 11-15
    weekly = series.to_weekly(d)
    assert [x.weekday() for x in weekly.index] == [4, 4, 4]
    assert weekly["value"].tolist() == [3.0, 8.0, 13.0]
    assert weekly["n_daily_obs"].tolist() == [5, 5, 5]


def test_unfinished_last_week_is_dropped():
    d = daily(list(range(1, 18)))                 # 3 full weeks plus Monday and Tuesday
    weekly = series.to_weekly(d)
    assert len(weekly) == 3


def test_holiday_gaps_are_ignored_in_the_mean_and_empty_weeks_stay_empty():
    values = [10, 10, np.nan, 10, 10] + [np.nan] * 5 + [20, 20, 20, 20, 20]
    weekly = series.to_weekly(daily(values))
    assert weekly["value"].iloc[0] == 10 and weekly["n_daily_obs"].iloc[0] == 4
    assert np.isnan(weekly["value"].iloc[1]) and weekly["n_daily_obs"].iloc[1] == 0


# ---------- Duplicate keys ----------
def test_identical_duplicates_are_dropped_and_conflicts_are_reported():
    d = daily([1, 2, 3])
    same = pd.concat([d, d.iloc[[1]]])
    assert len(series.check_daily(same)) == 3

    conflict = pd.concat([d, pd.Series([99.0], index=[d.index[1]])])
    with pytest.raises(series.SeriesError):
        series.check_daily(conflict)


# ---------- Filling gaps ----------
def test_gaps_of_two_weeks_are_filled_and_marked():
    v = weekly_values([1, 2, np.nan, np.nan, 5])
    filled, flag = series.fill_short_gaps(v)
    assert filled.tolist() == [1, 2, 2, 2, 5]
    assert flag.tolist() == [False, False, True, True, False]


def test_gaps_longer_than_two_weeks_leave_everything_empty():
    v = weekly_values([1, 2, np.nan, np.nan, np.nan, 6])
    filled, flag = series.fill_short_gaps(v)
    assert filled.isna().tolist() == [False, False, True, True, True, False]
    assert not flag.any()


def test_leading_and_trailing_gaps_are_not_filled():
    v = weekly_values([np.nan, 2, 3, np.nan])
    filled, flag = series.fill_short_gaps(v)
    assert np.isnan(filled.iloc[0]) and np.isnan(filled.iloc[3])
    assert not flag.any()


# ---------- Outliers ----------
def test_a_spike_is_flagged_but_its_value_is_kept():
    rng = np.random.default_rng(0)
    base = list(100 + rng.normal(0, 1, 30))
    base[20] = 160
    table = series.clean_weekly(daily(np.repeat(base, 5)), "X", 2, 2)
    row = table.iloc[20]
    assert row["is_outlier"]
    assert row["value"] == 160
    assert table["is_outlier"].sum() < 3


def test_early_weeks_are_not_judged_for_lack_of_history():
    v = weekly_values([100, 100, 100, 500, 100, 100])
    z = series.robust_z_scores(v)
    assert z.isna().all()


def test_flat_history_flags_any_jump_and_not_the_flat_value():
    v = weekly_values([50.0] * 12 + [50.0, 55.0])
    z = series.robust_z_scores(v)
    assert z.iloc[12] == 0
    assert np.isinf(z.iloc[13]) and abs(z.iloc[13]) > series.Z_LIMIT


def test_scores_use_only_the_past():
    rng = np.random.default_rng(1)
    base = 100 + rng.normal(0, 1, 40)
    changed = base.copy()
    changed[35] = 500                           # change a later value
    z1 = series.robust_z_scores(weekly_values(base))
    z2 = series.robust_z_scores(weekly_values(changed))
    assert z1.iloc[:35].equals(z2.iloc[:35])


def test_filled_weeks_are_never_flagged_as_outliers():
    base = [100.0] * 15 + [np.nan, np.nan] + [100.0] * 5
    table = series.clean_weekly(daily(np.repeat(base, 5)), "X", 2, 2)
    filled = table[table["is_filled"]]
    assert len(filled) == 2 and not filled["is_outlier"].any()


# ---------- When is a value available? ----------
def test_oil_values_become_available_on_wednesdays():
    monday, tuesday = pd.Timestamp("2026-10-12"), pd.Timestamp("2026-10-13")
    assert series.available_at(monday, 2, 2) == pd.Timestamp("2026-10-14")
    assert series.available_at(tuesday, 2, 2) == pd.Timestamp("2026-10-21")


def test_exchange_rate_values_become_available_on_mondays():
    friday = pd.Timestamp("2026-10-09")
    assert series.available_at(friday, 0, 3) == pd.Timestamp("2026-10-12")


def test_available_at_is_always_after_the_observation_and_on_the_update_day():
    table = series.clean_weekly(daily(np.arange(100.0) + 50), "X", 2, 2)
    assert (table["available_at"] > table["observation_date"]).all()
    assert (table["available_at"].dt.weekday == 2).all()


# ---------- Whole pipeline ----------
def test_registry_has_publication_rules_for_both_series():
    registry = sources.load_registry()
    for sid in ["DCOILWTICO", "DEXINUS"]:
        entry = sources.get_entry(registry, sid)
        assert 0 <= int(entry["update_weekday"]) <= 6
        assert int(entry["min_lag_days"]) >= 0


def test_clean_series_end_to_end(tmp_path):
    idx = pd.bdate_range("2025-01-06", periods=120)
    lines = ["observation_date,DCOILWTICO"]
    for i, d in enumerate(idx):
        value = "." if i % 37 == 0 else f"{70 + (i % 7) * 0.5:.2f}"
        lines.append(f"{d.date()},{value}")
    raw = tmp_path / "oil.csv"
    raw.write_text("\n".join(lines))

    log = tmp_path / "log.csv"
    registry = sources.load_registry()
    sources.register_file("DCOILWTICO", raw, registry, tmp_path / "raw", log)
    table = series.clean_series("DCOILWTICO", log, registry)

    assert list(table.columns) == series.OUTPUT_COLUMNS
    assert table["observation_date"].is_monotonic_increasing
    assert not table["observation_date"].duplicated().any()
    assert table["series_id"].eq("DCOILWTICO").all()


def test_command_line_reports_missing_data(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert series.main(["DCOILWTICO"]) == 1
    assert "ERROR" in capsys.readouterr().err
