"""Clean daily indicator series into weekly observations (implementation plan, section 5)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from assumption_monitor import sources

# ---------- Settings (written down on purpose: they belong in the decision log) ----------
WEEK_RULE = "W-FRI"      # a week runs Saturday to Friday and is labelled by its Friday
MAX_FILL = 2             # fill at most 2 consecutive missing weeks
OUTLIER_WINDOW = 26      # look back at most 26 weeks
MIN_HISTORY = 10         # need at least 10 earlier real weeks before testing a value
Z_LIMIT = 4.0            # flag when |robust z| is above this
MAD_SCALE = 1.4826       # makes the MAD comparable to a standard deviation

OUTPUT_COLUMNS = [
    "series_id", "observation_date", "available_at", "value", "n_daily_obs",
    "is_missing", "is_filled", "is_outlier", "robust_z",
]


class SeriesError(Exception):
    """A problem with a data series."""


# ---------- Step 1: check the daily data ----------
def check_daily(daily):
    """Sort the series and handle repeated dates. Conflicting repeats must be investigated."""
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise SeriesError("The series needs a date index.")
    daily = daily.sort_index()
    repeated = daily.index.duplicated(keep=False)
    if repeated.any():
        groups = daily[repeated].groupby(level=0)
        conflicting = [d for d, g in groups if g.dropna().nunique() > 1]
        if conflicting:
            raise SeriesError(f"Same date with different values (investigate): {conflicting[:3]}")
        daily = daily[~daily.index.duplicated(keep="first")]
    return daily


# ---------- Step 2: daily to weekly ----------
def to_weekly(daily):
    """Weekly mean of the real daily values. The last week is dropped if it is not finished."""
    weekly = pd.DataFrame({
        "value": daily.resample(WEEK_RULE).mean(),
        "n_daily_obs": daily.resample(WEEK_RULE).count(),
    })
    weekly = weekly[weekly.index <= daily.index.max()]   # keep only weeks that have reached their Friday
    return weekly


# ---------- Step 3: outliers (flag, never delete) ----------
def robust_z_scores(values, window=OUTLIER_WINDOW, min_history=MIN_HISTORY):
    """Compare each value with the median of the EARLIER real values only."""
    arr = values.to_numpy(dtype=float)
    z = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        past = arr[max(0, i - window):i]
        past = past[~np.isnan(past)]
        if len(past) < min_history:
            continue                                   # not enough history: do not judge
        median = np.median(past)
        mad = np.median(np.abs(past - median))
        deviation = arr[i] - median
        if mad == 0:                                   # perfectly flat history
            z[i] = 0.0 if deviation == 0 else np.sign(deviation) * np.inf
        else:
            z[i] = deviation / (MAD_SCALE * mad)
    return pd.Series(z, index=values.index)


# ---------- Step 4: fill short gaps ----------
def fill_short_gaps(values, max_fill=MAX_FILL):
    """Carry the last real value into gaps of at most `max_fill` weeks.

    Longer gaps, gaps at the start (no past value) and gaps at the end (not published yet) stay empty.
    """
    filled = values.copy()
    is_filled = pd.Series(False, index=values.index)
    missing = values.isna().to_numpy()
    i, n = 0, len(values)
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_fill:
            filled.iloc[i:j] = values.iloc[i - 1]
            is_filled.iloc[i:j] = True
        i = j
    return filled, is_filled


# ---------- Step 5: when could this value be known? ----------
def available_at(date, update_weekday, min_lag_days):
    """First update day (a fixed weekday) that is at least `min_lag_days` after the observation date."""
    earliest = pd.Timestamp(date) + pd.Timedelta(days=int(min_lag_days))
    days_ahead = (int(update_weekday) - earliest.weekday()) % 7
    return (earliest + pd.Timedelta(days=days_ahead)).normalize()


# ---------- Put it together ----------
def clean_weekly(daily, series_id, update_weekday, min_lag_days):
    daily = check_daily(daily)
    weekly = to_weekly(daily)

    z = robust_z_scores(weekly["value"])                  # judged on real weeks only
    filled_values, is_filled = fill_short_gaps(weekly["value"])

    out = pd.DataFrame({
        "series_id": series_id,
        "observation_date": weekly.index,
        "available_at": [available_at(d, update_weekday, min_lag_days) for d in weekly.index],
        "value": filled_values.to_numpy(),
        "n_daily_obs": weekly["n_daily_obs"].to_numpy(),
        "is_missing": filled_values.isna().to_numpy(),
        "is_filled": is_filled.to_numpy(),
        "is_outlier": (z.abs() > Z_LIMIT).to_numpy(),
        "robust_z": z.to_numpy(),
    })
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def clean_series(series_id, log_path=sources.LOG_PATH, registry=None):
    registry = sources.load_registry() if registry is None else registry
    entry = sources.get_entry(registry, series_id)
    for field in ("update_weekday", "min_lag_days"):
        if field not in entry or pd.isna(entry[field]):
            raise SeriesError(f"{series_id}: add '{field}' to the source registry.")
    daily = sources.load_series(series_id, log_path)
    return clean_weekly(daily, series_id, entry["update_weekday"], entry["min_lag_days"])


def summary(table):
    return {
        "weeks": len(table),
        "first_week": str(table["observation_date"].min().date()),
        "last_week": str(table["observation_date"].max().date()),
        "missing_weeks": int(table["is_missing"].sum()),
        "filled_weeks": int(table["is_filled"].sum()),
        "outlier_weeks": int(table["is_outlier"].sum()),
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Clean stored daily series into weekly tables.")
    parser.add_argument("series", nargs="+")
    parser.add_argument("--out-dir", default="data/clean")
    args = parser.parse_args(argv)
    try:
        for sid in args.series:
            table = clean_series(sid)
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            table.to_csv(out / f"weekly_{sid}.csv", index=False)
            print(sid, summary(table))
    except (SeriesError, sources.SourceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
