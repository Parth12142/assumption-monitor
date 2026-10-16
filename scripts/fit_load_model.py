"""Fit regression models to the benchmark results and predict whether this PC can take bigger loads."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class PowerLaw(RegressorMixin, BaseEstimator):
    """y = a * x^b, fitted as a straight line on a log-log scale."""

    def fit(self, X, y):
        self.line = LinearRegression().fit(np.log(X), np.log(y))
        self.exponent = float(self.line.coef_[0])
        return self

    def predict(self, X):
        return np.exp(self.line.predict(np.log(X)))


def make_models():
    models = {}
    for degree in (1, 2, 3, 4):
        models[f"poly{degree}"] = make_pipeline(
            PolynomialFeatures(degree, include_bias=False), StandardScaler(), LinearRegression()
        )
    models["power_law"] = PowerLaw()
    return models


def mape(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true) / y_true) * 100)


def compare_models(x, y, test_share=0.3):
    """Two honest checks: leave-one-out (inside the range) and train-small/test-big (beyond the range)."""
    order = np.argsort(x[:, 0])
    x, y = x[order], y[order]
    n_test = max(2, int(round(len(x) * test_share)))
    x_train, y_train, x_test, y_test = x[:-n_test], y[:-n_test], x[-n_test:], y[-n_test:]

    rows = []
    for name, model in make_models().items():
        loo = cross_val_predict(model, x, y, cv=LeaveOneOut())
        fitted = make_models()[name].fit(x_train, y_train)
        rows.append({
            "model": name,
            "inside_range_error_%": round(mape(y, loo), 1),
            "beyond_range_error_%": round(mape(y_test, fitted.predict(x_test)), 1),
        })
    return pd.DataFrame(rows)


def choose_best(table, tolerance=5.0):
    """Lowest beyond-range error wins; if a simpler model is within `tolerance` points, prefer it."""
    best_error = table["beyond_range_error_%"].min()
    close = table[table["beyond_range_error_%"] <= best_error + tolerance]
    simplicity = {"poly1": 1, "power_law": 2, "poly2": 3, "poly3": 4, "poly4": 5}
    return close.sort_values("model", key=lambda s: s.map(simplicity)).iloc[0]["model"]


def verdict(minutes, mem_mb, n, n_max_measured, max_minutes, ram_mb):
    problems = []
    if minutes > max_minutes:
        problems.append("TOO SLOW")
    if mem_mb > ram_mb:
        problems.append("NOT ENOUGH RAM")
    text = " + ".join(problems) if problems else "OK"
    if n > 2 * n_max_measured:
        text += " (uncertain: far beyond measured sizes)"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--targets", type=int, nargs="+", default=[100000, 500000, 1000000, 2000000, 5000000])
    parser.add_argument("--max-minutes", type=float, default=10, help="longest run you would accept")
    parser.add_argument("--ram-mb", type=float, help="memory you allow the job to use (default: 80%% of free RAM)")
    args = parser.parse_args()

    results = Path(args.results_dir)
    data = pd.read_csv(results / "benchmark.csv")
    info = json.loads((results / "pc_info.json").read_text())
    ram_limit = args.ram_mb or 0.8 * info["ram_available_mb"]

    grouped = data.groupby("n", as_index=False).mean(numeric_only=True)   # average repeats
    x = (grouped["n"].to_numpy() / 1000).reshape(-1, 1)                   # thousands of sentences
    n_max = int(grouped["n"].max())
    print(f"This computer: {info['cpu_count']} CPU(s), {info['ram_total_mb']} MB RAM "
          f"({info['ram_available_mb']} MB free when measured)")
    print(f"Measured sizes: {int(grouped['n'].min()):,} to {n_max:,} sentences")
    print(f"Limits used: {args.max_minutes:g} minutes, {ram_limit:,.0f} MB RAM\n")

    fitted, plot_data = {}, {}
    for label, column, scale in [("time", "seconds", 1 / 60), ("memory", "peak_mb", 1.0)]:
        y = grouped[column].to_numpy() * scale          # time in minutes, memory in MB
        table = compare_models(x, y)
        best = choose_best(table)
        print(f"=== {label}: model comparison (lower error is better) ===")
        print(table.to_string(index=False))
        print(f"Best choice for {label}: {best}")
        model = make_models()[best].fit(x, y)
        if best == "power_law":
            print(f"   time/memory grows like n^{model.exponent:.2f} (1.0 means straight-line growth)")
        print()
        fitted[label] = (best, model)
        plot_data[label] = (y, make_models(), table)

    print("=== Can this PC take the load? ===")
    rows = []
    for n in args.targets:
        xt = np.array([[n / 1000]])
        minutes = float(fitted["time"][1].predict(xt)[0])
        mem = float(fitted["memory"][1].predict(xt)[0])
        rows.append({
            "sentences": f"{n:,}",
            "predicted_minutes": round(max(minutes, 0), 1),
            "predicted_memory_mb": round(max(mem, 0)),
            "verdict": verdict(minutes, mem, n, n_max, args.max_minutes, ram_limit),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPredictions beyond the measured sizes are estimates. Test the largest size you plan to use.")

    # ----- plot -----
    x_max = min(max(args.targets), 5 * n_max) / 1000
    grid = np.linspace(x.min(), x_max, 200).reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (label, unit, limit) in zip(axes, [("time", "minutes", args.max_minutes), ("memory", "MB", ram_limit)]):
        y, models, _ = plot_data[label]
        best_name = fitted[label][0]
        ax.scatter(x[:, 0], y, color="black", zorder=3, label="measured")
        for name, model in models.items():
            model.fit(x, y)
            ax.plot(grid[:, 0], model.predict(grid), "-" if name == best_name else ":",
                    lw=2 if name == best_name else 1, label=f"{name}" + (" (best)" if name == best_name else ""))
        ax.axhline(limit, color="red", ls="--", label="your limit")
        ax.set_xlabel("sentences (thousands)")
        ax.set_ylabel(unit)
        ax.set_title(f"{label.capitalize()} vs load")
        ax.set_ylim(0, max(limit * 1.3, y.max() * 1.5))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "load_model.png", dpi=130)
    print(f"\nPlot saved to {results / 'load_model.png'}")


if __name__ == "__main__":
    main()
