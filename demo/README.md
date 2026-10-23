# Assumption Monitor — Interactive Web & CLI Demo (`demo/`)

An end-to-end interactive web application and CLI demo showcasing the full **Assumption Monitor** pipeline (`reader.py` $\to$ `autolabel.py` & `model_a.py` $\to$ `sources.py` & `series.py` point-in-time indicator monitoring).

---

## Quick Start

### 1. Start the Web UI Server
From the project root:
```bash
python3 demo/app.py --port 8080
# or:
./demo/run_demo.sh --port 8080
```
Then open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser.

### 2. Run the Terminal Summary Report
```bash
python3 demo/app.py --cli
```

---

## What's Inside the UI

1. **Tab 1 — Plan Monitor & Live Alerts**:
   - Switch between **NorthStar Freight (`plan1.pdf`)**, **GreenCart Retail (`plan2.docx`)**, **Upload any `.pdf` or `.docx`**, or **Custom Business Plan Text**.
   - **Point-in-Time (`available_at <= as_of`) Time-Travel**: Jump between historical dates (e.g., *Jul 10, 2026* when WTI Crude was `$70.48/bbl` and assumptions held, *Aug 19, 2026* when oil drifted to `$84.05/bbl`, *Sep 23, 2026* at the `$103.54/bbl` spike, and *Oct 16, 2026* at `$96.88/bbl` & `96.60 INR/USD`).
   - **What-If Macro Shock Simulator**: Drag sliders for WTI Crude Oil (`$/bbl`) and `INR/USD` exchange rate or adjust the Model A probability threshold and detection mode (`Hybrid`, `Model A Only`, `Rules v1 Only`).
   - **Sentence Deep-Dive & Reader Audit Log**: Filter extracted sentences, click any sentence to inspect its TF-IDF feature weights and regex signals, and view removed headers/footers/table fragments from `reader.py`.
2. **Tab 2 — Sentence X-Ray (`model_a.py` + `autolabel.py`)**:
   - Real-time single-sentence sandbox showing exact per-feature TF-IDF $\times$ LogisticRegression log-odds contributions alongside `autolabel.classify` regex rule triggers (`EXPLICIT`, `EXPECT`, `FUTURE`, `HEDGE`, `BOILERPLATE`, `ASSUMED_THING`, `VALUATION`, `OWN_RESULT`, `OWN_PLAN`).
3. **Tab 3 — Indicator Series & Registry (`sources.py` + `series.py`)**:
   - Interactive SVG charts for `DCOILWTICO` and `DEXINUS` with overlaid plan assumption bands, causal rolling 26-week MAD robust-z outlier markers (`|robust_z| > 4.0`), forward-filled week counts, and live SHA-256 checksum verification (`sources.verify_log()`).
4. **Tab 4 — Model A & Labeling Lab (`model_a.py` + `labeler.py` + `scripts/fit_load_model.py`)**:
   - Document-separated 5-fold `GroupKFold` cross-validation comparison (`rules only` vs. `model, your labels` vs. `model, your labels + rule labels`), dataset split breakdown, top learned TF-IDF words, human audit 95% Wilson confidence interval, CV error browser, and PC scaling benchmark results.
