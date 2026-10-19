# Stuck log
- Day 1: pip install timed out on matplotlib. Fix: --timeout 120 --resume-retries 5
- Day 4: reader module with Word support, audit log and 22 passing tests. Problem hit: missing __init__.py and pytest.ini, fixed by creating both.
- Day 4: moved cleaning into src/assumption_monitor/reader.py; added Word support, audit log, 16 tests. Problem hit today: [write one]
- Day 5: source registry, approved-host downloader, checksum log, 34 tests. Problem hit: [write one]
- Regression practice: benchmarked this PC and compared polynomial (degree 1-4) and power-law models to predict time and memory for bigger jobs. Lesson: high-degree polynomials overfit and extrapolate badly.
- Day 6: weekly cleaning (Friday weeks, fill <=2 weeks, robust-z outlier flags, available_at from publication rules), 52 tests. Problem hit: [write one]
