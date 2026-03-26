Live-Mouse-Tracker modular pipeline

This repo now contains the cleaned module-based version of the LMT analysis code, rather than the older demo-output repository layout.

Layout

```text
code/
  main.py                  # entrypoint for one LMT sqlite recording
  cli.py                   # CLI arguments
  load_tables.py           # SQLite / DuckDB / Polars loading
  speed.py                 # trajectory + speed features
  thresholds.py            # automatic threshold estimation
  social.py                # social metrics + social index
  spatial.py               # spatial metrics + spatial index
  dominance.py            # pairwise wins + dominance index
  dominance_pl_stream.py   # time-resolved dominance ranking
  plots.py                 # publication-style plotting helpers
  theme.py                 # shared plotting theme
  metrics_utils.py         # scaling helpers
  stats.py                 # T-maze correlation analysis
```

What is intentionally not included

- EcoHab comparison dashboards and related outputs
- generated figures and tables
- large raw datasets and cohort folders
- the older monolithic `LMT.py` version

Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the modular LMT pipeline

```bash
python code/main.py path/to/Recording.sqlite out/Recording
```

Run the T-maze correlation analysis

```bash
python code/stats.py --base outputs --tmaze data/tmaze_summary.csv --out outputs/stats
```

Notes

- `code/main.py` is the modular replacement for the older spaghetti-style single-file pipeline.
- The repo is now source-first. Outputs should be generated locally and kept out of version control unless there is a specific reason to publish them.
