Live-Mouse-Tracker (LMT) pipeline + T-maze stats

This repo contains two complementary pieces:

LMT.py — computes Social, Spatial, and Dominance indices from a single LMT *.sqlite recording, and exports harmonized CSVs + publication-ready figures.

stats.py — correlates those LMT indices with T-maze performance (learning slope and mean accuracy) using Spearman’s ρ with permutation p-values, plus per-group meta-analysis and forest plots.

Both scripts share a unified visual theme for consistent figures.

1) LMT pipeline (LMT.py)
What it does

Social index (0–1): weighted combnationo of fast approaches/min, fast escapes/min, and % time in contact.

Spatial index (0–1): locomotion, arena use, and exploration strategy features.

Dominance index (0–1): pairwise wins from approach/escape events, normalized by exposure, ranked via Plackett–Luce/Bradley–Terry (0–1 scaling).

Inputs

One LMT recording.sqlite.

Key outputs (in the recording’s folder unless out_dir is provided)

social_metrics.csv, spatial_metrics.csv

social_index.csv, spatial_index.csv, dominance_index.csv

social_spatial_dom_index.csv (all three indices)

A set of themed PNGs (bars, heatmaps, component panels)

How to start:

# (optional) conda env
conda create -n lmt_social python=3.11 -y
conda activate lmt_social
pip install -r requirements-lmt.txt

# run
python LMT.py  path/to/Recording.sqlite  out/Recording --species mouse
# or e.g.
python LMT.py  path/to/Recording.sqlite  out/Recording --species rat


Useful flags

    --species {mouse,rat} (sets defaults for distances/speeds)

    --social-scale {minmax,robust,zscore} (default: robust)

    --weights-json weights.json (override component weights)

    Dominance tuning: --dominance {davids,plackettluce} (default: plackettluce), --dom-transform {None,log1p,rate}, --dom-cap <float>

    Speed/contact geometry: --move, --burst, --contact-mm, --iso-mm, --fast-thr, --margin-mm, --corner-mm

    --debug (limit frames for a fast smoke test)

    --no-heatmap (skip W heatmap)

2) T-maze correlates (stats.py)
What it does: 

Loads LMT outputs (indices) across one or multiple groups.

Loads T-maze summary (tmaze_summary.csv with columns: rfid, day, accuracy). This is based on my own experiment in T-maze, but it's a flexible framework and I guess it will work with other behavioural metrics.

Builds per-mouse learning slope (accuracy ~ day) and mean accuracy.

Computes pooled Spearman ρ (with permutation p) and saves:

corr_rho.csv, corr_p_perm.csv, corr_q_perm_bh.csv

corr_heatmap.png (plain), corr_heatmap_poster.png (poster-friendly)

scatter_grid.png

Performs per-group exact Spearman + meta-analysis (Fisher-z pooling, Stouffer p) and saves:

meta_spearman_meta.csv + forest plots per pair.

Expected layout;

repo/
├─ LMT.py
├─ stats.py
├─ data/
│  └─ tmaze_summary.csv        # columns: rfid, day, accuracy
├─ outputs/
│  └─ <GROUP_A>/
│      ├─ social_spatial_index.csv  OR  social_spatial_dom_index.csv
│      └─ dominance_index.csv
│  └─ <GROUP_B>/
│      ├─ ...
│      └─ ...
└─ requirements-*.txt


How to start:
pip install -r requirements-stats.txt

# Run across ALL groups under --base
python stats.py \
  --base outputs \
  --tmaze data/tmaze_summary.csv \
  --out outputs/stats_all

# Run for ONE specific group (e.g., C1G1)
python stats.py \
  --base outputs \
  --tmaze data/tmaze_summary.csv \
  --group C1G1 \
  --out outputs/stats_C1G1

Outputs (in --out)

corr_heatmap.png, corr_heatmap_poster.png, scatter_grid.png

corr_rho.csv, corr_p_perm.csv, corr_q_perm_bh.csv

meta_spearman_meta.csv, meta_forest_<Y>_vs_<X>.png (several)
