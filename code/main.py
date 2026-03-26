# main.py
from __future__ import annotations

import time
from pathlib import Path
import pandas as pd

from cli import parse_args as cli
from load_tables import load_tables_polars
from speed import build_traj_polars

from thresholds import (
    auto_thresholds_from_speed,
    auto_contact_iso_from_NN,
    infer_burst_run_length,
)

from social import (
    social_metrics,
    social_index,
    build_contact_table_parallel,
)

# fast Bradley–Terry/MM from pairwise W
from dominance import WinWeights, compute_pl_index

from spatial import spatial_metrics_polars, spatial_index
from theme import set_label_order

# OpenSkill stream (time-resolved)
from dominance_pl_stream import (
    events_from_interactions,
    rank_openskill_stream,
    resample_ord,
)

# plots
from plots import (
    plot_indices_combined,
    plot_scatter_social_vs_spatial,
    plot_bars_simple,
    plot_W_heatmap,
    plot_speed_distributions,
    plot_category_multipanel,
    # time-series dominance plots
    plot_dominance_ordinals,
    plot_dominance_rank_heatmap,
    plot_dominance_stability,
    plot_dominance_over_time,
    plot_dominance_normalized
)


def main():
    args = cli()
    t0 = time.time()

    db = Path(args.sqlite).expanduser()
    out = Path(args.out) if args.out else db.parent
    out.mkdir(parents=True, exist_ok=True)

    # ── Load tables once
    animals_lf, detection_lf, frames_lf = load_tables_polars(db, debug=args.debug)
    animals = animals_lf.collect().to_pandas().set_index("ID", drop=False)

    # ---- RFID tails + safe mapper (needed later for relabeling) ----
    tails = animals["RFID"].astype(str).str.strip().str[-4:]
    try:
        tails.index = tails.index.astype(int)
    except Exception:
        pass

    def map_id(x):
        if pd.isna(x):
            return x
        try:
            i = int(x)
            val = tails.get(i, None)
            return str(val) if val is not None else str(i)
        except Exception:
            return str(x)

    # ── First pass to get dt/speed; use harmless placeholders
    traj0 = build_traj_polars(
        detection_lf, frames_lf,
        move_thr=1.0, burst_thr=1.0, burst_runlen=2
    )

    # ── Learn thresholds from *this* dataset
    move_thr, burst_thr, fast_thr, _ = auto_thresholds_from_speed(traj0)
    contact_mm, iso_mm, _ = auto_contact_iso_from_NN(traj0, workers=args.workers)
    burst_runlen = infer_burst_run_length(traj0, min_burst_dur=0.08)

    # ── Second pass: rebuild trajectory with learned cutoffs
    traj = build_traj_polars(
        detection_lf, frames_lf,
        move_thr=move_thr,
        burst_thr=burst_thr,
        burst_runlen=burst_runlen,
    )

    animals_list = sorted(traj["ANIMALID"].dropna().astype(int).unique())

    # ---- Fixed label order for plots/exports ----
    animal_ids_all = sorted(traj["ANIMALID"].dropna().astype(int).unique())
    LABELS = pd.Index([map_id(i) for i in animal_ids_all], name="RFID")
    set_label_order(LABELS)

    def pad_to_all_labels(obj):
        if isinstance(obj, pd.Series):
            return obj.reindex(LABELS).fillna(0.0)
        if isinstance(obj, pd.DataFrame):
            return obj.reindex(LABELS).fillna(0.0)
        return obj

    # ── Learned config for downstream use (no arbitrary constants)
    cfg = {
        "move_thr": move_thr,
        "burst_thr": burst_thr,
        "fast_thr": fast_thr,
        "contact_mm": contact_mm,
        "iso_mm": iso_mm,
        "burst_runlen": burst_runlen,
    }

    print(
        "[auto] "
        f"move={move_thr:.2f}  burst={burst_thr:.2f}  fast={fast_thr:.2f}  "
        f"contact={contact_mm:.1f}  iso={iso_mm:.1f}  burst_runlen={burst_runlen}"
    )

    # ── Speed plots (show learned cutoffs)
    plot_speed_distributions(
        traj,
        out,
        logx=args.speed_logx,
        smooth_win=args.speed_smooth,
        workers=args.workers,
        stationary_cutoff_mm_s=cfg["move_thr"],
        fit_gmm=args.speed_fit_gmm,
        vlines_mm_s=[cfg["move_thr"], cfg["burst_thr"]],
        vline_labels=["move_thr", "burst_thr"],
    )

    # ── Contacts
    contacts = build_contact_table_parallel(traj, cfg["contact_mm"], args.workers)

    # exposure minutes (for optional 'rate' transform)
    det_minutes = (
        traj.loc[traj["ANIMALID"].notna()].groupby("ANIMALID")["dt"].sum().div(60.0)
    )

    # ── Social metrics
    soc_m, approaches, escapes, state = social_metrics(
        traj, contacts, cfg["fast_thr"], cfg["iso_mm"]
    )
    soc_idx = social_index(soc_m, scale=args.social_scale, weights=None)

    # ── Dominance from pairwise W (BT/MM baseline)
    W_raw, W_capped, dom_idx_bt = compute_pl_index(
        approaches,
        escapes,
        weights=WinWeights(args.approach_weight, args.escape_weight),
        transform=args.dom_transform,  # None | "log1p" | "rate"
        exposure_minutes=(det_minutes if args.dom_transform == "rate" else None),
        cap=args.dom_cap,
    )

    # ── Attach datetime to approaches/escapes (merge by FRAMENUMBER)
    frames_pd = frames_lf.select(["FRAMENUMBER", "t"]).collect().to_pandas()
    frames_pd["t"] = pd.to_datetime(frames_pd["t"], utc=False, errors="coerce")

    def _attach_time(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        if "t" not in out.columns:
            if "FRAMENUMBER" not in out.columns:
                return out  # no way to merge
            out = out.merge(frames_pd, on="FRAMENUMBER", how="left")
        out["t"] = pd.to_datetime(out["t"], utc=False, errors="coerce")
        out["datetime"] = out["t"]
        return out

    approaches = _attach_time(approaches)
    escapes = _attach_time(escapes)

    # list of actually present animals (ints)
    animals_list = sorted(traj["ANIMALID"].dropna().astype(int).unique())

    # Build weighted events for PL stream from approaches + escapes.
    # Weights mirror BT/MM settings so both dominance paths stay comparable.
    def _apply_event_weight(ev: pd.DataFrame, w: float) -> pd.DataFrame:
        if ev is None or ev.empty:
            return pd.DataFrame(columns=["loser", "winner", "datetime"])
        w = float(w)
        if w == 0:
            return ev.iloc[0:0].copy()
        e = ev.copy()
        if w < 0:
            e = e.rename(columns={"loser": "winner", "winner": "loser"})[
                ["loser", "winner", "datetime"]
            ]
            w = abs(w)
        whole = int(math.floor(w))
        frac = w - whole
        parts = []
        if whole > 0:
            parts.extend([e] * whole)
        if frac > 0:
            n_frac = int(math.floor(frac * len(e)))
            if n_frac > 0:
                parts.append(e.iloc[:n_frac].copy())
        if not parts:
            return e.iloc[0:0].copy()
        return pd.concat(parts, ignore_index=True)

    try:
        ev_approach = events_from_interactions(approaches, pd.DataFrame(), animals=animals_list)
        ev_escape = events_from_interactions(pd.DataFrame(), escapes, animals=animals_list)
        events = pd.concat(
            [
                _apply_event_weight(ev_approach, args.approach_weight),
                _apply_event_weight(ev_escape, args.escape_weight),
            ],
            axis=0,
            ignore_index=True,
        )
        if not events.empty:
            events = (
                events
                .dropna(subset=["loser", "winner", "datetime"])
                .sort_values("datetime")
                .reset_index(drop=True)
            )
    except KeyError:
        def _mk(df: pd.DataFrame, w_col: str, l_col: str) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame(columns=["t", "winner", "loser"])
            e = pd.DataFrame({
                "t": pd.to_datetime(df.get("datetime", df.get("t")), utc=False, errors="coerce"),
                "winner": df[w_col].astype(int),
                "loser":  df[l_col].astype(int),
            })
            return e.dropna(subset=["t"])

        # i→j => i wins; j escapes i => j wins
        ev_approach = _mk(approaches, "idA", "idB").rename(columns={"t": "datetime"})
        ev_escape = _mk(escapes, "idB", "idA").rename(columns={"t": "datetime"})
        events = pd.concat(
            [
                _apply_event_weight(ev_approach, args.approach_weight),
                _apply_event_weight(ev_escape, args.escape_weight),
            ],
            axis=0,
            ignore_index=True,
        )

        keep = set(animals_list)
        events = events[
            events["winner"].isin(keep) & events["loser"].isin(keep)
        ].sort_values("datetime").drop_duplicates()

    print(
        f"[dominance stream] events={len(events)} "
        f"(approach_w={args.approach_weight:g}, escape_w={args.escape_weight:g})"
    )

    # now rank the stream (use the list of ints)
    pl_stream = rank_openskill_stream(
        events, animal_ids=animals_list, prior=None, history_stride=1
    )

    # DeepEcoHab-like final ranking from current OpenSkill ratings (not from downsampled history).
    # Keep ordinal output and a normalized 0-1 projection for existing downstream compatibility.
    ord_final = pd.Series(
        {k: float(v.ordinal()) for k, v in pl_stream.ranking.items()},
        dtype=float,
        name="dominance_ordinal",
    ).round(3)
    if ord_final.empty:
        dom_idx_pl = pd.Series(dtype=float, name="dominance_index")
    else:
        mn, mx = float(ord_final.min()), float(ord_final.max())
        dom_idx_pl = ((ord_final - mn) / (mx - mn)) if mx > mn else (ord_final * 0.0)
        dom_idx_pl.name = "dominance_index"

    # resample snapshots for visualization/exports
    Ord_bin, Mu_bin, Sig_bin = resample_ord(pl_stream.Ord, pl_stream.Mu, pl_stream.Sig, bin="15min")

    # ── Spatial metrics: 3×3 grid (center included in transitions)
    spa_m = spatial_metrics_polars(traj, grid_n=3)
    spa_idx = spatial_index(spa_m, scale=args.social_scale, weights=None)

    # ── Relabel to RFID tails + pad all outputs to include everyone
    for df in (soc_m, spa_m):
        df.index = df.index.map(map_id)
        df.index.name = "RFID"

    soc_idx.index = soc_idx.index.map(map_id)
    spa_idx.index = spa_idx.index.map(map_id)

    # relabel dominance indices
    dom_idx_bt.index = dom_idx_bt.index.map(map_id)
    dom_idx_bt = pad_to_all_labels(dom_idx_bt)

    dom_idx_pl.index = dom_idx_pl.index.map(map_id)
    dom_idx_pl = pad_to_all_labels(dom_idx_pl)
    ord_final.index = ord_final.index.map(map_id)
    ord_final = pad_to_all_labels(ord_final)

    # choose primary dominance index for combined plots/flat CSV
    dom_idx = dom_idx_pl
    dom_title = "Dominance index (PL stream)"

    # pad the rest
    soc_idx = pad_to_all_labels(soc_idx)
    spa_idx = pad_to_all_labels(spa_idx)
    soc_m = pad_to_all_labels(soc_m)
    spa_m = pad_to_all_labels(spa_m)

    # ── Save CSVs / plots
    if not W_capped.empty:
        W_out = W_capped.copy()
        W_out.index = W_out.index.map(map_id)
        W_out.columns = W_out.columns.map(map_id)
        W_out.to_csv(out / "dominance_pairwise_W.csv")
        if not args.no_heatmap:
            plot_W_heatmap(W_out, out / "dominance_pairwise_W_heatmap.png")

    soc_m.to_csv(out / "social_metrics.csv")
    spa_m.to_csv(out / "spatial_metrics.csv")
    soc_idx.to_csv(out / "social_index.csv")
    spa_idx.to_csv(out / "spatial_index.csv")

    # save both dominance indices; keep legacy name pointing to PL stream
    dom_idx.to_csv(out / "dominance_index.csv")                 # primary (PL stream)
    dom_idx_pl.to_csv(out / "dominance_index_pl_stream.csv")    # explicit PL file
    ord_final.to_csv(out / "dominance_ordinal_pl_stream.csv")   # DeepEcoHab-style ordinal
    dom_idx_bt.to_csv(out / "dominance_index_bt_mm.csv")        # BT/MM baseline

    pd.concat([soc_idx, spa_idx, dom_idx], axis=1).to_csv(
        out / "social_spatial_dom_index.csv"
    )

    # also save time-resolved ordinals/skill for later analyses
    Ord_bin.columns = Ord_bin.columns.map(map_id)
    Mu_bin.columns  = Mu_bin.columns.map(map_id)
    Sig_bin.columns = Sig_bin.columns.map(map_id)
    Ord_bin.to_csv(out / "dominance_time_ordinals.csv")
    Mu_bin.to_csv(out / "dominance_time_mu.csv")
    Sig_bin.to_csv(out / "dominance_time_sigma.csv")

    # ── Summary plots (use primary dominance = PL)
    plot_indices_combined(soc_idx, spa_idx, dom_idx, out)
    plot_scatter_social_vs_spatial(soc_idx, spa_idx, out)
    plot_bars_simple(dom_idx, dom_title, out / "dominance_bar.png", ylabel="index")
    plot_bars_simple(soc_idx, "Social index", out / "social_index_bar.png", ylabel="index")
    plot_bars_simple(spa_idx, "Spatial index", out / "spatial_index_bar.png", ylabel="index")

    # Components (example groups)
    social_inputs = ["approaches_per_min", "escapes_per_min", "pct_time_social"]
    social_names = {
        "approaches_per_min": "Fast approaches / min",
        "escapes_per_min": "Fast escapes / min",
        "pct_time_social": "% time social",
    }
    plot_category_multipanel(
        soc_m,
        social_inputs,
        social_names,
        "Social index — component metrics",
        out / "social_components.png",
    )

    locomotion = [
        "avg_speed_mm_s",
        "max_speed_mm_s",
        "total_distance_mm",
        "pct_time_moving",
        "n_speed_bursts",
    ]
    loc_names = {
        "avg_speed_mm_s": "Mean speed (mm/s)",
        "max_speed_mm_s": "Max speed (mm/s)",
        "total_distance_mm": "Total distance (mm)",
        "pct_time_moving": "% time moving",
        "n_speed_bursts": "# speed bursts",
    }
    plot_category_multipanel(
        spa_m, locomotion, loc_names, "Spatial — Locomotion", out / "spatial_locomotion.png"
    )

    arena_use = ["pct_time_center", "pct_time_periphery", "corner_entries"]
    arena_names = {
        "pct_time_center": "% time center",
        "pct_time_periphery": "% time periphery",
        "corner_entries": "Corner entries",
    }
    plot_category_multipanel(
        spa_m, arena_use, arena_names, "Spatial — Arena use", out / "spatial_arena.png"
    )

    strategy = ["zone_transitions", "roaming_entropy", "mean_turn_angle"]
    strat_names = {
        "zone_transitions": "Zone transitions",
        "roaming_entropy": "Roaming entropy",
        "mean_turn_angle": "Mean turn angle",
    }
    plot_category_multipanel(
        spa_m, strategy, strat_names, "Spatial — Exploration strategy", out / "spatial_strategy.png"
    )

    vertical = ["pct_time_rearing"]
    vert_names = {"pct_time_rearing": "% time rearing"}
    plot_category_multipanel(
        spa_m, vertical, vert_names, "Spatial — Vertical activity", out / "spatial_vertical.png"
    )

    # ── Dominance over time (OpenSkill) pretty plots (if we have any events)
    if not pl_stream.Ord.empty:
        lbl_map = {str(i): map_id(i) for i in animals_list}
        Ord_lbl = pl_stream.Ord.copy(); Ord_lbl.columns = [lbl_map.get(c, c) for c in Ord_lbl.columns]
        Mu_lbl  = pl_stream.Mu.copy();  Mu_lbl.columns  = Ord_lbl.columns
        Sig_lbl = pl_stream.Sig.copy(); Sig_lbl.columns = Ord_lbl.columns

        Ord_b, Mu_b, Sig_b = resample_ord(Ord_lbl, Mu_lbl, Sig_lbl, bin="15min")
        plot_dominance_ordinals(Ord_b, Mu_b, Sig_b, out)
        plot_dominance_rank_heatmap(Ord_b, out)
        plot_dominance_stability(Ord_b, out)
        plot_dominance_over_time(Ord_bin, out, normalize=False, smooth_win=3)
        plot_dominance_normalized(Ord_b, out)

    print(f"\n🎉 Done in {time.time() - t0:.1f}s →  {out}\n")


if __name__ == "__main__":
    main()
