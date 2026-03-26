# plots.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import median_filter
from sklearn.mixture import GaussianMixture

from theme import (
    set_theme, style_axes, cute_bar_effects, save_fig,
    PRIMARY, MID, DARK, INK, ACCENTS, lighten, _blend,
    get_label_order
)

# ---- label-order helpers (no global leakage) ----
def _ord_s(s: pd.Series) -> pd.Series:
    order = get_label_order()
    return s.reindex(order).fillna(0.0) if order is not None else s

def _ord_df(df: pd.DataFrame) -> pd.DataFrame:
    order = get_label_order()
    return df.reindex(order).fillna(0.0) if order is not None else df

def _ord_square(W: pd.DataFrame) -> pd.DataFrame:
    order = get_label_order()
    return (W.reindex(index=order, columns=order).fillna(0.0)
            if order is not None else W)

# ---- colormap ----
ROSE_CMAP = LinearSegmentedColormap.from_list(
    "rose",
    [lighten(PRIMARY, 0.96), lighten(MID, 0.70), PRIMARY, DARK],
    N=256,
)

# ======================== BASIC PLOTS ========================
def plot_indices_combined(soc_idx, spa_idx, dom_idx, out: Path):
    set_theme()
    df = pd.concat(
        [soc_idx.rename("Social"), spa_idx.rename("Spatial"), dom_idx.rename("Dominance")],
        axis=1, join="outer"
    ).fillna(0.0)
    df = _ord_df(df)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    df.plot(kind="bar", ax=ax, color=ACCENTS, width=0.82, edgecolor="white", linewidth=0.8)
    cute_bar_effects(ax)
    ax.set_title("Social, Spatial, Dominance per animal")
    ax.set_ylabel("Index")
    ax.set_xlabel("Animal ID")
    ax.legend(frameon=False, ncol=3)
    style_axes(ax, y0line=True)
    save_fig(fig, out / "indices_combined.png")

def plot_scatter_social_vs_spatial(soc_idx: pd.Series, spa_idx: pd.Series, out: Path):
    set_theme()
    xy = pd.concat([soc_idx.rename("Social"), spa_idx.rename("Spatial")], axis=1, join="outer").fillna(0.0).sort_index()
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(xy["Social"], xy["Spatial"], s=240, c="#000000", alpha=0.06)
    ax.scatter(xy["Social"], xy["Spatial"], s=140, c=PRIMARY, edgecolors="white", linewidths=1.4)
    # labels
    from matplotlib import patheffects as pe
    for lab, row in xy.iterrows():
        ax.text(row["Social"], row["Spatial"], str(lab),
                color=INK, fontsize=8, ha="center", va="center", weight="bold",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white", alpha=0.8)])
    ax.set_xlabel("Social index"); ax.set_ylabel("Spatial index")
    ax.set_title("Social vs Spatial")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    style_axes(ax, y0line=False)
    save_fig(fig, out / "social_vs_spatial.png")

def plot_bars_simple(series: pd.Series, title: str, fname: Path, ylim=None, ylabel=""):
    set_theme()
    s = _ord_s(series)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(s.index.astype(str), s.values, color=PRIMARY, edgecolor="white", linewidth=0.9, width=0.82)
    cute_bar_effects(ax)
    ax.set_title(title); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    style_axes(ax, y0line=True)
    save_fig(fig, fname)

def plot_W_heatmap(W: pd.DataFrame, out_path: Path, title: str = "Pairwise wins (row beats column)"):
    set_theme()
    W = _ord_square(W)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(W.values, aspect="auto", cmap=ROSE_CMAP)
    ax.set_title(title)
    ax.set_xticks(range(len(W.columns))); ax.set_xticklabels(W.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(W.index)));   ax.set_yticklabels(W.index, fontsize=8)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("value", rotation=90)
    save_fig(fig, out_path)

# ======================== SPEED PLOTS + HELPERS ========================
def _median_filter_animal(a):
    aid, arr, win = a
    return aid, median_filter(arr, size=win, mode="nearest")

def _smooth_speeds_parallel(df: pd.DataFrame, smooth_win: int, workers: int) -> pd.DataFrame:
    if not (smooth_win and smooth_win > 1 and smooth_win % 2 == 1):
        return df
    items = [(aid, g["speed"].to_numpy(), smooth_win) for aid, g in df.groupby("ANIMALID", sort=False)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_median_filter_animal, items, chunksize=32))
    parts = []
    for aid, arr in results:
        gg = df[df["ANIMALID"] == aid].copy()
        gg["speed"] = arr
        parts.append(gg)
    return pd.concat(parts, axis=0, ignore_index=True)

def plot_speed_distributions(
    traj: pd.DataFrame,
    out: Path,
    logx: bool = False,
    smooth_win: int = 0,
    workers: int = 1,
    stationary_cutoff_mm_s: float = 2.0,
    fit_gmm: bool = False,
    vlines_mm_s: Optional[List[float]] = None,
    vline_labels: Optional[List[str]] = None,
):
    set_theme()
    df = traj.loc[traj["ANIMALID"].notna(), ["ANIMALID", "speed"]].copy()
    df["ANIMALID"] = df["ANIMALID"].astype(int)
    df = _smooth_speeds_parallel(df, smooth_win, workers)

    df["speed"] = df["speed"].clip(lower=0).astype(float)
    moving = df[df["speed"] > stationary_cutoff_mm_s].copy()
    stationary = df[df["speed"] <= stationary_cutoff_mm_s].copy()

    ub = float(np.nanpercentile(moving["speed"], 99.5)) if len(moving) else float(np.nanpercentile(df["speed"], 99.5))
    ub = np.clip(ub * 1.1, 100.0, 2000.0)
    lin_bins = np.linspace(0.0, ub, 60)

    # overall (linear)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    if vlines_mm_s and len(vlines_mm_s) >= 2:
        move_thr, burst_thr = float(vlines_mm_s[0]), float(vlines_mm_s[1])
        stationary = df[df["speed"] <= move_thr]
        locomotor = df[(df["speed"] > move_thr) & (df["speed"] < burst_thr)]
        bursts = df[df["speed"] >= burst_thr]
        ub = float(np.nanpercentile(df["speed"], 99.5)); ub = np.clip(ub * 1.1, 100.0, 2000.0)
        lin_bins = np.linspace(0.0, ub, 60)
        ax.hist(stationary["speed"].values, bins=np.linspace(0.0, move_thr, 12),
                edgecolor="white", linewidth=0.8, color=lighten(PRIMARY, 0.40),
                label=f"≤ {move_thr:.1f} mm/s (stationary)")
        ax.hist(locomotor["speed"].values, bins=lin_bins,
                edgecolor="white", linewidth=0.8, color=PRIMARY, alpha=0.95,
                label=f"{move_thr:.1f}–{burst_thr:.1f} mm/s (locomotor)")
        ax.hist(bursts["speed"].values, bins=lin_bins,
                edgecolor="white", linewidth=0.8, color=DARK, alpha=0.9,
                label=f"≥ {burst_thr:.1f} mm/s (burst)")
        for k, thr in enumerate(vlines_mm_s[:2]):
            lbl = (vline_labels[k] if (vline_labels and k < len(vline_labels)) else f"{thr:.0f} mm/s")
            ax.axvline(thr, linestyle="--", linewidth=1.4, color="#000000", alpha=0.7)
            ax.text(thr, ax.get_ylim()[1] * 0.92, lbl, rotation=90, va="top", ha="right",
                    fontsize=8, color="#000000")
    else:
        ax.hist(stationary["speed"].values, bins=np.linspace(0.0, stationary_cutoff_mm_s, 6),
                edgecolor="white", linewidth=0.8, color=lighten(PRIMARY, 0.4),
                label=f"≤ {stationary_cutoff_mm_s} mm/s")
        ax.hist(moving["speed"].values, bins=lin_bins,
                edgecolor="white", linewidth=0.8, color=PRIMARY, alpha=0.95,
                label=f"> {stationary_cutoff_mm_s} mm/s")

    ax.set_xlabel("Speed (mm/s)"); ax.set_ylabel("Count"); ax.set_title("Mouse speed distribution (overall)")
    ax.legend(frameon=False); style_axes(ax, y0line=False)
    save_fig(fig, out / "speed_hist_overall.png")

    # per-animal (linear)
    ids = sorted(df["ANIMALID"].unique())
    n = len(ids); ncols = min(3, max(1, int(np.ceil(np.sqrt(n))))); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols + 1.2, 2.8*nrows + 1.0), squeeze=False)
    k = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if k < n:
                aid = ids[k]
                s_all = df.loc[df["ANIMALID"] == aid, "speed"].values
                s_mov = s_all[s_all > stationary_cutoff_mm_s]
                ax.hist(s_mov, bins=lin_bins, edgecolor="white", linewidth=0.7, color=MID)
                ax.set_title(f"Animal {aid}", fontsize=10, pad=6)
                ax.set_xlabel("mm/s"); ax.set_ylabel("Count")
                style_axes(ax, y0line=False)
            else:
                ax.set_visible(False)
            k += 1
    fig.suptitle(f"Speed distributions by animal (> {stationary_cutoff_mm_s} mm/s)", x=0.02, ha="left",
                 color=_blend(PRIMARY, "#000000", 0.25), fontsize=13, fontweight="bold")
    save_fig(fig, out / "speed_hist_by_animal.png")

    # log-x variants (optional)
    if logx:
        s = moving["speed"].values
        lo = max(np.nanmin(s[s > 0]) if s.size else 0.1, 0.1)
        hi = max(lo * 1.01, np.nanmax(s) if s.size else 10.0)
        bins = np.geomspace(lo, hi, 60)

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.set_xscale("log")
        if vlines_mm_s and len(vlines_mm_s) >= 2:
            move_thr, burst_thr = float(vlines_mm_s[0]), float(vlines_mm_s[1])
            lo = max(np.nanmin(df["speed"][df["speed"] > 0]), 0.1)
            hi = max(lo * 1.01, np.nanmax(df["speed"]))
            bins = np.geomspace(lo, hi, 60)
            stationary = df[df["speed"] <= move_thr]["speed"].values
            locomotor = df[(df["speed"] > move_thr) & (df["speed"] < burst_thr)]["speed"].values
            bursts = df[df["speed"] >= burst_thr]["speed"].values
            if stationary.size:
                ax.hist(stationary, bins=bins, edgecolor="white", linewidth=0.8, color=lighten(PRIMARY, 0.40),
                        label=f"≤ {move_thr:.1f} mm/s (stationary)")
            if locomotor.size:
                ax.hist(locomotor, bins=bins, edgecolor="white", linewidth=0.8, color=PRIMARY, alpha=0.95,
                        label=f"{move_thr:.1f}–{burst_thr:.1f} mm/s (locomotor)")
            if bursts.size:
                ax.hist(bursts, bins=bins, edgecolor="white", linewidth=0.8, color=DARK, alpha=0.9,
                        label=f"≥ {burst_thr:.1f} mm/s (burst)")
            for k, thr in enumerate(vlines_mm_s[:2]):
                lbl = (vline_labels[k] if (vline_labels and k < len(vline_labels)) else f"{thr:.0f} mm/s")
                ax.axvline(thr, linestyle="--", linewidth=1.4, color="#000000", alpha=0.7)
                ax.text(thr, ax.get_ylim()[1] * 0.92, lbl, rotation=90, va="top", ha="right", fontsize=8, color="#000000")
        else:
            ax.hist(s, bins=bins, edgecolor="white", linewidth=0.8, color=PRIMARY)

        total_n = len(df)
        pct_burst = (len(df[df["speed"] >= (vlines_mm_s[1] if vlines_mm_s else np.inf)]) / total_n * 100.0) if total_n else 0.0
        ax.text(0.98, 0.98, f"Burst frames: {pct_burst:.1f}%", transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=INK)
        ax.set_xlabel("Speed (mm/s) [log scale]"); ax.set_ylabel("Count")
        ax.set_title("Mouse speed distribution (overall, log-x)")
        style_axes(ax, y0line=False); ax.legend(frameon=False)
        save_fig(fig, out / "speed_hist_overall_logx.png")

# ======================== MULTIPANELS ========================
def plot_category_multipanel(df: pd.DataFrame, metrics: list[str], display_names: dict[str,str],
                             title: str, out_path: Path, ncols: int = 3):
    set_theme()
    df = _ord_df(df)
    n = len(metrics); ncols = min(ncols, n)
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols+1.6, 3.0*nrows+1.4), squeeze=False)
    cols = [MID, PRIMARY, DARK, _blend(MID, "#000000", .1)]
    for k, metric in enumerate(metrics):
        r,c = divmod(k, ncols); ax = axes[r][c]
        s = df[metric]
        ax.bar(s.index.astype(str), s.values, color=cols[k % len(cols)], edgecolor="white", linewidth=0.8, width=0.82)
        cute_bar_effects(ax)
        ax.set_title(display_names.get(metric, metric), pad=8)
        ax.set_ylabel("")
        style_axes(ax, y0line=True)
    # hide empty cells
    for k in range(n, nrows*ncols):
        r,c = divmod(k, ncols); axes[r][c].set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", color=_blend(PRIMARY,"#000000",0.25), fontsize=13, fontweight="bold")
    save_fig(fig, out_path)
# ======================== DOMINANCE (TIME SERIES) ========================
def plot_dominance_ordinals(Ord_bin: pd.DataFrame, Mu_bin: pd.DataFrame, Sig_bin: pd.DataFrame, out: Path):
    if Ord_bin.empty: return
    set_theme()
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for a in Ord_bin.columns:
        ord_s = Ord_bin[a]
        ax.plot(Ord_bin.index, ord_s, linewidth=1.6, label=str(a))
        if not Mu_bin.empty and not Sig_bin.empty:
            sig = Sig_bin[a]
            if sig.notna().any():
                # Keep uncertainty centered on the plotted ordinal line.
                ax.fill_between(Ord_bin.index, ord_s - 2*sig, ord_s + 2*sig, alpha=0.10)
    ax.set_title("Dominance (ordinal) over time")
    ax.set_xlabel("time"); ax.set_ylabel("ordinal rating")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    style_axes(ax, y0line=False)
    save_fig(fig, out / "dominance_ordinals.png")

def plot_dominance_rank_heatmap(Ord_bin: pd.DataFrame, out: Path):
    if Ord_bin.empty: return
    set_theme()
    Ranks = Ord_bin.rank(axis=1, ascending=False, method="min")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.imshow(Ranks.T.values, aspect="auto", interpolation="nearest")
    ax.set_title("Rank positions over time (1=top)")
    ax.set_yticks(range(Ranks.shape[1])); ax.set_yticklabels(Ranks.columns)
    # sparse x ticks
    xt = np.linspace(0, Ranks.shape[0]-1, num=min(10, Ranks.shape[0])).astype(int)
    ax.set_xticks(xt); ax.set_xticklabels([str(Ranks.index[i]) for i in xt], rotation=45, ha="right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("rank", rotation=90)
    save_fig(fig, out / "dominance_rank_heatmap.png")

def plot_dominance_stability(Ord_bin: pd.DataFrame, out: Path):
    if Ord_bin.empty: return
    set_theme()
    # hierarchy stability via pairwise inversions between consecutive bins
    flips = []
    prev = Ord_bin.iloc[0].sort_values(ascending=False).index.tolist()
    for t in Ord_bin.index[1:]:
        cur = Ord_bin.loc[t].sort_values(ascending=False).index.tolist()
        pos = {a:i for i,a in enumerate(prev)}
        inv = 0
        for i, a in enumerate(cur):
            for j in range(i+1, len(cur)):
                b = cur[j]
                if pos.get(a, 0) > pos.get(b, 0):
                    inv += 1
        flips.append((t, inv))
        prev = cur
    INV = pd.Series(dict(flips))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(INV.index, INV.values, linewidth=1.6, label="pairwise inversions")
    ax.set_title("Hierarchy stability"); ax.set_xlabel("time"); ax.set_ylabel("inversions")
    ax.legend(frameon=False); style_axes(ax, y0line=False)
    save_fig(fig, out / "dominance_stability.png")

def plot_dominance_over_time(
    Ord_bin: pd.DataFrame,
    out: Path,
    title: str = "Dominance over time (OpenSkill ordinal)",
    normalize: bool = False,           # 0–1 scale across all animals/time
    smooth_win: Optional[int] = None,  # e.g., 3 or 5 bins for a gentle moving average
):
    if Ord_bin.empty:
        return
    set_theme()

    # enforce RFID plotting order on columns
    df = _ord_df(Ord_bin.T).T

    # optional smoothing (centered rolling mean on the time axis)
    if smooth_win and smooth_win >= 3:
        df = df.rolling(window=smooth_win, center=True, min_periods=max(1, smooth_win // 2)).mean()

    # optional 0–1 normalization across the whole figure (not per animal)
    if normalize:
        mn = float(np.nanmin(df.values))
        mx = float(np.nanmax(df.values))
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            df = (df - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for col in df.columns:
        ax.plot(df.index, df[col].values, linewidth=1.8, label=str(col))

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Dominance" + (" (0–1)" if normalize else " (ordinal)"))
    ax.legend(frameon=False, ncol=2, fontsize=8)
    style_axes(ax, y0line=False)
    fig.autofmt_xdate()
    save_fig(fig, out / "dominance_over_time.png")

def plot_dominance_normalized(Ord_bin: pd.DataFrame, out: Path, title: str = "Dominance rank over time"):
    """
    Line plot: each animal is a line, y-axis = rank (1 = most dominant).
    """
    if Ord_bin.empty:
        return

    set_theme()

    # 1 = highest ordinal (most dominant), larger numbers = lower rank
    R = Ord_bin.rank(axis=1, ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for col in R.columns:
        ax.plot(R.index, R[col].values, linewidth=2.0, label=str(col))

    # nicer: rank 1 at the top of the axis
    ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("rank (1 = most dominant)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    style_axes(ax, y0line=False)
    fig.autofmt_xdate()
    save_fig(fig, out / "dominance_rank_lines.png")
