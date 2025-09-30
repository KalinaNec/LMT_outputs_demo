#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, glob, warnings, itertools, re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss

# ---- headless matplotlib for servers/CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap


# ============================ CLI ============================
def parse_args():
    p = argparse.ArgumentParser(
        description="Correlate LMT indices with T-maze performance (Spearman + permutation p + meta)."
    )
    p.add_argument("--base", type=Path, default=Path("outputs"),
                   help="Root folder that contains per-group CSVs (social/spatial/dominance).")
    p.add_argument("--tmaze", type=Path, default=Path("data/tmaze_summary.csv"),
                   help="Path to tmaze_summary.csv (columns: rfid, day, accuracy).")
    p.add_argument("--group", default=None,
                   help="Optional group filter like C1G1 (case-insensitive). Omit to run all groups.")
    p.add_argument("--out", type=Path, default=Path("outputs/stats"),
                   help="Where to save CSVs/plots.")
    return p.parse_args()


# ====================== CONFIG / THEME =======================
N_PERM = 10_000
EXACT_N_MAX = 9
warnings.filterwarnings("ignore", message="`kurtosistest` p-value may be inaccurate")

PRIMARY = "#a93a5e"; MID = "#a67081"; DARK = "#831b3c"
def _hex2rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
def _rgb2hex(rgb): return "#{:02x}{:02x}{:02x}".format(*[max(0,min(255,int(round(x)))) for x in rgb])
def _blend(c1,c2,a):
    r1,g1,b1=_hex2rgb(c1); r2,g2,b2=_hex2rgb(c2)
    return _rgb2hex((r1*(1-a)+r2*a, g1*(1-a)+g2*a, b1*(1-a)+b2*a))
def lighten(c,amt=0.92): return _blend(c, "#ffffff", amt)
def darken(c,amt=0.20):  return _blend(c, "#000000", amt)
BLUSH_BG = lighten(PRIMARY, 0.92); GRID = lighten(MID, 0.75); INK = _blend(DARK, "#000000", 0.35)
ROSE_CMAP = LinearSegmentedColormap.from_list("rose",
    [lighten(PRIMARY, 0.96), lighten(MID, 0.70), PRIMARY, DARK], N=256)

def _set_theme():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": BLUSH_BG,
        "axes.edgecolor": _blend(PRIMARY, "#000000", 0.25),
        "axes.linewidth": 1.1,
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": _blend(PRIMARY, "#000000", 0.25),
        "xtick.color": INK, "ytick.color": INK,
        "grid.color": GRID, "grid.alpha": 0.9,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 300,
    })

def _style_axes(ax, y0line=True):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="both", linestyle=":", linewidth=0.9)
    ymin, ymax = ax.get_ylim()
    if y0line and (ymin < 0 < ymax):
        ax.axhline(0, color=_blend(PRIMARY, "#000000", 0.25), linewidth=1.0, alpha=0.6, zorder=0)
    ax.tick_params(axis="x", labelrotation=0)

def _cute_scatter(ax, x, y, labels=None):
    ax.scatter(x, y, s=240, c="#000000", alpha=0.06)     # glow
    ax.scatter(x, y, s=120, c=PRIMARY, edgecolors="white", linewidths=1.2)
    if labels is not None:
        for xi, yi, lab in zip(x, y, labels):
            ax.text(xi, yi, str(lab), color=INK, fontsize=9, ha="center", va="center", weight="bold",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white", alpha=0.9)])

def _save(fig, path: Path):
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# ================== STATS / META HELPERS ====================
def spearman_perm(x, y, n_perm=N_PERM, exact_n_max=EXACT_N_MAX, rng=None):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    r_obs, _ = ss.spearmanr(x, y)
    if n <= exact_n_max:
        cnt = 0; tot = 0
        for perm in itertools.permutations(range(n)):
            r_perm, _ = ss.spearmanr(x, y[list(perm)])
            if abs(r_perm) >= abs(r_obs): cnt += 1
            tot += 1
        return float(r_obs), cnt / tot
    if rng is None:
        rng = np.random.default_rng(42)
    cnt = 0
    for _ in range(n_perm):
        r_perm, _ = ss.spearmanr(x, rng.permutation(y))
        if abs(r_perm) >= abs(r_obs): cnt += 1
    p = (cnt + 1) / (n_perm + 1)  # add-one smoothing
    return float(r_obs), float(p)

def spearman_perm_matrix(df: pd.DataFrame):
    cols = list(df.columns)
    R = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    P = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            r, p = spearman_perm(df[a], df[b])
            R.iloc[i, j] = r; P.iloc[i, j] = p
    return R, P

def bh_fdr(pvals: pd.DataFrame) -> pd.DataFrame:
    n = len(pvals)
    tri = [(i, j) for i in range(n) for j in range(i+1, n)]
    flat = np.array([pvals.values[i, j] for (i, j) in tri], dtype=float)
    valid = np.isfinite(flat)
    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        q = np.full_like(pvals.values, np.nan, dtype=float)
        np.fill_diagonal(q, 0.0)
        return pd.DataFrame(q, index=pvals.index, columns=pvals.columns)
    order = np.argsort(flat[valid])
    m = order.size
    ranks = np.arange(1, m+1)
    q_sorted = np.minimum.accumulate((flat[valid][order] * m / ranks)[::-1])[::-1]
    q = np.full_like(pvals.values, np.nan, dtype=float)
    for (k, qv) in zip(idx_valid[order], q_sorted):
        i, j = tri[k]
        q[i, j] = qv; q[j, i] = qv
    np.fill_diagonal(q, 0.0)
    return pd.DataFrame(q, index=pvals.index, columns=pvals.columns)

def _fisher_z(r, eps=1e-12):
    r = np.clip(r, -1+eps, 1-eps)
    return np.arctanh(r)

def _fisher_z_ci(r, n, alpha=0.05):
    if not np.isfinite(r) or n <= 3: return (np.nan, np.nan)
    z = _fisher_z(r); se = 1/np.sqrt(max(n-3, 1))
    z_lo = z - ss.norm.ppf(1-alpha/2)*se
    z_hi = z + ss.norm.ppf(1-alpha/2)*se
    return (np.tanh(z_lo), np.tanh(z_hi))

def run_groupwise_meta(df_all: pd.DataFrame,
                       pairs: list[tuple[str, str]],
                       out_dir: Path,
                       out_prefix: str = "meta_") -> pd.DataFrame:
    """Per-pair per-group exact Spearman + Fisher-z meta + Stouffer p; forest plots."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def _exact_spearman(x, y, exact_n_max=9, n_perm=10000, rng=None):
        x = np.asarray(x); y = np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        n = len(x)
        if n < 3:
            return np.nan, np.nan, n
        r_obs, _ = ss.spearmanr(x, y)
        if n <= exact_n_max:
            cnt = 0; tot = 0
            for perm in itertools.permutations(range(n)):
                r_perm, _ = ss.spearmanr(x, y[list(perm)])
                if abs(r_perm) >= abs(r_obs): cnt += 1
                tot += 1
            return float(r_obs), cnt / tot, n
        if rng is None: rng = np.random.default_rng(42)
        cnt = 0
        for _ in range(n_perm):
            r_perm, _ = ss.spearmanr(x, rng.permutation(y))
            if abs(r_perm) >= abs(r_obs): cnt += 1
        p = (cnt + 1) / (n_perm + 1)
        return float(r_obs), float(p), n

    def _meta_spearman(rows):
        rows = [r for r in rows if np.isfinite(r["r"]) and r["n"] > 3]
        if not rows:
            return {"r_meta": np.nan, "lo": np.nan, "hi": np.nan, "p_stouffer": np.nan}
        z = np.array([_fisher_z(r["r"]) for r in rows])
        w = np.array([r["n"] - 3 for r in rows], float)
        z_bar = np.sum(w * z) / np.sum(w)
        se_bar = 1 / np.sqrt(np.sum(w))
        lo = np.tanh(z_bar - ss.norm.ppf(0.975) * se_bar)
        hi = np.tanh(z_bar + ss.norm.ppf(0.975) * se_bar)
        r_meta = np.tanh(z_bar)
        p_list = np.clip([r["p"] for r in rows], 1e-12, 1-1e-12)
        z_one = ss.norm.isf(np.array(p_list)/2.0)
        Z = np.sum(np.sqrt(w) * z_one) / np.sqrt(np.sum(w))
        p_st = 2*ss.norm.sf(abs(Z))
        return {"r_meta": r_meta, "lo": lo, "hi": hi, "p_stouffer": p_st}

    def _forest_plot(per_group_rows, meta, title, fname):
        _set_theme()
        rows = [r for r in per_group_rows if np.isfinite(r["r"])]
        labels = [r["group"] for r in rows] + ["META"]
        rvals  = [r["r"] for r in rows] + [meta["r_meta"]]
        los    = [r["lo"] for r in rows] + [meta["lo"]]
        his    = [r["hi"] for r in rows] + [meta["hi"]]
        ptxt   = [f"p={r['p']:.3f}" for r in rows] + [f"p={meta['p_stouffer']:.3f}"]
        y = np.arange(len(labels))[::-1]
        fig, ax = plt.subplots(figsize=(7.2, 0.6*len(labels)+2.0))
        for yi, lo, hi in zip(y, los, his):
            ax.plot([lo, hi], [yi, yi], color=MID, lw=2)
        ax.scatter(rvals, y, s=90, c=PRIMARY, edgecolors="white", linewidths=1.2, zorder=3)
        ax.axvline(0, color=_blend(PRIMARY,"#000",0.35), lw=1, ls="--", alpha=0.6)
        for yi, t in zip(y, ptxt):
            ax.text(1.02, yi, t, transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=DARK)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("Spearman ρ (95% CI via Fisher z)")
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(title)
        _style_axes(ax, y0line=False)
        fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)

    results = []
    groups = sorted(df_all["group"].dropna().unique())
    for x, y in pairs:
        per_group = []
        for g in groups:
            sub = df_all[df_all["group"] == g]
            r, p, n = _exact_spearman(sub[x], sub[y]) if len(sub) >= 3 else (np.nan, np.nan, len(sub))
            lo, hi = _fisher_z_ci(r, n)
            per_group.append({"pair": f"{y}~{x}", "group": g, "n": n, "r": r, "lo": lo, "hi": hi, "p": p})
        meta = _meta_spearman(per_group)

        results.append({
            "pair": f"{y}~{x}", "level": "meta", "group": "ALL",
            "n_total": int(np.nansum([r["n"] for r in per_group])),
            "r": meta["r_meta"], "lo": meta["lo"], "hi": meta["hi"], "p_stouffer": meta["p_stouffer"]
        })
        for rrow in per_group:
            results.append({
                "pair": rrow["pair"], "level": "group", "group": rrow["group"], "n": rrow["n"],
                "r": rrow["r"], "lo": rrow["lo"], "hi": rrow["hi"], "p_exact": rrow["p"]
            })

        _forest_plot(
            per_group, meta,
            title=f"{y} vs {x} (Spearman, exact p; meta z & Stouffer p)",
            fname=out_dir / f"{out_prefix}forest_{y}_vs_{x}.png"
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_dir / f"{out_prefix}spearman_meta.csv", index=False)
    print(f"Saved: {out_dir / f'{out_prefix}spearman_meta.csv'} and forest plots per pair.")
    return out_df


# ====================== PLOTTING HELPERS ====================
def corr_heatmap(R: pd.DataFrame, out_png: Path,
                 title: str = "Spearman ρ (permutation p in CSV)",
                 label_map: dict[str,str] | None = None,
                 max_label_width: int = 12):
    _set_theme()

    def _wrap(s: str, w: int = 12):
        s = label_map.get(s, s) if label_map else s
        if len(s) <= w: return s
        cut = max(s.rfind(" ", 0, w), 0)
        return s if cut == 0 else s[:cut] + "\n" + s[cut+1:]

    xlabels = [_wrap(c, max_label_width) for c in R.columns]
    ylabels = [_wrap(r, max_label_width) for r in R.index]

    n = len(R.columns)
    fig_w = max(9.0, 1.6 * n)
    fig_h = max(7.0, 1.3 * n)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(R.values, cmap=ROSE_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(xlabels, rotation=25, ha="right",
                                               rotation_mode="anchor", linespacing=1.1)
    ax.set_yticks(range(n)); ax.set_yticklabels(ylabels)
    ax.set_title(title, pad=18)

    for i in range(n):
        for j in range(n):
            v = float(R.values[i, j])
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.45 else DARK)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cbar.set_label("ρ", rotation=0, labelpad=10)
    ax.tick_params(axis="x", pad=10); ax.tick_params(axis="y", pad=8)
    fig.subplots_adjust(left=0.22, right=0.92, bottom=0.22, top=0.86)
    _style_axes(ax, y0line=False)
    _save(fig, Path(out_png))

def scatter_grid(df: pd.DataFrame, pairs: list[tuple[str,str]], out_png: Path, suptitle: str):
    _set_theme()
    n = len(pairs); ncols = 2; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8*ncols/2, 4.8*nrows), squeeze=False)
    for k, (x, y) in enumerate(pairs):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        _cute_scatter(ax, df[x], df[y], df["RFID4"])
        rho, p = spearman_perm(df[x], df[y])
        ax.set_title(f"{y} vs {x}  (ρ={rho:+.2f}, p_perm={p:.3f})")
        ax.set_xlabel(x); ax.set_ylabel(y)
        if "index" in x or "accuracy" in x: ax.set_xlim(0, 1)
        if "index" in y or "accuracy" in y: ax.set_ylim(0, 1)
        _style_axes(ax, y0line=False)
    for k in range(n, nrows*ncols):
        r, c = divmod(k, ncols); axes[r][c].set_visible(False)
    fig.suptitle(suptitle, x=0.02, ha="left", color=_blend(PRIMARY,"#000000",0.25),
                 fontsize=13, fontweight="bold")
    _save(fig, Path(out_png))


# ============================ MAIN ============================
def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # -------- file discovery (scoped to BASE) --------
    def find_files(base: Path, patterns):
        files = []
        for pat in patterns:
            files.extend(glob.glob(str(base / "**" / pat), recursive=True))
        return sorted(set(files))

    def group_from_path(f: str) -> str:
        """Best-effort group from parent dir; falls back to parent name."""
        name = Path(f).parent.name.upper()
        m = re.search(r"C\d+G\d+", name)
        return m.group(0) if m else name

    idx_files = find_files(args.base, ["social_spatial_index.csv", "social_spatial_dom_index.csv"])
    dom_files = find_files(args.base, ["dominance_index.csv"])
    if not idx_files or not dom_files:
        warnings.warn("Could not find required CSVs under --base; check paths.")
        raise SystemExit(1)

    # -------- load indices --------
    idx_frames = []
    for f in idx_files:
        df_i = pd.read_csv(f, index_col=0).reset_index(names="RFID4")
        # normalize column names, drop dominance if present (we'll bring clean one)
        df_i.columns = [c if c == "RFID4" else c.lower() for c in df_i.columns]
        if "dominance_index" in df_i.columns:
            df_i = df_i.drop(columns=["dominance_index"])
        have = [c for c in ["RFID4", "social_index", "spatial_index"] if c in df_i.columns]
        if not have:
            continue
        df_i = df_i[have]
        df_i["group"] = group_from_path(f)
        idx_frames.append(df_i)

    if not idx_frames:
        raise SystemExit("No index CSVs with expected columns were found.")

    idx_all = (pd.concat(idx_frames, ignore_index=True)
                 .drop_duplicates("RFID4", keep="first"))

    dom_all = (pd.concat([
        pd.read_csv(f, index_col=0).reset_index(names="RFID4").assign(group=group_from_path(f))
        for f in dom_files
    ], ignore_index=True).drop_duplicates("RFID4"))
    dom_all = dom_all.drop(columns=[c for c in ["group"] if c in dom_all.columns])

    # -------- T-maze summary -> learning slope & mean accuracy --------
    df_raw = pd.read_csv(args.tmaze)
    if not {"rfid","day","accuracy"}.issubset(df_raw.columns):
        raise SystemExit("tmaze_summary.csv must have columns: rfid, day, accuracy")
    df_raw["RFID4"] = df_raw.rfid.astype(str).str[-4:]

    recs = []
    for rf, sub in df_raw.groupby("RFID4", sort=False):
        slope = np.polyfit(sub.day, sub.accuracy, 1)[0] if sub.day.nunique() > 1 else np.nan
        mean_acc = sub.accuracy.mean()
        recs.append({"RFID4": str(rf), "learning_slope": slope, "mean_accuracy": mean_acc})
    learn = pd.DataFrame(recs)

    # -------- merge --------
    for tbl in (learn, idx_all, dom_all):
        tbl["RFID4"] = tbl["RFID4"].astype(str)

    df = (learn
          .merge(idx_all, on="RFID4", how="inner")
          .merge(dom_all[["RFID4", "dominance_index"]], on="RFID4", how="left", suffixes=("", "_dom")))

    # coalesce possible duplicate dom cols (defensive)
    for a, b in [("dominance_index", "dominance_index_dom"),
                 ("dominance_index_x", "dominance_index_y")]:
        if a in df.columns and b in df.columns:
            df["dominance_index"] = df[a].fillna(df[b]); df = df.drop(columns=[a, b])
    if "dominance_index_y" in df.columns and "dominance_index" not in df.columns:
        df = df.rename(columns={"dominance_index_y": "dominance_index"})
    if "dominance_index_x" in df.columns and "dominance_index" not in df.columns:
        df = df.rename(columns={"dominance_index_x": "dominance_index"})

    # bring group from idx_all (we deduped by RFID4 earlier)
    if "group" in idx_all.columns:
        df = df.merge(idx_all[["RFID4","group"]], on="RFID4", how="left")

    # optional group filter (affects pooled + meta input)
    if args.group:
        sel = str(args.group).upper()
        df = df[df.get("group", "").str.upper() == sel].copy()
        print(f"[info] Running only for group: {sel}")

    if df.empty:
        raise SystemExit("No rows after merging/filtering. Check --base / --tmaze / --group.")

    # -------- pooled correlations --------
    corr_pairs = [
        ("learning_slope", "social_index"),
        ("learning_slope", "spatial_index"),
        ("learning_slope", "dominance_index"),
        ("mean_accuracy",  "social_index"),
        ("mean_accuracy",  "spatial_index"),
        ("mean_accuracy",  "dominance_index"),
        ("social_index",   "dominance_index"),
        ("spatial_index",  "dominance_index"),
    ]

    print("\n--- Spearman ρ with permutation p (pooled across groups) ---")
    for a, b in corr_pairs:
        r, p = spearman_perm(df[a], df[b])
        print(f"{a:<15} ⊣ {b:<15}   ρ={r:+.3f}  p_perm={p:.4f}")

    corr_vars = ["learning_slope","mean_accuracy","social_index","spatial_index","dominance_index"]
    R, Pperm = spearman_perm_matrix(df[corr_vars])
    Qperm = bh_fdr(Pperm)

    args.out.mkdir(parents=True, exist_ok=True)
    R.to_csv(args.out / "corr_rho.csv")
    Pperm.to_csv(args.out / "corr_p_perm.csv")
    Qperm.to_csv(args.out / "corr_q_perm_bh.csv")
    print("\nSaved: corr_rho.csv, corr_p_perm.csv, corr_q_perm_bh.csv")

    # -------- plots (pooled) --------
    corr_heatmap(R, args.out / "corr_heatmap.png", title="Spearman (ρ) — permutation p in CSV")

    label_map = {
        "learning_slope": "Learning\nslope",
        "mean_accuracy": "Mean\naccuracy",
        "social_index": "Social\nindex",
        "spatial_index": "Spatial\nindex",
        "dominance_index": "Dominance\nindex",
    }
    title = "Spearman correlation (ρ) with permutation tests:\nLMT indices vs. T-maze performance"
    corr_heatmap(R, args.out / "corr_heatmap_poster.png", label_map=label_map, max_label_width=11, title=title)

    pairs_for_plot = [
        ("social_index", "learning_slope"),
        ("spatial_index","learning_slope"),
        ("dominance_index","learning_slope"),
        ("social_index", "mean_accuracy"),
        ("spatial_index","mean_accuracy"),
        ("dominance_index","mean_accuracy"),
        ("social_index","dominance_index"),
        ("spatial_index","dominance_index"),
    ]
    scatter_grid(df, pairs_for_plot, args.out / "scatter_grid.png",
                 suptitle="Indices vs Learning — permutation p")

    # -------- per-group meta (across whatever groups remain in df) --------
    pairs_for_meta = [
        ("social_index", "learning_slope"),
        ("spatial_index","learning_slope"),
        ("dominance_index","learning_slope"),
        ("social_index", "mean_accuracy"),
        ("spatial_index","mean_accuracy"),
        ("dominance_index","mean_accuracy"),
    ]
    if "group" in df.columns and df["group"].notna().any():
        run_groupwise_meta(df, pairs_for_meta, out_dir=args.out, out_prefix="meta_")
    else:
        print("[warn] No 'group' column present; skipping per-group meta.")

    print(f"\n✅ Done. Outputs in: {args.out.resolve()}\n")


if __name__ == "__main__":
    main()
