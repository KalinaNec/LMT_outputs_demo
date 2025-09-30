#!/usr/bin/env python
"""
Live-Mouse-Tracker  ➜  joint social + spatial + dominance pipeline
────────────────────────────────────────────────────────────────────────────
This script ingests a single LMT *.sqlite* recording and produces:

• A **Social index**  (three interaction metrics → 0‒1 score)
• A ** Spatial index, because I workn ia SPATIAL memory lab, so we know a lot about spatial stuff**  (12+ exploratory metrics → 0‒1 score)
• A **Dominance index**  ( Plackett-Luce model or David's score, rescaled 0‒1)
• A folder full of harmonised cute pink PNG plots and CSV tables.

Under construction **species presets** (mouse/rat), plus extra **profile variables**:
- Contact maintenance: Contact/Group2/Group3/Group4 → TotalLen, Nb, MeanDur, PropTimeDetection
- Movement context: Move isolated / Move in contact → PropTimeDetection
- Initiation/Avoidance: Fast approach/escape → PropNbContact (per contact episode)

Usage
─────
python LMT.py  recording.sqlite  [out_dir]

Examples
────────
python LMT.py  C1G1.sqlite
python LMT.py  C1G1.sqlite  out/C1G1 --species rat

Optional flags
──────────────
--debug         → limit to first 50k frames (fast test)
--species {mouse,rat}  → set all defaults for that species (mouse = default)
--move 20       → moving threshold in mm/s (overrides species default)
--burst 300     → burst threshold in mm/s (overrides species default)
--contact-mm 30 → contact distance in mm (overrides species default)
--iso-mm 50     → isolation distance in mm (overrides species default)
--fast-thr 200  → fast approach/escape threshold in mm/s (overrides species default)
--margin-mm 50  → center/periphery margin in mm (overrides species default)
--corner-mm 100 → corner radius in mm (overrides species default)
--social-scale {minmax,robust,zscore}  → scaling method for indices (default robust)
--weights-json  weights.json  → optional weights override for social/spatial indices

Outputs
───────
• social_metrics.csv, spatial_metrics.csv
• social_index.csv, spatial_index.csv, dominance_index.csv
• social_spatial_dom_index.csv
• profile_variables.csv (radar-ready subset of classic variables)
• social_index_bar.png, spatial_index_bar.png, dominance_bar.png, social_spatial_bar.png, social_vs_spatial.png
"""
from __future__ import annotations

# ╭────────────────────────────  PYTHON STD  ────────────────────────────────╮
import argparse
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

# ╭────────────────────────────  SCI/NUMPY STACK  ────────────────────────────╮
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from openskill.models import PlackettLuce
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

# Scaling helper for indices
def _scale(df: pd.DataFrame, method: str = "robust") -> pd.DataFrame:
    """Return a scaled copy of df. method ∈ {minmax, robust, zscore}."""
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:  # robust by default
        scaler = RobustScaler()
    arr = scaler.fit_transform(df)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

# ╰───────────────────────────────────────────────────────────────────────────╯

# ╔═══════════════════════════════════════════════════════════════════════════╗
# 0.  SPECIES PRESETS + CLI
# ╚═══════════════════════════════════════════════════════════════════════════╝
SPECIES_PRESETS: Dict[str, Dict[str, float]] = {
    # MOUSE defaults
    "mouse": {
        "contact_mm": 30.0,   # dyads if < 30 mm apart
        "iso_mm": 50.0,       # considered isolated if > 50 mm from everyone
        "move_thr": 20.0,     # moving if speed > 20 mm/s
        "burst_thr": 300.0,   # speed burst if ≥ 2 frames > 300 mm/s
        "fast_thr": 200.0,    # fast approach/escape threshold
        "margin_mm": 50.0,    # center/periphery margin
        "corner_mm": 100.0,   # corner square radius
    },
    # RAT defaults (scaled up from mouse, currently unvalidated)
    "rat": {
        "contact_mm": 60.0,
        "iso_mm": 100.0,
        "move_thr": 50.0,
        "burst_thr": 600.0,
        "fast_thr": 400.0,
        "margin_mm": 100.0,
        "corner_mm": 200.0,
    },
}

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Social, Spatial & Dominance indices for one LMT .sqlite with species presets")

    # positional args
    p.add_argument("sqlite", help="path to the input LMT recording (*.sqlite)")
    p.add_argument("out", nargs="?", default=None,
                   help="output folder for CSV/plots (default: alongside input file)")

    # species preset + overrides
    p.add_argument("--species", choices=["mouse", "rat"], default="mouse",
                   help="which species preset to use [mouse]")
    p.add_argument("--move", type=float, default=None, metavar="MM_S",
                   help="moving-vs-idle threshold in mm/s (override preset)")
    p.add_argument("--burst", type=float, default=None, metavar="MM_S",
                   help="burst threshold in mm/s (override preset)")
    p.add_argument("--contact-mm", type=float, default=None,
                   help="contact distance in mm (override preset)")
    p.add_argument("--iso-mm", type=float, default=None,
                   help="isolation distance in mm (override preset)")
    p.add_argument("--fast-thr", type=float, default=None,
                   help="fast approach/escape speed threshold in mm/s (override preset)")
    p.add_argument("--margin-mm", type=float, default=None,
                   help="center/periphery margin in mm (override preset)")
    p.add_argument("--corner-mm", type=float, default=None,
                   help="corner radius in mm (override preset)")

    # scaling/weights
    p.add_argument("--social-scale", choices=["minmax","robust","zscore"], default="robust",
                   help="scaling method for social & spatial indices [robust]")
    p.add_argument("--weights-json", default=None,
                   help="optional JSON with weights dicts, e.g. {'social': {...}, 'spatial': {...}}")

    # misc
    p.add_argument("--debug", action="store_true",
                   help="use only the first 50k frames (fast sanity check)")
    p.add_argument("--dominance", choices=["davids", "plackettluce"], default="plackettluce",
                   help="dominance method: David’s Score or Plackett–Luce [plackettluce]")
    p.add_argument("--approach-weight", type=float, default=1.0,
                   help="weight for approach events [1.0]")
    p.add_argument("--escape-weight", type=float, default=1.0,
                   help="weight for escape events [1.0]")
    p.add_argument("--dom-transform", choices=[None, "log1p", "rate"], default="rate",
                   help="transform of pairwise counts; use 'rate' for per-minute normalization [rate]")
    p.add_argument("--dom-cap", type=float, default=None,
                   help="cap applied AFTER transform, per dyad cell (e.g. wins/min ≤ cap) [none]")
    p.add_argument("--no-heatmap", action="store_true",
                   help="skip W heatmap plot")

    return p.parse_args()

# two shades of pink used across all plots because pink slaps :3
PINK = "#e76ef5"; PINK_DARK = "#b03cde"
FRAME_LIMIT = 50_000


def resolve_params(args: argparse.Namespace) -> Dict[str, float]:
    """Start from species preset and apply CLI overrides."""
    cfg = SPECIES_PRESETS[args.species].copy()
    # apply overrides if provided
    for k_cli, k_cfg in [
        ("move", "move_thr"), ("burst", "burst_thr"), ("fast_thr", "fast_thr"),
        ("contact_mm", "contact_mm"), ("iso_mm", "iso_mm"),
        ("margin_mm", "margin_mm"), ("corner_mm", "corner_mm"),
    ]:
        v = getattr(args, k_cli)
        if v is not None:
            cfg[k_cfg] = float(v)
    return cfg

# ╔═══════════════════════════════════════════════════════════════════════════╗
# 1.  DATA LOADING  (ANIMAL, DETECTION, FRAME)
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_tables(db: Path, debug: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db) as conn:
        animals = pd.read_sql_query("SELECT * FROM ANIMAL", conn)
        detection = pd.read_sql_query("SELECT * FROM DETECTION", conn)
        frames = pd.read_sql_query("SELECT * FROM FRAME", conn)

    detection["ANIMALID"] = pd.to_numeric(detection["ANIMALID"], errors="coerce")

    animals.set_index("ID", inplace=True)
    frames["t"] = pd.to_datetime(frames["TIMESTAMP"], unit="ms")

    if debug:
        max_fr = frames["FRAMENUMBER"].min() + FRAME_LIMIT
        frames = frames[frames["FRAMENUMBER"] <= max_fr]
        detection = detection[detection["FRAMENUMBER"].isin(frames["FRAMENUMBER"])].copy()

    return animals, detection, frames

# ╔═══════════════════════════════════════════════════════════════════════════╗
# 2.  SOCIAL PIPELINE  (fast approaches / escapes / time alone)
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class Contact:
    idA: int
    idB: int
    dist_mm: float

# ---------- helper: find dyads closer than `thresh` mm on one frame ----------

def contacts_from_frame(frame: pd.DataFrame, contact_mm: float) -> list[tuple[int,int,float]]:
    f = frame.loc[:, ["ANIMALID", "MASS_X", "MASS_Y"]].copy()
    f["ANIMALID"] = pd.to_numeric(f["ANIMALID"], errors="coerce")
    f = f[f["ANIMALID"].notna()]
    if len(f) < 2:
        return []
    ids = f["ANIMALID"].astype(np.int64).to_numpy()
    D   = squareform(pdist(f[["MASS_X","MASS_Y"]]))
    i, j = np.where(np.tril(D, k=-1) < contact_mm)
    return [(int(ids[a]), int(ids[b]), float(D[a, b])) for a, b in zip(i, j)]

# ---------- turn per-frame detections into a contact table ----------

def build_contact_table(det_xy: pd.DataFrame, contact_mm: float) -> pd.DataFrame:
    rec = []
    for fr, fdf in det_xy.groupby("FRAMENUMBER"):
        if fdf["ANIMALID"].notna().sum() < 2:
            continue
        rec.extend([(fr, *c) for c in contacts_from_frame(fdf, contact_mm)])
    return pd.DataFrame(rec, columns=["FRAMENUMBER","idA","idB","dist_mm"])

# ---------- compute group sizes per frame (connected components of contact graph) ----------

def _group_sizes_from_contacts(contacts: pd.DataFrame) -> pd.DataFrame:
    rec = []
    for fr, df in contacts.groupby("FRAMENUMBER"):
        # adjacency
        adj: Dict[int, set] = {}
        for a, b in zip(df["idA"].astype(int), df["idB"].astype(int)):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        visited = set()
        for node in list(adj.keys()):
            if node in visited:
                continue
            # DFS to get component
            stack = [node]; comp = []
            visited.add(node)
            while stack:
                n = stack.pop()
                comp.append(n)
                for nb in adj.get(n, ()):  # neighbors
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            size = len(comp)
            for u in comp:
                rec.append((fr, u, size))
    if not rec:
        return pd.DataFrame(columns=["FRAMENUMBER", "ANIMALID", "group_size"], dtype=int)
    return pd.DataFrame(rec, columns=["FRAMENUMBER", "ANIMALID", "group_size"]).astype({"ANIMALID": int, "group_size": int})

# ---------- augment trajectory with speed / dt etc ----------

def build_traj(det: pd.DataFrame, frames: pd.DataFrame, move_thr: float, burst_thr: float) -> pd.DataFrame:
    #add per-frame features required for spatial metrics and social speed
    xy_cols = ["MASS_X", "MASS_Y"]
    t = (
        det.merge(frames[["FRAMENUMBER", "t"]], on="FRAMENUMBER")
        .sort_values(["ANIMALID", "t"])  # align
        .reset_index(drop=True)
    )

    t[["dx", "dy"]] = t.groupby("ANIMALID")[xy_cols].diff()
    t["step_mm"] = np.linalg.norm(t[["dx", "dy"]].values, axis=1)
    t["dt"] = t.groupby("ANIMALID")["t"].diff().dt.total_seconds()
    t["speed"] = t["step_mm"] / t["dt"]
    t.replace([np.inf, -np.inf], np.nan, inplace=True)

    t["speed"] = t["speed"].fillna(0.0)

    t["moving"] = t["speed"] > move_thr
    t["rearing"] = t["DATA"].astype(str).str.contains('isRearing="true"', na=False)
    t["burst"] = False

    # bursts: ≥2 consecutive frames above burst_thr
    for aid, grp in t.groupby("ANIMALID"):
        mask = grp["speed"].gt(burst_thr)
        runs = (mask != mask.shift()).cumsum()
        long_runs = runs[mask].value_counts()[lambda s: s >= 2].index
        t.loc[grp.index, "burst"] = runs.isin(long_runs)

    return t

# ---------- derive social + profile metrics ----------

def social_metrics(
    traj: pd.DataFrame,
    contacts: pd.DataFrame,
    fast_thr: float,
    iso_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # join speeds for idA/idB into the contact table
    merged = (
        contacts
        .merge(
            traj[["FRAMENUMBER", "ANIMALID", "speed"]],
            left_on=["FRAMENUMBER", "idA"],
            right_on=["FRAMENUMBER", "ANIMALID"],
            how="left",
        )
        .rename(columns={"speed": "speed_A"})
        .drop(columns="ANIMALID")
        .merge(
            traj[["FRAMENUMBER", "ANIMALID", "speed"]],
            left_on=["FRAMENUMBER", "idB"],
            right_on=["FRAMENUMBER", "ANIMALID"],
            how="left",
        )
        .rename(columns={"speed": "speed_B"})
        .drop(columns="ANIMALID")
    )

    approaches = merged.query("speed_A > @fast_thr and speed_B < @fast_thr")
    escapes    = merged.query("speed_B > @fast_thr and speed_A < @fast_thr")

    # Isolation: ids with all pairwise distances > iso_mm in a frame
    def isolated_ids(frame: pd.DataFrame, thresh: float) -> np.ndarray:
        ids = frame["ANIMALID"].astype(int).to_numpy()
        if len(ids) == 0:
            return np.array([], dtype=int)
        if len(ids) == 1:
            return ids
        D = squareform(pdist(frame[["MASS_X", "MASS_Y"]]))
        np.fill_diagonal(D, np.inf)
        return ids[(D > thresh).all(axis=1)]

    iso_counts = (
        traj[traj["dt"].notna()]
        .groupby("FRAMENUMBER")
        .apply(lambda fr: pd.Series(isolated_ids(fr, iso_mm)), include_groups=False)
        .droplevel(0)
        .rename("ANIMALID")
        .astype(int)
        .value_counts()
    )

    metrics = pd.DataFrame(index=traj["ANIMALID"].dropna().astype(int).unique())
    metrics["fast_approaches"] = approaches.groupby("idA").size()
    metrics["fast_escapes"]   = escapes.groupby("idB").size()
    metrics["iso_frames"]     = iso_counts
    metrics.fillna(0, inplace=True)
    metrics.index.name = "ANIMALID"

    # per-minute rates
    total_minutes = traj.groupby("ANIMALID")["dt"].sum().div(60.0).replace(0, np.nan)
    metrics["approaches_per_min"] = metrics["fast_approaches"].div(total_minutes).reindex(metrics.index).fillna(0.0)
    metrics["escapes_per_min"]    = metrics["fast_escapes"].div(total_minutes).reindex(metrics.index).fillna(0.0)

    # isolation fractions
    total_frames = traj.groupby("ANIMALID").size()
    alone_frac = metrics["iso_frames"].div(total_frames).reindex(metrics.index).fillna(0.0)
    metrics.drop(columns="iso_frames", inplace=True)
    metrics["pct_time_alone"]  = alone_frac.clip(0, 1)
    metrics["pct_time_social"] = (1.0 - alone_frac).clip(0, 1)

    # ---------- profile-like variables (Contact/Group/move context) ----------
    ic = contacts.melt(id_vars=["FRAMENUMBER"], value_vars=["idA","idB"], value_name="ANIMALID")[
        ["FRAMENUMBER","ANIMALID"]
    ].drop_duplicates()
    ic["in_contact"] = True

    state = (traj[["FRAMENUMBER", "ANIMALID", "dt", "moving"]]
             .merge(ic, on=["FRAMENUMBER", "ANIMALID"], how="left")
             .assign(in_contact=lambda d: d["in_contact"].fillna(False).astype(bool)))

    # group sizes for frames with any contacts
    gs = _group_sizes_from_contacts(contacts)
    state = state.merge(gs, on=["FRAMENUMBER","ANIMALID"], how="left")
    state["group_size"] = state["group_size"].fillna(1).astype(int)

    # detection time & contact time (sec)
    det_time = state.groupby("ANIMALID")["dt"].sum()
    contact_time = state.loc[state["in_contact"], "dt"].groupby(state["ANIMALID"]).sum().reindex(det_time.index).fillna(0.0)

    # contact episodes (rising edges of in_contact)
    def count_rises(s: pd.Series) -> int:
        s = s.astype(bool)
        return int(((s.astype(int).diff() == 1).sum()))
    contact_nb = state.sort_values(["ANIMALID","FRAMENUMBER"]).groupby("ANIMALID")["in_contact"].apply(count_rises)

    # mean duration in sec
    contact_mean_dur = contact_time / contact_nb.replace(0, np.nan)

    # movement context proportions
    move_iso = state.loc[(~state["in_contact"]) & (state["moving"]), "dt"].groupby(state["ANIMALID"]).sum().reindex(det_time.index).fillna(0.0)
    move_ctc = state.loc[(state["in_contact"]) & (state["moving"]), "dt"].groupby(state["ANIMALID"]).sum().reindex(det_time.index).fillna(0.0)

    # group2/3/4 durations + episode counts
    def group_dur_k(k: int) -> pd.Series:
        return state.loc[state["group_size"] == k, "dt"].groupby(state["ANIMALID"]).sum().reindex(det_time.index).fillna(0.0)

    def group_nb_k(k: int) -> pd.Series:
        def count_group_rises(df: pd.DataFrame) -> int:
            s = (df["group_size"] == k).astype(int)
            return int((s.diff() == 1).sum())
        return (
            state.sort_values(["ANIMALID","FRAMENUMBER"])
            .groupby("ANIMALID")
            .apply(count_group_rises, include_groups=False)
        )

    g2_time = group_dur_k(2); g3_time = group_dur_k(3); g4_time = group_dur_k(4)
    g2_nb   = group_nb_k(2);  g3_nb   = group_nb_k(3);  g4_nb   = group_nb_k(4)

    # add to metrics (maintenance + movement context)
    metrics["contact_total_len_s"] = contact_time
    metrics["contact_nb"] = contact_nb
    metrics["contact_mean_dur_s"] = contact_mean_dur
    metrics["contact_prop_time_detection"] = contact_time / det_time

    metrics["move_isolated_prop_time_detection"] = move_iso / det_time
    metrics["move_in_contact_prop_time_detection"] = move_ctc / det_time

    metrics["group2_total_len_s"] = g2_time
    metrics["group2_nb"] = g2_nb
    metrics["group2_mean_dur_s"] = g2_time / g2_nb.replace(0, np.nan)
    metrics["group2_prop_time_detection"] = g2_time / det_time

    metrics["group3_total_len_s"] = g3_time
    metrics["group3_nb"] = g3_nb
    metrics["group3_mean_dur_s"] = g3_time / g3_nb.replace(0, np.nan)
    metrics["group3_prop_time_detection"] = g3_time / det_time

    metrics["group4_total_len_s"] = g4_time
    metrics["group4_nb"] = g4_nb
    metrics["group4_mean_dur_s"] = g4_time / g4_nb.replace(0, np.nan)
    metrics["group4_prop_time_detection"] = g4_time / det_time

    # fast approach/escape normalized by #contacts
    denom = contact_nb.replace(0, np.nan)
    metrics["fast_approach_prop_nb_contact"] = metrics["fast_approaches"] / denom
    metrics["fast_escape_prop_nb_contact"] = metrics["fast_escapes"] / denom

    return metrics.fillna(0.0), approaches, escapes, state

def social_index(metrics: pd.DataFrame, scale: str = "robust", weights: dict | None = None) -> pd.Series:
    default_w = {
        "approaches_per_min": 0.25,
        "escapes_per_min": 0.25,
        "pct_time_social": 0.50,
    }
    w = pd.Series(weights or default_w)
    X = metrics.reindex(columns=w.index)
    X = _scale(X, method=scale)
    return X.mul(w, axis=1).sum(axis=1).rename("social_index")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# 3.  DOMINANCE  Packett–Luce
# ╚═══════════════════════════════════════════════════════════════════════════╝

def dominance_table(approaches: pd.DataFrame, escapes: pd.DataFrame) -> pd.DataFrame:
    A = approaches.groupby(["idA", "idB"]).size().unstack(fill_value=0)
    E = escapes.groupby(["idB", "idA"]).size().unstack(fill_value=0)
    W = (A - E).clip(lower=0)
    return W.fillna(0)
# Weighted pairwise wins with optional transforms (rate, log1p)  + capping

@dataclass
class WinWeights:
    approach: float = 1.0
    escape: float = 1.0

def dominance_pairwise_counts_weighted(
    approaches: pd.DataFrame,
    escapes: pd.DataFrame,
    weights: WinWeights = WinWeights(),
    transform: str | None = None,            # None | "log1p" | "rate"
    exposure_minutes: pd.Series | None = None # per-animal minutes (required for "rate")
) -> pd.DataFrame:
    """
    Build W where W[i,j] = weighted 'wins' of i over j:
      W = weight_approach * #(approach i->j) + weight_escape * #(escape j from i)
    Optional transform:
      - "log1p":   log(1 + W)
      - "rate":    W / dyad_exposure_minutes  (harmonic mean of minutes_i, minutes_j)
    """
    A = approaches.groupby(["idA","idB"]).size().unstack(fill_value=0).astype(float) * weights.approach
    E = escapes.groupby(["idB","idA"]).size().unstack(fill_value=0).astype(float) * weights.escape
    W = A.add(E, fill_value=0.0)

    ids = sorted(set(W.index).union(W.columns))
    W = W.reindex(index=ids, columns=ids, fill_value=0.0)
    np.fill_diagonal(W.values, 0.0)

    if transform == "log1p":
        W = np.log1p(W)

    if transform == "rate":
        if exposure_minutes is None:
            raise ValueError("transform='rate' requires exposure_minutes.")
        ids = list(W.index)
        exp = exposure_minutes.reindex(ids).astype(float)

        # Guard: clamp tiny exposure to avoid huge rates
        MIN_EXP = 0.25  # minutes; adjust if you want
        exp = exp.clip(lower=MIN_EXP)

        # harmonic mean exposure for each dyad (with floor)
        hm = pd.DataFrame(index=ids, columns=ids, dtype=float)
        for i in ids:
            ai = float(exp.loc[i]) if i in exp.index else np.nan
            for j in ids:
                if i == j:
                    hm.loc[i, j] = np.nan
                else:
                    aj = float(exp.loc[j]) if j in exp.index else np.nan
                    if np.isfinite(ai) and np.isfinite(aj):
                        hm.loc[i, j] = 2 * ai * aj / (ai + aj)
                    else:
                        hm.loc[i, j] = np.nan

        # final safety net for any remaining NaNs
        hm = hm.fillna(MIN_EXP)

        W = W.div(hm).fillna(0.0)

    return W



def davids_score(W: pd.DataFrame) -> pd.Series:
    if W.shape[0] == 0:
        return pd.Series(dtype=float, name="dominance_index")
    Wij = W + 1e-9
    Wji = W.T + 1e-9
    P = Wij / (Wij + Wji)
    w = P.sum(axis=1) - P.sum(axis=0)
    w2 = (P @ P.sum(axis=1)) - (P.T @ P.sum(axis=0))
    DS = w + w2
    DS01 = (DS - DS.min())/(DS.max() - DS.min()) if DS.max() > DS.min() else DS*0
    DS01.name = "dominance_index"
    return DS01

def bt_pl_strengths(W: pd.DataFrame, max_iter: int = 500, tol: float = 1e-7, prior: float = 1e-3) -> pd.Series:
    # Bradley–Terry via MM updates on pairwise counts W[i,j] (#wins of i over j)
    if W.empty:
        return pd.Series(dtype=float)
    ids = list(W.index)
    W = W.loc[ids, ids].astype(float)
    N = W + W.T  # total comparisons per dyad

    # init with >=1 to avoid zeros
    s = pd.Series(np.maximum(W.sum(axis=1).values, 1.0), index=ids, dtype=float)
    s /= s.mean()

    for _ in range(max_iter):
        s_old = s.copy()
        num = W.sum(axis=1) + prior
        denom = pd.Series(0.0, index=ids)
        for i in ids:
            denom[i] = np.sum(N.loc[i, ids].values / (s[i] + s[ids].values + 1e-12))
        s = num / np.maximum(denom, 1e-12)
        s /= s.mean()  # fix scale each iter
        if np.max(np.abs(s - s_old) / (s_old + 1e-12)) < tol:
            break
    return s

def plackett_luce_dominance_index(W: pd.DataFrame) -> pd.Series:
    # Return 0–1 dominance index from BT/PL strengths
    if W.empty:
        return pd.Series(dtype=float, name="dominance_index")
    s = bt_pl_strengths(W)
    mn, mx = s.min(), s.max()
    di = (s - mn) / (mx - mn) if mx > mn else s*0.0
    di.name = "dominance_index"
    return di


    # ---------- =relabel IDs (numeric → RFID tail) for *all* outputs ----------

# ╔═══════════════════════════════════════════════════════════════════════════╗
# 4.  SPATIAL PIPELINE
# ╚═══════════════════════════════════════════════════════════════════════════╝

def spatial_metrics(traj: pd.DataFrame, margin: float, corner: float, grid_n: int = 3) -> pd.DataFrame:
    xmin, xmax = traj["MASS_X"].min(), traj["MASS_X"].max()
    ymin, ymax = traj["MASS_Y"].min(), traj["MASS_Y"].max()

    traj["center"] = (
        (traj["MASS_X"] > xmin + margin)
        & (traj["MASS_X"] < xmax - margin)
        & (traj["MASS_Y"] > ymin + margin)
        & (traj["MASS_Y"] < ymax - margin)
    )
    traj["periphery"] = ~traj["center"]

    traj["corner"] = (
        ((traj["MASS_X"] < xmin + corner) | (traj["MASS_X"] > xmax - corner))
        & ((traj["MASS_Y"] < ymin + corner) | (traj["MASS_Y"] > ymax - corner))
    )

    xbins = np.linspace(xmin, xmax, grid_n + 1)
    ybins = np.linspace(ymin, ymax, grid_n + 1)
    X = pd.cut(traj["MASS_X"], bins=xbins, labels=False, include_lowest=True)
    Y = pd.cut(traj["MASS_Y"], bins=ybins, labels=False, include_lowest=True)
    traj["zone"] = (Y * grid_n + X).astype(int)

    m = {}
    g = traj.groupby("ANIMALID")
    m["avg_speed_mm_s"] = g["speed"].mean()
    m["max_speed_mm_s"] = g["speed"].max()
    m["total_distance_mm"] = g["step_mm"].sum()
    m["pct_time_moving"] = g["moving"].mean()
    m["pct_time_center"] = g["center"].mean()
    m["pct_time_periphery"] = g["periphery"].mean()
    m["pct_time_rearing"] = g["rearing"].mean()
    m["n_speed_bursts"] = g["burst"].sum()

    corner_entries = []
    for aid, grp in g:
        ce = (grp["corner"].astype(int).diff() == 1).sum()
        corner_entries.append((aid, ce))
    m["corner_entries"] = pd.Series(dict(corner_entries))

    zone_transitions = []
    for aid, grp in g:
        z = grp["zone"].values
        zone_transitions.append((aid, np.sum(z[1:] != z[:-1])))
    m["zone_transitions"] = pd.Series(dict(zone_transitions))

    roam_ent = []
    for aid, grp in g:
        p = grp["zone"].value_counts(normalize=True).sort_index().values
        roam_ent.append((aid, entropy(p + 1e-12)))
    m["roaming_entropy"] = pd.Series(dict(roam_ent))

    turns = []
    for aid, grp in g:
        dx = grp["dx"].to_numpy()
        dy = grp["dy"].to_numpy()
        mask = np.isfinite(dx) & np.isfinite(dy) & ~((dx == 0) & (dy == 0))
        ang = np.arctan2(dy[mask], dx[mask])
        if ang.size >= 2:
            dtheta = np.diff(np.unwrap(ang))
            val = float(np.nanmean(np.abs(dtheta)))
        else:
            val = np.nan
        turns.append((aid, val))
    m["mean_turn_angle"] = pd.Series(dict(turns))

    M = pd.DataFrame(m).sort_index()
    M.index.name = "ANIMALID"
    return M


def spatial_index(metrics: pd.DataFrame, scale: str = "robust", weights: dict | None = None) -> pd.Series:
    default_w = {
        "avg_speed_mm_s": 0.08,
        "max_speed_mm_s": 0.08,
        "total_distance_mm": 0.12,
        "pct_time_moving": 0.12,
        "pct_time_center": 0.08,
        "pct_time_periphery": 0.08,
        "pct_time_rearing": 0.05,
        "n_speed_bursts": 0.08,
        "corner_entries": 0.05,
        "zone_transitions": 0.08,
        "roaming_entropy": 0.08,
        "mean_turn_angle": 0.08,
    }
    w = pd.Series(weights or default_w)
    X = metrics.reindex(columns=w.index)
    X = _scale(X, method=scale)
    return X.mul(w, axis=1).sum(axis=1).rename("spatial_index")


def _dom_diag(approaches, escapes, det_minutes, W_raw, W_capped):
    print("\n[dominance diagnostics]")
    print("  approaches:", len(approaches), "escapes:", len(escapes))
    if not det_minutes.empty:
        print("  det_minutes (min/median/max):",
              float(det_minutes.min()), float(det_minutes.median()), float(det_minutes.max()))
    if not W_raw.empty:
        od = W_raw.values[~np.eye(len(W_raw), dtype=bool)]
        print("  W_raw offdiag min/median/max:",
              float(np.min(od)), float(np.median(od)), float(np.max(od)))
    if not W_capped.empty:
        od = W_capped.values[~np.eye(len(W_capped), dtype=bool)]
        print("  W_capped offdiag min/median/max:",
              float(np.min(od)), float(np.median(od)), float(np.max(od)))
        uniq = np.unique(np.round(od, 6))
        print("  unique offdiag values:", uniq[:10], "…", f"(count={uniq.size})")

## ╔═══════════════════════════════════════════════════════════════════════════╗
# 5.  plot helpers because pink slaps as i said earlier :3
# ╚═══════════════════════════════════════════════════════════════════════════╝
from math import ceil

PRIMARY   = "#a93a5e"   # rose
MID       = "#a67081"   # mauve
DARK      = "#831b3c"   # wine

# fixed RFID order for plotting (set once in main)
LABEL_ORDER = None  # pd.Index of RFID labels

def _ord_s(s: pd.Series) -> pd.Series:
    return s.reindex(LABEL_ORDER).fillna(0.0) if LABEL_ORDER is not None else s

def _ord_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(LABEL_ORDER).fillna(0.0) if LABEL_ORDER is not None else df

def _ord_square(W: pd.DataFrame) -> pd.DataFrame:
    return (W.reindex(index=LABEL_ORDER, columns=LABEL_ORDER).fillna(0.0)
            if LABEL_ORDER is not None else W)


def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(round(x)))) for x in rgb])

def _blend(c1, c2, a):
    r1,g1,b1 = _hex2rgb(c1); r2,g2,b2 = _hex2rgb(c2)
    return _rgb2hex((r1*(1-a)+r2*a, g1*(1-a)+g2*a, b1*(1-a)+b2*a))

def lighten(c, amt=0.85):  # towards white
    return _blend(c, "#ffffff", amt)

def darken(c, amt=0.20):   # towards black
    return _blend(c, "#000000", amt)

# derived tones
ACCENT     = DARK
BAR_1      = MID
BAR_2      = PRIMARY
BAR_3      = DARK
BLUSH_BG   = lighten(PRIMARY, 0.92)
GRID       = lighten(MID, 0.75)
INK        = darken(DARK, 0.35)

def _set_theme():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": BLUSH_BG,
        "axes.edgecolor": darken(PRIMARY, 0.25),
        "axes.linewidth": 1.1,
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": darken(PRIMARY, 0.25),
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": GRID,
        "grid.alpha": 0.9,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 300,
    })

def _style_axes(ax, y0line=True):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.9)
    ymin, ymax = ax.get_ylim()
    if y0line and (ymin < 0 < ymax):  # zero line only when needed
        ax.axhline(0, color=_blend(PRIMARY, "#000000", 0.25), linewidth=1.0, alpha=0.6, zorder=0)
    ax.tick_params(axis="x", labelrotation=45)

def _cute_bar_effects(ax):
    # white edges + soft shadow for every bar
    for p in ax.patches:
        p.set_edgecolor("white")
        p.set_linewidth(1.0)
        p.set_alpha(0.95)
        p.set_path_effects([pe.withSimplePatchShadow(offset=(0,-1), alpha=.18, shadow_rgbFace="#000000")])

def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_indices_combined(soc_idx, spa_idx, dom_idx, out: Path):
    _set_theme()
    df = pd.concat(
        [soc_idx.rename("Social"), spa_idx.rename("Spatial"), dom_idx.rename("Dominance")],
        axis=1, join="outer"
    ).fillna(0.0)
    df = _ord_df(df)  # <— enforce fixed RFID order

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    df.plot(kind="bar", ax=ax, color=ACCENTS, width=0.82, edgecolor="white", linewidth=0.8)
    _cute_bar_effects(ax)

    ax.set_title("Social, Spatial, Dominance per animal")
    ax.set_ylabel("Index")
    ax.legend(frameon=False, ncol=3)
    _style_axes(ax, y0line=True)
    _save(fig, out / "indices_combined.png")


def plot_scatter_social_vs_spatial(soc_idx: pd.Series, spa_idx: pd.Series, out: Path):
    _set_theme()
    # outer-join, keep everyone, fill gaps with 0
    xy = pd.concat(
        [soc_idx.rename("Social"), spa_idx.rename("Spatial")],
        axis=1, join="outer"
    ).fillna(0.0).sort_index()

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    # soft glow under points
    ax.scatter(xy["Social"], xy["Spatial"], s=240, c="#000000", alpha=0.06)
    # cute dots with white stroke
    ax.scatter(xy["Social"], xy["Spatial"], s=140, c=PRIMARY, edgecolors="white", linewidths=1.4)

    # tiny labels on top of points
    for lab, row in xy.iterrows():
        ax.text(row["Social"], row["Spatial"], str(lab),
                color=INK, fontsize=8, ha="center", va="center", weight="bold",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white", alpha=0.8)])

    ax.set_xlabel("Social index")
    ax.set_ylabel("Spatial index")
    ax.set_title("Social vs Spatial")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style_axes(ax, y0line=False)
    _save(fig, out / "social_vs_spatial.png")


def plot_bars_simple(series: pd.Series, title: str, fname: Path, ylim=None, ylabel=""):
    _set_theme()
    s = _ord_s(series)  # <— keep fixed order; no sort by value
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(s.index.astype(str), s.values, color=PRIMARY, edgecolor="white", linewidth=0.9, width=0.82)
    _cute_bar_effects(ax)
    ax.set_title(title); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    _style_axes(ax, y0line=True)
    _save(fig, fname)

def plot_W_heatmap(W: pd.DataFrame, out_path: Path, title: str = "Pairwise wins (row beats column)"):
    _set_theme()
    W = _ord_square(W)  # <— rows & cols in fixed RFID order
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(W.values, aspect="auto", cmap=ROSE_CMAP)
    ax.set_title(title)
    ax.set_xticks(range(len(W.columns))); ax.set_xticklabels(W.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(W.index)));   ax.set_yticklabels(W.index, fontsize=8)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("value", rotation=90)
    _save(fig, out_path)

# ---------- category multipanels (small multiples) ----------
def plot_category_multipanel(df: pd.DataFrame, metrics: list[str], display_names: dict[str,str],
                             title: str, out_path: Path, ncols: int = 3):
    from math import ceil
    _set_theme()
    df = _ord_df(df)  # <— enforce fixed order

    n = len(metrics); ncols = min(ncols, n); nrows = ceil(n/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols+1.6, 3.0*nrows+1.4), squeeze=False)
    cols = [MID, PRIMARY, DARK, _blend(MID, "#000000", .1)]

    for k, metric in enumerate(metrics):
        r,c = divmod(k, ncols); ax = axes[r][c]
        s = df[metric]  # <— no sort by values
        ax.bar(s.index.astype(str), s.values, color=cols[k % len(cols)], edgecolor="white", linewidth=0.8, width=0.82)
        _cute_bar_effects(ax)
        ax.set_title(display_names.get(metric, metric), pad=8)
        ax.set_ylabel("")
        _style_axes(ax, y0line=True)

    for k in range(n, nrows*ncols):
        r,c = divmod(k, ncols); axes[r][c].set_visible(False)

    fig.suptitle(title, x=0.02, ha="left", color=_blend(PRIMARY,"#000000",0.25), fontsize=13, fontweight="bold")
    _save(fig, out_path)

# brand accents for grouped bars
ACCENTS = [MID, PRIMARY, DARK]

# cozy rose colormap for heatmaps
ROSE_CMAP = LinearSegmentedColormap.from_list(
    "rose",
    [lighten(PRIMARY, 0.96), lighten(MID, 0.70), PRIMARY, DARK],
    N=256,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# 6.  main (idk what else to say, it's just main)
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    args = cli()
    cfg = resolve_params(args)
    t0 = time.time()

    db = Path(args.sqlite).expanduser()
    out = Path(args.out) if args.out else db.parent
    out.mkdir(parents=True, exist_ok=True)

    # optional weights
    weights = None
    if args.weights_json:
        with open(args.weights_json) as f:
            weights = json.load(f)

    # load tables
    animals, detection, frames = load_tables(db, args.debug)

    # build trajectory ONCE (gives speed/dt etc.)
    traj = build_traj(detection, frames, cfg["move_thr"], cfg["burst_thr"])

    # ---- map numeric ID → RFID tail (for relabeling outputs) ----
    tails = animals["RFID"].astype(str).str[-4:]
    try:
        tails.index = tails.index.astype(int)
    except Exception:
        pass

    def map_id(i: int) -> str:
        """Return RFID tail if available; otherwise the numeric ID as a string (no 'id' prefix)."""
        try:
            return str(tails.loc[int(i)])
        except Exception:
            return str(int(i))

    # alist of all labels present in the traj (after relabeling)
    animal_ids_all = sorted(traj["ANIMALID"].dropna().astype(int).unique())
    labels_all = pd.Index([map_id(i) for i in animal_ids_all], name="RFID")

    def _read_label_order_file(path: Path) -> list[str] | None:
        if path.exists():
            return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        return None

    def _numeric_ascending(idx: pd.Index) -> pd.Index:
        try:
            return pd.Index(sorted(idx, key=lambda x: int(str(x))))
        except Exception:
            return idx.sort_values()

    order_file = out / "label_order.txt"
    from_file = _read_label_order_file(order_file)

    if from_file:
        base = [x for x in from_file if x in set(labels_all)]
        missing = [x for x in labels_all if x not in base]
        LABELS = pd.Index(base + list(_numeric_ascending(pd.Index(missing))), name="RFID")
    else:
        LABELS = _numeric_ascending(labels_all).rename("RFID")
        try:
            order_file.write_text("\n".join(map(str, LABELS)))
        except Exception:
            print(f"[warn] could not write {order_file}; order will be in-memory only")

    global LABEL_ORDER
    LABEL_ORDER = LABELS


    #this is to reindex any series/dataFrame to that full list and fill 0
    def pad_to_all_labels(obj):
        if isinstance(obj, pd.Series):
            return obj.reindex(labels_all).fillna(0.0)
        elif isinstance(obj, pd.DataFrame):
            return obj.reindex(labels_all).fillna(0.0)
        else:
            return obj

    # contacts + social metrics (+ profile-like state table)
    contacts = build_contact_table(traj, cfg["contact_mm"])

    # per-animal detection exposure in minutes (for rate normalization)
    det_minutes = (
        traj.loc[traj["ANIMALID"].notna()]
            .groupby("ANIMALID")["dt"].sum()
            .div(60.0)
    )

    soc_m, approaches, escapes, state = social_metrics(
        traj, contacts, cfg["fast_thr"], cfg["iso_mm"]
    )
    soc_idx = social_index(
        soc_m, scale=args.social_scale, weights=(weights.get("social") if weights else None)
    )

    # ---------- dominance (rate-normalized + cap, then estimator) ----------
    W_raw = dominance_pairwise_counts_weighted(
        approaches,
        escapes,
        weights=WinWeights(args.approach_weight, args.escape_weight),
        transform=args.dom_transform,  # "rate" by default per CLI
        exposure_minutes=(det_minutes if args.dom_transform == "rate" else None)
    )

    # apply cap after transform
    W_capped = W_raw if args.dom_cap is None else W_raw.clip(upper=float(args.dom_cap))
    _dom_diag(approaches, escapes, det_minutes, W_raw, W_capped)

    # warn if W is (near) uniform off-diagonal -> dominance will be flat
    offdiag = W_capped.values[~np.eye(W_capped.shape[0], dtype=bool)]
    if offdiag.size > 0 and np.allclose(offdiag, offdiag[0], rtol=1e-6, atol=1e-8):
        print("[warn] Pairwise W is uniform after transform/cap; dominance index will be flat. "
              "Try --dom-cap None (or larger), --dom-transform log1p, or increase --fast-thr.")

    # choose estimator
    if args.dominance == "davids":
        dom_idx = davids_score(W_capped)
        dom_title = "Dominance index (David’s Score)"
    else:
        dom_idx = plackett_luce_dominance_index(W_capped)
        dom_title = "Dominance index (Plackett–Luce)"

    # reindex labels to RFID tails for outputs
    W_out = W_capped.copy()
    W_out.index = W_out.index.map(map_id)
    W_out.columns = W_out.columns.map(map_id)
    W_out.to_csv(out / "dominance_pairwise_W.csv")

    if not args.no_heatmap:
        plot_W_heatmap(W_out, out / "dominance_pairwise_W_heatmap.png")

    # save dom index csv + plot
    dom_idx.index = dom_idx.index.map(map_id)
    dom_idx.to_csv(out / "dominance_index.csv")
    plot_bars_simple(dom_idx, dom_title, out / "dominance_bar.png", ylim=(0, 1), ylabel="0–1")

    # ---------- spatial metrics & index ----------
    spa_m = spatial_metrics(traj, margin=cfg["margin_mm"], corner=cfg["corner_mm"])
    spa_idx = spatial_index(
        spa_m, scale=args.social_scale, weights=(weights.get("spatial") if weights else None)
    )

    # relabel to RFID tails + ensure every animal appears (fill missing with 0)
    for df in (soc_m, spa_m):
        df.index = df.index.map(map_id)
        df.index.name = "RFID"

    soc_idx.index = soc_idx.index.map(map_id)
    spa_idx.index = spa_idx.index.map(map_id)
    dom_idx.index = dom_idx.index.map(map_id)

    # pad Series/DataFrames so plots & CSVs include ALL animals (zeros where missing)
    soc_idx = pad_to_all_labels(soc_idx)
    spa_idx = pad_to_all_labels(spa_idx)
    dom_idx = pad_to_all_labels(dom_idx)
    soc_m   = pad_to_all_labels(soc_m)
    spa_m   = pad_to_all_labels(spa_m)

    # ---------- relabel IDs (numeric → RFID tail) for *all* outputs ----------
    for df in (soc_m, spa_m):
        df.index = df.index.map(map_id)
        df.index.name = "RFID"

    soc_idx.index = soc_idx.index.map(map_id)
    spa_idx.index = spa_idx.index.map(map_id)
    dom_idx.index = dom_idx.index.map(map_id)

    # ---------- save csvs for metrics/indices ----------
    soc_m.to_csv(out / "social_metrics.csv")
    spa_m.to_csv(out / "spatial_metrics.csv")
    soc_idx.to_csv(out / "social_index.csv")
    spa_idx.to_csv(out / "spatial_index.csv")
    dom_idx.to_csv(out / "dominance_index.csv")
    pd.concat([soc_idx, spa_idx, dom_idx], axis=1).to_csv(out / "social_spatial_dom_index.csv")

    # ---------- profile variables ----------
    profile = pd.DataFrame(index=soc_m.index)  # already relabeled to RFID4
    profile["totalDistance"] = spa_m["total_distance_mm"].div(1000.0)  # meters
    profile["Move isolated PropTimeDetection"] = soc_m["move_isolated_prop_time_detection"]
    profile["Move in contact PropTimeDetection"] = soc_m["move_in_contact_prop_time_detection"]
    profile["Contact TotalLen"] = soc_m["contact_total_len_s"]
    profile["Contact Nb"] = soc_m["contact_nb"]
    profile["Contact MeanDur"] = soc_m["contact_mean_dur_s"]
    profile["Contact PropTimeDetection"] = soc_m["contact_prop_time_detection"]
    profile["Group2 TotalLen"] = soc_m["group2_total_len_s"]
    profile["Group2 Nb"] = soc_m["group2_nb"]
    profile["Group2 MeanDur"] = soc_m["group2_mean_dur_s"]
    profile["Group2 PropTimeDetection"] = soc_m["group2_prop_time_detection"]
    profile["Group3 TotalLen"] = soc_m["group3_total_len_s"]
    profile["Group3 Nb"] = soc_m["group3_nb"]
    profile["Group3 MeanDur"] = soc_m["group3_mean_dur_s"]
    profile["Group3 PropTimeDetection"] = soc_m["group3_prop_time_detection"]
    profile["Group4 TotalLen"] = soc_m["group4_total_len_s"]
    profile["Group4 Nb"] = soc_m["group4_nb"]
    profile["Group4 MeanDur"] = soc_m["group4_mean_dur_s"]
    profile["Group4 PropTimeDetection"] = soc_m["group4_prop_time_detection"]
    profile["Fast approach contact PropNbContact"] = soc_m["fast_approach_prop_nb_contact"]
    profile["Fast escape contact PropNbContact"] = soc_m["fast_escape_prop_nb_contact"]
    profile.to_csv(out / "profile_variables.csv")

    # ---------- Summary plots ----------
    plot_indices_combined(soc_idx, spa_idx, dom_idx, out)
    plot_scatter_social_vs_spatial(soc_idx, spa_idx, out)

    plot_bars_simple(dom_idx, dom_title, out / "dominance_bar.png", ylabel="index")
    plot_bars_simple(soc_idx, "Social index", out / "social_index_bar.png", ylabel="index")
    plot_bars_simple(spa_idx, "Spatial index", out / "spatial_index_bar.png", ylabel="index")

    # category figures (inputs)
    social_inputs = ["approaches_per_min", "escapes_per_min", "pct_time_social"]
    social_names = {
        "approaches_per_min": "Fast approaches / min",
        "escapes_per_min": "Fast escapes / min",
        "pct_time_social": "% time social",
    }
    plot_category_multipanel(soc_m, social_inputs, social_names,
                             "Social index — component metrics", out / "social_components.png")

    locomotion = ["avg_speed_mm_s", "max_speed_mm_s", "total_distance_mm", "pct_time_moving", "n_speed_bursts"]
    loc_names = {
        "avg_speed_mm_s": "Mean speed (mm/s)",
        "max_speed_mm_s": "Max speed (mm/s)",
        "total_distance_mm": "Total distance (mm)",
        "pct_time_moving": "% time moving",
        "n_speed_bursts": "# speed bursts",
    }
    plot_category_multipanel(spa_m, locomotion, loc_names,
                             "Spatial — Locomotion", out / "spatial_locomotion.png")  # <-- fixed filename

    arena_use = ["pct_time_center", "pct_time_periphery", "corner_entries"]
    arena_names = {
        "pct_time_center": "% time center",
        "pct_time_periphery": "% time periphery",
        "corner_entries": "Corner entries",
    }
    plot_category_multipanel(spa_m, arena_use, arena_names,
                             "Spatial — Arena use", out / "spatial_arena.png")

    strategy = ["zone_transitions", "roaming_entropy", "mean_turn_angle"]
    strat_names = {
        "zone_transitions": "Zone transitions",
        "roaming_entropy": "Roaming entropy",
        "mean_turn_angle": "Mean turn angle",
    }
    plot_category_multipanel(spa_m, strategy, strat_names,
                             "Spatial — Exploration strategy", out / "spatial_strategy.png")

    vertical = ["pct_time_rearing"]
    vert_names = {"pct_time_rearing": "% time rearing"}
    plot_category_multipanel(spa_m, vertical, vert_names,
                             "Spatial — Vertical activity", out / "spatial_vertical.png")

    # SOCIAL inputs (the features that *compose* the social index)
    social_inputs = ["approaches_per_min", "escapes_per_min", "pct_time_social"]
    social_names  = {
        "approaches_per_min": "Fast approaches / min",
        "escapes_per_min":    "Fast escapes / min",
        "pct_time_social":    "% time social",
    }
    plot_category_multipanel(soc_m, social_inputs, social_names,
                             "Social index — component metrics", out / "social_components.png")

    # SPATIAL inputs grouped into readable categories
    locomotion = ["avg_speed_mm_s","max_speed_mm_s","total_distance_mm","pct_time_moving","n_speed_bursts"]
    loc_names = {
        "avg_speed_mm_s": "Mean speed (mm/s)",
        "max_speed_mm_s": "Max speed (mm/s)",
        "total_distance_mm": "Total distance (mm)",
        "pct_time_moving": "% time moving",
        "n_speed_bursts": "# speed bursts",
    }
    plot_category_multipanel(spa_m, locomotion, loc_names,
                             "Spatial — Locomotion", out / "spatial_locmotion.png")

    arena_use = ["pct_time_center","pct_time_periphery","corner_entries"]
    arena_names = {
        "pct_time_center": "% time center",
        "pct_time_periphery": "% time periphery",
        "corner_entries": "Corner entries",
    }
    plot_category_multipanel(spa_m, arena_use, arena_names,
                             "Spatial — Arena use", out / "spatial_arena.png")

    strategy = ["zone_transitions","roaming_entropy","mean_turn_angle"]
    strat_names = {
        "zone_transitions": "Zone transitions",
        "roaming_entropy":  "Roaming entropy",
        "mean_turn_angle":  "Mean turn angle",
    }
    plot_category_multipanel(spa_m, strategy, strat_names,
                             "Spatial — Exploration strategy", out / "spatial_strategy.png")

    vertical = ["pct_time_rearing"]
    vert_names = {"pct_time_rearing": "% time rearing"}
    plot_category_multipanel(spa_m, vertical, vert_names,
                             "Spatial — Vertical activity", out / "spatial_vertical.png")


    # friendly summary of parameters
    print(f"\nPreset: {args.species}")
    for k in ["contact_mm","iso_mm","move_thr","burst_thr","fast_thr","margin_mm","corner_mm"]:
        print(f"  {k:>11}: {cfg[k]}")
    print(f"\n🎉 All done in {time.time() - t0:.1f}s →  {out}\n")


if __name__ == "__main__":
    main()
#!/usr/bin/env python