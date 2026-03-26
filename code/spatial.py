# spatial.py
from __future__ import annotations
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import entropy

# if you have metrics_utils.scale_df, use it. otherwise replace with your own.
try:
    from metrics_utils import scale_df
except Exception:
    # minimal local fallback
    from sklearn.preprocessing import RobustScaler
    def scale_df(df: pd.DataFrame, method: str = "robust") -> pd.DataFrame:
        scaler = RobustScaler()  # keep it simple; swap if you need minmax/zscore
        arr = scaler.fit_transform(df)
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

__all__ = ["spatial_metrics_polars", "spatial_index"]

def spatial_metrics_polars(
    traj_pd: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    grid_n: int = 3
) -> pd.DataFrame:
    """
    Compute spatial metrics on a 3×3 grid using Polars for the heavy groupby parts.
    Expects traj_pd to contain: ANIMALID, FRAMENUMBER, MASS_X, MASS_Y, speed, step_mm, moving, rearing, burst, dx, dy.
    """
    assert grid_n == 3
    if isinstance(traj_pd, pl.LazyFrame):
        t_lf = traj_pd
    elif isinstance(traj_pd, pl.DataFrame):
        t_lf = traj_pd.lazy()
    else:
        if len(traj_pd) == 0:
            return pd.DataFrame(index=pd.Index([], name="ANIMALID"))
        t_lf = pl.from_pandas(traj_pd).lazy()

    if t_lf.select(pl.len().alias("n")).collect().item() == 0:
        return pd.DataFrame(index=pd.Index([], name="ANIMALID"))

    # arena bounds
    xmin, xmax, ymin, ymax = (
        t_lf.select(
            pl.col("MASS_X").min().alias("xmin"),
            pl.col("MASS_X").max().alias("xmax"),
            pl.col("MASS_Y").min().alias("ymin"),
            pl.col("MASS_Y").max().alias("ymax"),
        ).collect().row(0)
    )

    # bin to 0..2 with floor; guard zero-range
    def idx(col, lo, hi):
        span = max(hi - lo, 1e-9)
        return (((pl.col(col) - lo) / span) * grid_n).floor().clip(0, grid_n - 1).cast(pl.Int32)

    t_lf = (
        t_lf.with_columns([
            idx("MASS_X", xmin, xmax).alias("zone_x"),
            idx("MASS_Y", ymin, ymax).alias("zone_y"),
        ])
        .with_columns([
            (pl.col("zone_y") * grid_n + pl.col("zone_x")).alias("zone"),
            ((pl.col("zone_x") == 1) & (pl.col("zone_y") == 1)).alias("center"),
        ])
        .with_columns([
            (~pl.col("center")).alias("periphery"),
            pl.col("zone").is_in([0, 2, 6, 8]).alias("corner"),
        ])
    )

    # simple aggregations
    base = (
        t_lf.group_by("ANIMALID", maintain_order=True)
         .agg([
            pl.col("speed").mean().alias("avg_speed_mm_s"),
            pl.col("speed").max().alias("max_speed_mm_s"),
            pl.col("step_mm").sum().alias("total_distance_mm"),
            pl.col("moving").mean().alias("pct_time_moving"),
            pl.col("center").mean().alias("pct_time_center"),
            pl.col("periphery").mean().alias("pct_time_periphery"),
            pl.col("rearing").mean().alias("pct_time_rearing"),
            pl.col("burst").sum().alias("n_speed_bursts"),
         ])
         .collect()
         .to_pandas()
         .set_index("ANIMALID")
    )

    # order-dependent metrics: need FRAMENUMBER to sort within each animal
    keep = (
        t_lf.select(["ANIMALID", "FRAMENUMBER", "zone", "corner", "dx", "dy"])
        .collect()
        .to_pandas()
    )

    # corner entries & zone transitions
    corner_entries = {}
    zone_transitions = {}

    # mean turn angle (abs change of unwrapped heading)
    mean_turn_angle = {}

    for aid, grp in keep.groupby("ANIMALID", sort=False):
        g = grp.sort_values("FRAMENUMBER").reset_index(drop=True)

        # corner rising edges
        ce = (g["corner"].astype(int).diff() == 1).sum()
        corner_entries[aid] = int(ce)

        # zone transitions
        z = g["zone"].to_numpy()
        zone_transitions[aid] = int(np.sum(z[1:] != z[:-1])) if len(z) > 1 else 0

        # mean turn angle
        dx = g["dx"].to_numpy()
        dy = g["dy"].to_numpy()
        mask = np.isfinite(dx) & np.isfinite(dy) & ~((dx == 0) & (dy == 0))
        if mask.sum() >= 2:
            ang = np.arctan2(dy[mask], dx[mask])
            mean_turn_angle[aid] = float(np.nanmean(np.abs(np.diff(np.unwrap(ang)))))
        else:
            mean_turn_angle[aid] = np.nan

    base["corner_entries"] = pd.Series(corner_entries)
    base["zone_transitions"] = pd.Series(zone_transitions)
    base["mean_turn_angle"] = pd.Series(mean_turn_angle)

    # roaming entropy (9-bin occupancy)
    roam = {}
    for aid, grp in keep.groupby("ANIMALID", sort=False):
        p = (
            grp["zone"]
            .value_counts(normalize=True)
            .sort_index()
            .reindex(range(9), fill_value=0)
            .values
        )
        roam[aid] = float(entropy(p + 1e-12))
    base["roaming_entropy"] = pd.Series(roam)

    base.index.name = "ANIMALID"
    return base.sort_index()

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
    X = scale_df(X, method=scale)
    return X.mul(w, axis=1).sum(axis=1).rename("spatial_index")
