import numpy as np
import pandas as pd
import polars as pl
from sklearn.mixture import GaussianMixture

def _as_polars(df) -> pl.DataFrame:
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)

def _dist_expr(x1: str, y1: str, x2: str, y2: str) -> pl.Expr:
    return (((pl.col(x1) - pl.col(x2)) ** 2 + (pl.col(y1) - pl.col(y2)) ** 2).sqrt())

# --- SPEED THRESHOLDS (tiny guard; unchanged logic otherwise) ---
def auto_thresholds_from_speed(traj):
    s = traj["speed"].to_numpy(dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size < 50:
        return 2.0, 20.0, 20.0, {"method":"quantile-fallback","n":int(s.size)}
    x = np.log10(s).reshape(-1,1)
    gmm = GaussianMixture(n_components=3, random_state=0).fit(x)
    mu = np.sort(gmm.means_.ravel())
    move_thr  = 10 ** np.sqrt(mu[0]*mu[1])
    burst_thr = 10 ** np.sqrt(mu[1]*mu[2])
    return float(move_thr), float(burst_thr), float(burst_thr), {"method":"logGMM","n":int(s.size)}

# --- CONTACT/ISO: robust against zeros/inf + safe fallbacks ---
def auto_contact_iso_from_NN(traj, workers: int = 1):
    _ = workers
    det = (
        _as_polars(traj).select(["FRAMENUMBER", "ANIMALID", "MASS_X", "MASS_Y"])
        .with_columns([
            pl.col("FRAMENUMBER").cast(pl.Int64, strict=False),
            pl.col("ANIMALID").cast(pl.Int64, strict=False),
            pl.col("MASS_X").cast(pl.Float64, strict=False),
            pl.col("MASS_Y").cast(pl.Float64, strict=False),
        ])
        .drop_nulls()
    )
    if det.is_empty():
        nn = np.array([], dtype=float)
    else:
        nn_df = (
            det.join(
                det.select([
                    "FRAMENUMBER",
                    pl.col("ANIMALID").alias("ANIMALID_B"),
                    pl.col("MASS_X").alias("MASS_X_B"),
                    pl.col("MASS_Y").alias("MASS_Y_B"),
                ]),
                on="FRAMENUMBER",
                how="inner",
            )
            .filter(pl.col("ANIMALID") != pl.col("ANIMALID_B"))
            .with_columns(_dist_expr("MASS_X", "MASS_Y", "MASS_X_B", "MASS_Y_B").alias("dist_mm"))
            .group_by(["FRAMENUMBER", "ANIMALID"])
            .agg(pl.col("dist_mm").min().alias("nn_mm"))
        )
        nn = nn_df["nn_mm"].to_numpy()

    # clean: finite & positive only
    nn = nn[np.isfinite(nn)]
    nn = nn[nn > 0]

    # if no data or too few samples → quantile fallback
    if nn.size < 200:
        if nn.size == 0:
            contact = 30.0
            iso = 50.0
            return contact, iso, {"method":"no-data-fallback","n":0}
        contact = float(np.nanpercentile(nn, 20))
        iso = float(np.nanpercentile(nn, 88))
        iso = max(iso, 1.6 * contact)
        return contact, iso, {"method":"quantile-fallback","n":int(nn.size)}

    # guard zeros before log10 (should be none, but be extra safe)
    eps = 1e-6
    nn = np.clip(nn, eps, None)
    x = np.log10(nn).reshape(-1,1)

    try:
        gmm = GaussianMixture(n_components=2, random_state=0).fit(x)
        mus = np.sort(gmm.means_.ravel())
        centers = 10 ** mus
        contact = float(np.sqrt(centers[0] * centers[1]))
        iso = float(max(np.nanpercentile(nn, 88), 1.6 * contact))
        return contact, iso, {"method":"NN-logGMM","centers_mm":centers.tolist(),"n":int(nn.size)}
    except Exception:
        # last-resort fallback if GMM can’t fit (singular data, etc.)
        contact = float(np.nanpercentile(nn, 20))
        iso = float(max(np.nanpercentile(nn, 88), 1.6 * contact))
        return contact, iso, {"method":"gmm-fallback","n":int(nn.size)}

# --- BURST RUN-LEN ---
def infer_burst_run_length(traj, min_burst_dur=0.08):
    dt = traj["dt"].dropna()
    if dt.empty: return 2
    return max(2, int(np.ceil(min_burst_dur / dt.median())))
