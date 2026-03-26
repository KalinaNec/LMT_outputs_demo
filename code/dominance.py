# dominance_fast.py
from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------- build W via Polars (fast) ----------
def _wins_matrix_polars(approaches_pd, escapes_pd, w_app: float = 1.0, w_esc: float = 1.0) -> pd.DataFrame:
    def _weighted_matrix(
        df_pd,
        winner_col: str,
        loser_col: str,
        w: float,
    ) -> pd.DataFrame:
        if df_pd is None or len(df_pd) == 0:
            return pd.DataFrame()
        w = float(w)
        if w == 0:
            return pd.DataFrame()

        # Signed weight convention:
        # positive => keep winner/loser direction
        # negative => flip direction and use abs(weight)
        src_w, src_l = winner_col, loser_col
        ww = abs(w)
        if w < 0:
            src_w, src_l = src_l, src_w

        return (
            pl.from_pandas(df_pd)
            .group_by([src_w, src_l])
            .count()
            .rename({src_w: "idA", src_l: "idB"})
            .with_columns([
                (pl.col("count") * ww).alias("w"),
                pl.col("idA").cast(pl.Int64),
                pl.col("idB").cast(pl.Int64),
            ])
            .select(["idA", "idB", "w"])
            .pivot(values="w", index="idA", columns="idB", aggregate_function="first")
            .fill_null(0.0)
            .to_pandas()
            .set_index("idA", drop=True)
            .rename_axis(index=None, columns=None)
        )

    # approaches: canonical winner/loser = idA over idB
    A_pd = _weighted_matrix(approaches_pd, "idA", "idB", w_app)

    # escapes: canonical winner/loser = idB over idA (escaper over initiator)
    E_pd = _weighted_matrix(escapes_pd, "idB", "idA", w_esc)

    if A_pd.empty and E_pd.empty:
        return pd.DataFrame()

    # combine A and E into total wins
    W = A_pd.add(E_pd, fill_value=0.0)

    # --- robust label normalization & sorting ---
    def _try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    def _sort_key(x):
        # numbers first (ascending), then strings (alphabetical)
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))

    W.index = W.index.map(_try_int)
    W.columns = W.columns.map(_try_int)

    ids = sorted(set(list(W.index) + list(W.columns)), key=_sort_key)
    W = W.reindex(index=ids, columns=ids, fill_value=0.0)

    np.fill_diagonal(W.values, 0.0)
    return W

@dataclass
class WinWeights:
    approach: float = 1.0
    # Signed convention: positive keeps escaper as winner, negative flips to initiator as winner.
    escape: float = -0.5


# ---------- optional transforms (log1p / rate) ----------
def _apply_transform(
    W: pd.DataFrame,
    transform: Optional[str],
    exposure_minutes: Optional[pd.Series],
    min_exp: float = 0.25,
) -> pd.DataFrame:
    if W.empty or transform is None:
        return W

    if transform == "log1p":
        return np.log1p(W).fillna(0.0)

    if transform == "rate":
        if exposure_minutes is None:
            raise ValueError("transform='rate' requires exposure_minutes.")
        ids = list(W.index)
        e = exposure_minutes.reindex(ids).astype(float).to_numpy()
        e = np.clip(e, min_exp, None)

        # harmonic mean matrix, fully vectorized
        HM = (2.0 * e[:, None] * e[None, :]) / (e[:, None] + e[None, :])
        np.fill_diagonal(HM, np.nan)  # never used
        Wn = W.to_numpy(dtype=float, copy=True)
        out = Wn / HM
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = pd.DataFrame(out, index=ids, columns=ids)
        return out.fillna(0.0)

    return W


# ---------- BT / Plackett–Luce via vectorized MM ----------
def _bt_strengths_mm(W: pd.DataFrame, max_iter: int = 500, tol: float = 1e-7, prior: float = 1e-3) -> pd.Series:
    if W.empty:
        return pd.Series(dtype=float)

    ids = list(W.index)
    M = W.loc[ids, ids].to_numpy(dtype=float, copy=True)  # wins i over j
    N = M + M.T                                          # total comparisons
    # init strengths (avoid zeros), scale to mean 1
    s = np.maximum(M.sum(axis=1), 1.0)
    s = s / s.mean()

    for _ in range(max_iter):
        s_old = s
        num = M.sum(axis=1) + prior
        denom = np.sum(N / (s[:, None] + s[None, :] + 1e-12), axis=1)
        s = num / np.maximum(denom, 1e-12)
        s = s / s.mean()
        if np.max(np.abs(s - s_old) / (s_old + 1e-12)) < tol:
            break

    return pd.Series(s, index=ids, dtype=float)


def plackett_luce_dominance_index(W: pd.DataFrame) -> pd.Series:
    """Return 0–1 scaled PL index from W (wins)."""
    if W.empty:
        return pd.Series(dtype=float, name="dominance_index")
    s = _bt_strengths_mm(W)
    mn, mx = s.min(), s.max()
    di = (s - mn) / (mx - mn) if mx > mn else s * 0.0
    di.name = "dominance_index"
    return di


# ---------- one-shot API used by main ----------
def compute_pl_index(
    approaches: pd.DataFrame,
    escapes: pd.DataFrame,
    weights: WinWeights,
    transform: Optional[str],
    exposure_minutes: Optional[pd.Series],
    cap: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build W with Polars, apply transform/rate, optional cap, return (W_raw, W_capped, PL_index).
    W_raw includes the chosen transform (if any). W_capped is just W_raw with upper cap.
    """
    W = _wins_matrix_polars(approaches, escapes, w_app=weights.approach, w_esc=weights.escape)
    if W.empty:
        return W, W, pd.Series(dtype=float, name="dominance_index")

    W_raw = _apply_transform(W, transform=transform, exposure_minutes=exposure_minutes)
    W_capped = W_raw if cap is None else W_raw.clip(upper=float(cap))

    di = plackett_luce_dominance_index(W_capped)
    return W_raw, W_capped, di
