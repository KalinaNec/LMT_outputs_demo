# social.py
from __future__ import annotations
from typing import Dict
import pandas as pd
import polars as pl

# public API
__all__ = [
    "build_contact_table",
    "build_contact_table_parallel",
    "social_metrics",
    "social_index",
]

def _to_polars(df: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)

def _dist_expr(x1: str, y1: str, x2: str, y2: str) -> pl.Expr:
    return (((pl.col(x1) - pl.col(x2)) ** 2 + (pl.col(y1) - pl.col(y2)) ** 2).sqrt())

def _safe_ratio(num: str, den: str, out: str) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(0.0).alias(out)

def _contacts_polars(det_xy: pd.DataFrame | pl.DataFrame | pl.LazyFrame, contact_mm: float) -> pl.DataFrame:
    d = (
        _to_polars(det_xy)
        .select(["FRAMENUMBER", "ANIMALID", "MASS_X", "MASS_Y"])
        .with_columns([
            pl.col("FRAMENUMBER").cast(pl.Int64, strict=False),
            pl.col("ANIMALID").cast(pl.Int64, strict=False),
            pl.col("MASS_X").cast(pl.Float64, strict=False),
            pl.col("MASS_Y").cast(pl.Float64, strict=False),
        ])
        .drop_nulls()
    )
    if d.is_empty():
        return pl.DataFrame(
            schema={
                "FRAMENUMBER": pl.Int64,
                "idA": pl.Int64,
                "idB": pl.Int64,
                "dist_mm": pl.Float64,
            }
        )

    pairs = (
        d.join(d, on="FRAMENUMBER", suffix="_B")
        .filter(pl.col("ANIMALID") < pl.col("ANIMALID_B"))
        .with_columns(_dist_expr("MASS_X", "MASS_Y", "MASS_X_B", "MASS_Y_B").alias("dist_mm"))
        .filter(pl.col("dist_mm") < float(contact_mm))
        .select([
            "FRAMENUMBER",
            pl.col("ANIMALID").alias("idA"),
            pl.col("ANIMALID_B").alias("idB"),
            "dist_mm",
        ])
    )
    return pairs

# ---------- contacts table (serial/parallel) ----------
def build_contact_table(det_xy: pd.DataFrame, contact_mm: float) -> pd.DataFrame:
    return _contacts_polars(det_xy, contact_mm).to_pandas()

def build_contact_table_parallel(det_xy: pd.DataFrame, contact_mm: float, workers: int) -> pd.DataFrame:
    # Polars join-based implementation is already vectorized; parallel workers are unnecessary.
    _ = workers
    return build_contact_table(det_xy, contact_mm)

# ---------- group sizes from contact graph ----------
def _group_sizes_from_contacts(contacts: pd.DataFrame) -> pd.DataFrame:
    rec = []
    for fr, df in contacts.groupby("FRAMENUMBER"):
        adj: Dict[int, set] = {}
        for a, b in zip(df["idA"].astype(int), df["idB"].astype(int)):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        visited = set()
        for node in list(adj.keys()):
            if node in visited:
                continue
            stack = [node]; comp = []
            visited.add(node)
            while stack:
                n = stack.pop()
                comp.append(n)
                for nb in adj.get(n, ()):
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            size = len(comp)
            for u in comp:
                rec.append((fr, u, size))
    if not rec:
        return pd.DataFrame(columns=["FRAMENUMBER","ANIMALID","group_size"], dtype=int)
    return pd.DataFrame(rec, columns=["FRAMENUMBER","ANIMALID","group_size"]).astype({"ANIMALID": int, "group_size": int})

# ---------- social metrics ----------
def social_metrics(
    traj: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    contacts: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    fast_thr: float,
    iso_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traj_pl = (
        _to_polars(traj)
        .with_columns([
            pl.col("FRAMENUMBER").cast(pl.Int64, strict=False),
            pl.col("ANIMALID").cast(pl.Int64, strict=False),
            pl.col("MASS_X").cast(pl.Float64, strict=False),
            pl.col("MASS_Y").cast(pl.Float64, strict=False),
            pl.col("speed").cast(pl.Float64, strict=False),
            pl.col("dt").cast(pl.Float64, strict=False),
            pl.col("moving").cast(pl.Boolean, strict=False),
        ])
        .drop_nulls(["FRAMENUMBER", "ANIMALID"])
    )
    contacts_pl = _to_polars(contacts).with_columns([
        pl.col("FRAMENUMBER").cast(pl.Int64, strict=False),
        pl.col("idA").cast(pl.Int64, strict=False),
        pl.col("idB").cast(pl.Int64, strict=False),
    ])

    speed = traj_pl.select(["FRAMENUMBER", "ANIMALID", "speed"])
    merged = (
        contacts_pl
        .join(speed.rename({"ANIMALID": "idA", "speed": "speed_A"}), on=["FRAMENUMBER", "idA"], how="left")
        .join(speed.rename({"ANIMALID": "idB", "speed": "speed_B"}), on=["FRAMENUMBER", "idB"], how="left")
    )
    approaches_pl = merged.filter((pl.col("speed_A") > fast_thr) & (pl.col("speed_B") < fast_thr))
    escapes_pl = merged.filter((pl.col("speed_B") > fast_thr) & (pl.col("speed_A") < fast_thr))

    det = traj_pl.select(["FRAMENUMBER", "ANIMALID", "MASS_X", "MASS_Y", "dt"]).drop_nulls(["MASS_X", "MASS_Y", "dt"])
    frame_n = det.group_by("FRAMENUMBER").agg(pl.col("ANIMALID").n_unique().alias("n_animals"))
    single_ids = (
        frame_n.filter(pl.col("n_animals") == 1)
        .join(det.select(["FRAMENUMBER", "ANIMALID"]).unique(), on="FRAMENUMBER", how="inner")
        .select("ANIMALID")
    )
    nn_pairs = (
        det.join(det.select([
            pl.col("FRAMENUMBER"),
            pl.col("ANIMALID").alias("ANIMALID_B"),
            pl.col("MASS_X").alias("MASS_X_B"),
            pl.col("MASS_Y").alias("MASS_Y_B"),
        ]), on="FRAMENUMBER", how="inner")
        .filter(pl.col("ANIMALID") != pl.col("ANIMALID_B"))
        .with_columns(_dist_expr("MASS_X", "MASS_Y", "MASS_X_B", "MASS_Y_B").alias("dist_mm"))
    )
    iso_multi = (
        nn_pairs.group_by(["FRAMENUMBER", "ANIMALID"])
        .agg(pl.col("dist_mm").min().alias("nn_mm"))
        .filter(pl.col("nn_mm") > float(iso_mm))
        .select("ANIMALID")
    )
    iso_counts = (
        pl.concat([single_ids, iso_multi], how="vertical_relaxed")
        .group_by("ANIMALID")
        .len()
        .rename({"len": "iso_frames"})
    )

    animals = traj_pl.select("ANIMALID").unique()
    base = (
        animals
        .join(approaches_pl.group_by("idA").len().rename({"idA": "ANIMALID", "len": "fast_approaches"}), on="ANIMALID", how="left")
        .join(escapes_pl.group_by("idB").len().rename({"idB": "ANIMALID", "len": "fast_escapes"}), on="ANIMALID", how="left")
        .join(iso_counts, on="ANIMALID", how="left")
        .join(
            traj_pl.group_by("ANIMALID").agg([
                pl.col("dt").sum().alias("det_time_s"),
                pl.len().alias("total_frames"),
            ]),
            on="ANIMALID",
            how="left",
        )
        .with_columns([
            pl.col("fast_approaches").fill_null(0).cast(pl.Float64),
            pl.col("fast_escapes").fill_null(0).cast(pl.Float64),
            pl.col("iso_frames").fill_null(0).cast(pl.Float64),
            pl.col("det_time_s").fill_null(0.0).cast(pl.Float64),
            pl.col("total_frames").fill_null(0).cast(pl.Float64),
        ])
    )

    ic = pl.concat(
        [
            contacts_pl.select(["FRAMENUMBER", pl.col("idA").alias("ANIMALID")]),
            contacts_pl.select(["FRAMENUMBER", pl.col("idB").alias("ANIMALID")]),
        ],
        how="vertical_relaxed",
    ).unique().with_columns(pl.lit(True).alias("in_contact"))

    state_pl = (
        traj_pl.select(["FRAMENUMBER", "ANIMALID", "dt", "moving"])
        .join(ic, on=["FRAMENUMBER", "ANIMALID"], how="left")
        .with_columns(pl.col("in_contact").fill_null(False))
    )
    gs = pl.from_pandas(_group_sizes_from_contacts(contacts_pl.to_pandas()))
    state_pl = (
        state_pl
        .join(gs, on=["FRAMENUMBER", "ANIMALID"], how="left")
        .with_columns(pl.col("group_size").fill_null(1).cast(pl.Int64))
    )

    state_sorted = state_pl.sort(["ANIMALID", "FRAMENUMBER"]).with_columns([
        pl.col("in_contact").cast(pl.Int8).diff().over("ANIMALID").alias("contact_edge"),
        (pl.col("group_size") == 2).cast(pl.Int8).diff().over("ANIMALID").alias("g2_edge"),
        (pl.col("group_size") == 3).cast(pl.Int8).diff().over("ANIMALID").alias("g3_edge"),
        (pl.col("group_size") == 4).cast(pl.Int8).diff().over("ANIMALID").alias("g4_edge"),
    ])

    per_state = animals
    for cond, out in [
        (pl.col("in_contact"), "contact_total_len_s"),
        ((~pl.col("in_contact")) & pl.col("moving"), "move_isolated_s"),
        (pl.col("in_contact") & pl.col("moving"), "move_in_contact_s"),
        (pl.col("group_size") == 2, "group2_total_len_s"),
        (pl.col("group_size") == 3, "group3_total_len_s"),
        (pl.col("group_size") == 4, "group4_total_len_s"),
    ]:
        per_state = per_state.join(
            state_pl.filter(cond).group_by("ANIMALID").agg(pl.col("dt").sum().alias(out)),
            on="ANIMALID",
            how="left",
        )
    for edge, out in [("contact_edge", "contact_nb"), ("g2_edge", "group2_nb"), ("g3_edge", "group3_nb"), ("g4_edge", "group4_nb")]:
        per_state = per_state.join(
            state_sorted.group_by("ANIMALID").agg((pl.col(edge) == 1).sum().alias(out)),
            on="ANIMALID",
            how="left",
        )
    per_state = per_state.with_columns([
        pl.col("contact_total_len_s").fill_null(0.0),
        pl.col("move_isolated_s").fill_null(0.0),
        pl.col("move_in_contact_s").fill_null(0.0),
        pl.col("group2_total_len_s").fill_null(0.0),
        pl.col("group3_total_len_s").fill_null(0.0),
        pl.col("group4_total_len_s").fill_null(0.0),
        pl.col("contact_nb").fill_null(0).cast(pl.Float64),
        pl.col("group2_nb").fill_null(0).cast(pl.Float64),
        pl.col("group3_nb").fill_null(0).cast(pl.Float64),
        pl.col("group4_nb").fill_null(0).cast(pl.Float64),
    ])

    metrics_pl = (
        base
        .join(per_state, on="ANIMALID", how="left")
        .with_columns([
            pl.when(pl.col("det_time_s") > 0).then(pl.col("det_time_s") / 60.0).otherwise(None).alias("total_minutes"),
            _safe_ratio("iso_frames", "total_frames", "pct_time_alone"),
        ])
        .with_columns([
            (1.0 - pl.col("pct_time_alone")).clip(0.0, 1.0).alias("pct_time_social"),
            _safe_ratio("fast_approaches", "total_minutes", "approaches_per_min"),
            _safe_ratio("fast_escapes", "total_minutes", "escapes_per_min"),
            _safe_ratio("contact_total_len_s", "contact_nb", "contact_mean_dur_s"),
            _safe_ratio("contact_total_len_s", "det_time_s", "contact_prop_time_detection"),
            _safe_ratio("move_isolated_s", "det_time_s", "move_isolated_prop_time_detection"),
            _safe_ratio("move_in_contact_s", "det_time_s", "move_in_contact_prop_time_detection"),
            _safe_ratio("group2_total_len_s", "group2_nb", "group2_mean_dur_s"),
            _safe_ratio("group3_total_len_s", "group3_nb", "group3_mean_dur_s"),
            _safe_ratio("group4_total_len_s", "group4_nb", "group4_mean_dur_s"),
            _safe_ratio("group2_total_len_s", "det_time_s", "group2_prop_time_detection"),
            _safe_ratio("group3_total_len_s", "det_time_s", "group3_prop_time_detection"),
            _safe_ratio("group4_total_len_s", "det_time_s", "group4_prop_time_detection"),
            _safe_ratio("fast_approaches", "contact_nb", "fast_approach_prop_nb_contact"),
            _safe_ratio("fast_escapes", "contact_nb", "fast_escape_prop_nb_contact"),
        ])
        .select([
            "ANIMALID",
            "fast_approaches",
            "fast_escapes",
            "approaches_per_min",
            "escapes_per_min",
            "pct_time_alone",
            "pct_time_social",
            "contact_total_len_s",
            "contact_nb",
            "contact_mean_dur_s",
            "contact_prop_time_detection",
            "move_isolated_prop_time_detection",
            "move_in_contact_prop_time_detection",
            "group2_total_len_s",
            "group2_nb",
            "group2_mean_dur_s",
            "group2_prop_time_detection",
            "group3_total_len_s",
            "group3_nb",
            "group3_mean_dur_s",
            "group3_prop_time_detection",
            "group4_total_len_s",
            "group4_nb",
            "group4_mean_dur_s",
            "group4_prop_time_detection",
            "fast_approach_prop_nb_contact",
            "fast_escape_prop_nb_contact",
        ])
        .sort("ANIMALID")
    )

    metrics = metrics_pl.to_pandas().set_index("ANIMALID").fillna(0.0)
    metrics.index = metrics.index.astype(int)
    metrics.index.name = "ANIMALID"
    approaches = approaches_pl.to_pandas()
    escapes = escapes_pl.to_pandas()
    state = state_pl.select(["FRAMENUMBER", "ANIMALID", "dt", "moving", "in_contact", "group_size"]).to_pandas()
    return metrics, approaches, escapes, state

# ---------- scaling + social index ----------
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
def _scale(df: pd.DataFrame, method: str = "robust") -> pd.DataFrame:
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        scaler = RobustScaler()
    arr = scaler.fit_transform(df)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

def social_index(metrics: pd.DataFrame, scale: str = "robust", weights: dict | None = None) -> pd.Series:
    default_w = {"approaches_per_min": 0.25, "escapes_per_min": 0.25, "pct_time_social": 0.50}
    w = pd.Series(weights or default_w)
    X = metrics.reindex(columns=w.index)
    X = _scale(X, method=scale)
    return X.mul(w, axis=1).sum(axis=1).rename("social_index")
