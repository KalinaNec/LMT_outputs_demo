from __future__ import annotations
import polars as pl
import pandas as pd

__all__ = ["build_traj_polars"]

def build_traj_polars(
    det_pl: pl.DataFrame | pl.LazyFrame,
    frames_pl: pl.DataFrame | pl.LazyFrame,
    move_thr: float,
    burst_thr: float,
    burst_runlen: int = 2,
) -> pd.DataFrame:
    """
    Build trajectory table with a Polars LazyFrame pipeline and collect once.
    """
    det_lf = det_pl.lazy() if isinstance(det_pl, pl.DataFrame) else det_pl
    frames_lf = frames_pl.lazy() if isinstance(frames_pl, pl.DataFrame) else frames_pl

    t_lf = (
        det_lf.join(
            frames_lf.select(["FRAMENUMBER", "t"]),
            on="FRAMENUMBER",
            how="left",
        )
        .sort(["ANIMALID", "t"])
        .with_columns([
            pl.col("MASS_X").cast(pl.Float64, strict=False),
            pl.col("MASS_Y").cast(pl.Float64, strict=False),
            pl.col("t").cast(pl.Datetime("ns")).dt.epoch("ns").alias("t_ns"),
        ])
        .with_columns([
            pl.col("MASS_X").diff().over("ANIMALID").alias("dx"),
            pl.col("MASS_Y").diff().over("ANIMALID").alias("dy"),
            (pl.col("t_ns").diff().over("ANIMALID").cast(pl.Float64) / 1e9).alias("dt"),
        ])
        .with_columns([
            ((pl.col("dx")**2 + pl.col("dy")**2) ** 0.5).alias("step_mm"),
        ])
        .with_columns([
            pl.when(pl.col("dt") > 0.0)
              .then(pl.col("step_mm") / pl.col("dt"))
              .otherwise(0.0)
              .alias("speed"),
        ])
        .with_columns([
            (pl.col("speed") > move_thr).alias("moving"),
        ])
        .with_columns([
            pl.col("DATA").cast(pl.Utf8, strict=False)
                          .str.contains('isRearing="true"', literal=True)
                          .fill_null(False)
                          .alias("rearing"),
        ])
        .with_columns([
            pl.col("speed").fill_null(0.0).cast(pl.Float64),
            pl.when(pl.col("dt").is_finite() & (pl.col("dt") > 0.0))
              .then(pl.col("dt"))
              .otherwise(0.0)
              .cast(pl.Float64)
              .alias("dt"),
        ])
        .drop(["t_ns"])
    )

    t = t_lf.collect().to_pandas()
    if t.empty:
        t["burst"] = False
        return t

    t = t.sort_values(["ANIMALID", "t", "FRAMENUMBER"], kind="stable").reset_index(drop=True)
    burst_mask = t["speed"].gt(burst_thr)
    starts = burst_mask.ne(burst_mask.groupby(t["ANIMALID"]).shift())
    burst_group = starts.groupby(t["ANIMALID"]).cumsum()
    run_len = burst_mask.groupby([t["ANIMALID"], burst_group]).transform("sum")
    t["burst"] = burst_mask & run_len.ge(burst_runlen)
    return t
