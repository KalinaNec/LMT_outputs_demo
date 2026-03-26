from pathlib import Path
import os
import sqlite3
import polars as pl
import duckdb

__all__ = ["load_tables_polars"]

def _load_sqlite_extension(con: duckdb.DuckDBPyConnection) -> None:
    """Load DuckDB sqlite extension without forcing a network install every run."""
    try:
        con.execute("LOAD sqlite;")
    except duckdb.Error:
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")


def _load_tables_sqlite3(db: str, debug: bool, frame_limit: int) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    con = sqlite3.connect(db)
    try:
        animals_df = pl.read_database("SELECT * FROM ANIMAL", con)

        frame_hi_sql = ""
        params: tuple[int, ...] | None = None
        if debug:
            min_fr_df = pl.read_database(
                "SELECT MIN(FRAMENUMBER) AS min_fr FROM FRAME",
                con,
            )
            min_fr = min_fr_df.item() if min_fr_df.height else None
            min_fr = int(min_fr) if min_fr is not None else 0
            max_fr = min_fr + frame_limit
            frame_hi_sql = " WHERE FRAMENUMBER <= ?"
            params = (max_fr,)

        detection_df = pl.read_database(
            (
                "SELECT FRAMENUMBER, ANIMALID, MASS_X, MASS_Y, DATA "
                "FROM DETECTION"
            ) + frame_hi_sql,
            con,
            execute_options={"parameters": params} if params is not None else None,
        )
        frames_df = pl.read_database(
            "SELECT FRAMENUMBER, TIMESTAMP FROM FRAME" + frame_hi_sql,
            con,
            execute_options={"parameters": params} if params is not None else None,
        )
    finally:
        con.close()

    animals_lf = animals_df.lazy()
    detection_lf = detection_df.lazy()
    frames_lf = frames_df.lazy()

    detection_lf = detection_lf.with_columns(
        pl.col("ANIMALID").cast(pl.Float64, strict=False)
    )
    frames_lf = frames_lf.with_columns(
        pl.col("TIMESTAMP").cast(pl.Int64),
        (pl.col("TIMESTAMP") * 1_000_000).cast(pl.Datetime("ns")).alias("t"),
    )
    return animals_lf, detection_lf, frames_lf


def load_tables_polars(db: Path, debug: bool, frame_limit: int = 50_000):
    """
    Load ANIMAL fully, and only the necessary columns from DETECTION and FRAME.

    This still loads the full tables (unless debug=True), but avoids dragging
    useless columns into RAM.
    """
    db = str(db)
    use_duckdb = os.getenv("TM_USE_DUCKDB", "1").strip().lower() not in {"0", "false", "no"}
    if not use_duckdb:
        return _load_tables_sqlite3(db, debug=debug, frame_limit=frame_limit)

    con = duckdb.connect()
    try:
        _load_sqlite_extension(con)
        con.execute(f"ATTACH '{db}' AS sqdb (TYPE sqlite);")
        frame_filter = ""
        if debug:
            min_fr = con.execute(
                "SELECT MIN(FRAMENUMBER) AS min_fr FROM sqdb.FRAME"
            ).fetchone()[0]
            min_fr = int(min_fr) if min_fr is not None else 0
            max_fr = min_fr + frame_limit
            frame_filter = f" WHERE FRAMENUMBER <= {max_fr}"

        # ANIMAL is tiny, just load all columns
        animals_lf = con.execute("SELECT * FROM sqdb.ANIMAL").pl().lazy()

        # DETECTION: keep only what you actually use downstream
        detection_lf = con.execute(f"""
            SELECT
                FRAMENUMBER,
                ANIMALID,
                MASS_X,
                MASS_Y,
                DATA
            FROM sqdb.DETECTION
            {frame_filter}
        """).pl().lazy()

        # FRAME: keep only FRAMENUMBER + TIMESTAMP
        frames_lf = con.execute(f"""
            SELECT
                FRAMENUMBER,
                TIMESTAMP
            FROM sqdb.FRAME
            {frame_filter}
        """).pl().lazy()
    finally:
        con.close()

    # types & derived columns
    detection_lf = detection_lf.with_columns(
        pl.col("ANIMALID").cast(pl.Float64, strict=False)
    )

    # FRAME.TIMESTAMP is ms since epoch in your LMT DBs
    frames_lf = frames_lf.with_columns(
        pl.col("TIMESTAMP").cast(pl.Int64),
        (pl.col("TIMESTAMP") * 1_000_000).cast(pl.Datetime("ns")).alias("t"),
    )

    return animals_lf, detection_lf, frames_lf
