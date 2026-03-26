# dominance_pl_stream.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict

import pandas as pd
from openskill.models import PlackettLuce

# builds a stream of PlackettLuce ratings from a stream of events (loser,winner,datetime)
def events_from_interactions(
    approaches: pd.DataFrame,
    escapes: pd.DataFrame,
    animals: Iterable[int | str] | None = None,
    time_col_candidates: tuple[str, ...] = ("datetime", "time", "timestamp", "DateTime"),
) -> pd.DataFrame:
    def _timecol(df: pd.DataFrame) -> str:
        for c in time_col_candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            lc = c.lower()
            if "time" in lc or "date" in lc:
                return c
        raise KeyError("No datetime-like column in approaches/escapes.")

    A = pd.DataFrame(columns=["winner", "loser", "datetime"])
    if approaches is not None and len(approaches):
        t = _timecol(approaches)
        A = approaches.rename(
            columns={t: "datetime", "idA": "winner", "idB": "loser"}
        )[["winner", "loser", "datetime"]]

    E = pd.DataFrame(columns=["winner", "loser", "datetime"])
    if escapes is not None and len(escapes):
        t = _timecol(escapes)
        # B escapes from A → count as B wins over A
        E = escapes.rename(columns={t: "datetime"})
        E = E.rename(columns={"idB": "winner", "idA": "loser"})[
            ["winner", "loser", "datetime"]
        ]

    ev = pd.concat([A, E], ignore_index=True).dropna()
    if ev.empty:
        return ev.reindex(columns=["loser", "winner", "datetime"])

    ev["winner"] = ev["winner"].astype(str)
    ev["loser"] = ev["loser"].astype(str)
    ev = ev[ev["winner"] != ev["loser"]].copy()

    if animals is not None:
        S = set(map(str, animals))
        ev = ev[ev["winner"].isin(S) & ev["loser"].isin(S)]

    ev["datetime"] = pd.to_datetime(ev["datetime"])
    ev = ev.sort_values("datetime").reset_index(drop=True)
    return ev[["loser", "winner", "datetime"]]


# ---------- sequential OpenSkill stream ----------
@dataclass
class PLStream:
    ranking: Dict[str, any]      # final openskill ratings
    Ord: pd.DataFrame            # ordinal in time (index=time)
    Mu: pd.DataFrame             # mu in time
    Sig: pd.DataFrame            # sigma in time
    match_df: pd.DataFrame       # events (loser,winner,datetime)


def rank_openskill_stream(
    events: pd.DataFrame,
    animal_ids: Iterable[int | str],
    prior: Optional[Dict[str, any]] = None,
    history_stride: int = 1,
) -> PLStream:
    """
    OpenSkill stream:
    - always keeps full final ranking
    - stores every `history_stride`-th snapshot in Ord/Mu/Sig
    """
    history_stride = int(history_stride) if history_stride is not None else 1
    if history_stride < 1:
        history_stride = 1

    model = PlackettLuce(limit_sigma=True, balance=True)
    ids = [str(x) for x in animal_ids]
    R: Dict[str, any] = dict(prior) if isinstance(prior, dict) else {}
    for a in ids:
        R.setdefault(a, model.rating())

    snaps_o, snaps_m, snaps_s, dts = [], [], [], []

    for i, (loser, winner, dt) in enumerate(
        events[["loser", "winner", "datetime"]].itertuples(index=False, name=None)
    ):
        loser = str(loser)
        winner = str(winner)
        new = model.rate([[R[loser]], [R[winner]]], ranks=[1, 0])
        R[loser], R[winner] = new[0][0], new[1][0]

        # Keep a full per-event trace when history_stride=1 (DeepEcoHab-like).
        if (i % history_stride) == 0:
            snaps_o.append({k: round(float(R[k].ordinal()), 3) for k in R})
            snaps_m.append({k: R[k].mu for k in R})
            snaps_s.append({k: R[k].sigma for k in R})
            dts.append(pd.to_datetime(dt))

    if snaps_o:
        Ord = pd.DataFrame(snaps_o, index=pd.to_datetime(dts)).sort_index()
        Mu = pd.DataFrame(snaps_m, index=Ord.index).sort_index()
        Sig = pd.DataFrame(snaps_s, index=Ord.index).sort_index()
        # Match DeepEcoHab handling of possible duplicate event timestamps.
        keep = ~Ord.index.duplicated(keep="last")
        Ord = Ord.loc[keep]
        Mu = Mu.loc[keep]
        Sig = Sig.loc[keep]
    else:
        Ord = pd.DataFrame(columns=ids)
        Mu = pd.DataFrame(columns=ids)
        Sig = pd.DataFrame(columns=ids)

    # we keep a *slim* copy of events
    match_df = events[["loser", "winner", "datetime"]].copy()

    return PLStream(ranking=R, Ord=Ord, Mu=Mu, Sig=Sig, match_df=match_df)


# ---------- helpers ----------
def final_dom_index_from_ord(Ord: pd.DataFrame) -> pd.Series:
    if Ord.empty:
        return pd.Series(dtype=float, name="dominance_index")
    s = Ord.iloc[-1].astype(float)
    mn, mx = float(s.min()), float(s.max())
    out = (s - mn) / (mx - mn) if mx > mn else s * 0.0
    out.name = "dominance_index"
    return out


def resample_ord(
    Ord: pd.DataFrame,
    Mu: pd.DataFrame,
    Sig: pd.DataFrame,
    bin: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if Ord.empty:
        return Ord, Mu, Sig
    Ord_b = Ord.resample(bin).last().ffill()
    Mu_b = (
        Mu.resample(bin).last().ffill()
        if not Mu.empty
        else pd.DataFrame(index=Ord_b.index, columns=Ord_b.columns)
    )
    Sig_b = (
        Sig.resample(bin).last().ffill()
        if not Sig.empty
        else pd.DataFrame(index=Ord_b.index, columns=Ord_b.columns)
    )
    return Ord_b, Mu_b, Sig_b
