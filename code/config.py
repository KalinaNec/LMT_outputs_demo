from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Weights:
    approach: float = 1.0
    escape: float = -0.5

@dataclass
class Thresholds:
    move_thr: float
    burst_thr: float
    fast_thr: float
    contact_mm: float
    iso_mm: float
    burst_runlen: int

@dataclass
class RunConfig:
    sqlite_path: Path
    out_dir: Path
    workers: int
    dominance_method: str          # "davids" | "plackettluce"
    dom_transform: Optional[str]   # None | "log1p" | "rate"
    dom_cap: Optional[float]
    social_scale: str              # "minmax" | "robust" | "zscore"
    debug: bool = False
    speed_logx: bool = False
    speed_smooth: int = 0
    speed_fit_gmm: bool = False
    approach_weight: float = 1.0
    escape_weight: float = -0.5
