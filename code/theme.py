
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import matplotlib.pyplot as plt

# --- palette ---
PRIMARY   = "#a93a5e"   # rose
MID       = "#a67081"   # mauve
DARK      = "#831b3c"   # wine

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
ACCENT   = DARK
BAR_1    = MID
BAR_2    = PRIMARY
BAR_3    = DARK
BLUSH_BG = lighten(PRIMARY, 0.92)
GRID     = lighten(MID, 0.75)
INK      = darken(DARK, 0.35)

ACCENTS = [MID, PRIMARY, DARK]

# label ordering is kept privately; manipulate via setter/getter
_LABEL_ORDER: Optional[Iterable] = None

def set_label_order(order: Optional[Iterable]) -> None:
    """Set fixed x-axis label order (e.g., RFID tails). Pass None to clear."""
    global _LABEL_ORDER
    _LABEL_ORDER = tuple(order) if order is not None else None

def get_label_order() -> Optional[tuple]:
    return _LABEL_ORDER

# ----- mpl theme + small helpers -----
def set_theme():
    import matplotlib as mpl
    mpl.rcParams.update({
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

def style_axes(ax, y0line=True, zero_color=None):
    for s in ["top","right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.9)
    ymin, ymax = ax.get_ylim()
    if y0line and (ymin < 0 < ymax):
        ax.axhline(0, color=zero_color or _blend(PRIMARY, "#000000", 0.25), linewidth=1.0, alpha=0.6, zorder=0)
    ax.tick_params(axis="x", labelrotation=45)

def cute_bar_effects(ax):
    from matplotlib import patheffects as pe
    for p in ax.patches:
        p.set_edgecolor("white")
        p.set_linewidth(1.0)
        p.set_alpha(0.95)
        p.set_path_effects([pe.withSimplePatchShadow(offset=(0,-1), alpha=.18, shadow_rgbFace="#000000")])

def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
