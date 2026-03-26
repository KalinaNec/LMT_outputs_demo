import argparse, os
def parse_args():
    p = argparse.ArgumentParser(description="Compute Social, Spatial & Dominance indices for one LMT .sqlite")
    p.add_argument("sqlite")
    p.add_argument("out", nargs="?", default=None)
    p.add_argument("--social-scale", choices=["minmax","robust","zscore"], default="robust")
    p.add_argument("--dominance", choices=["davids","plackettluce"], default="plackettluce")
    p.add_argument("--approach-weight", type=float, default=1.0,
                   help="Weight for approach-win events in dominance calculations [1.0]")
    p.add_argument("--escape-weight", type=float, default=-0.5,
                   help="Signed weight for escape events; negative flips event direction [default: -0.5]")
    p.add_argument("--dom-transform", choices=[None,"log1p","rate"], default="rate")
    p.add_argument("--dom-cap", type=float, default=None)
    p.add_argument("--no-heatmap", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--speed-logx", action="store_true")
    p.add_argument("--speed-smooth", type=int, default=0)
    p.add_argument("--speed-fit-gmm", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return p.parse_args()
