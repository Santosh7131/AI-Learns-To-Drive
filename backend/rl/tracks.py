"""Track definitions.

Two kinds of tracks:
  * "default" — a procedurally generated star-shaped circuit (always available).
  * Real F1 circuits — loaded from `track_data/*.json`, produced by
    `scripts/fetch_tracks.py` from real lat/lon geometry (github.com/bacinger/f1-circuits).

Each track resolves to (centerline points [M,2], half_width). Real circuits can
self-approach, which is why the environment tracks progress with a *local*
arc-length window (see environment.py) rather than a global nearest-point search.
"""

import os
import json
import glob

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_data")


def _build_default(M=240):
    ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
    r = 230.0 + 55.0 * np.sin(3 * ang) + 20.0 * np.cos(2 * ang)
    P = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    return P, 46.0


# ---- load real circuits from disk ----
_REAL = {}  # id -> {label, points (np), halfWidth}


def _load_real():
    if not os.path.isdir(DATA_DIR):
        return
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            _REAL[d["id"]] = {
                "label": d.get("label", d.get("name", d["id"])),
                "points": np.array(d["points"], dtype=np.float64),
                "halfWidth": float(d.get("halfWidth", 30.0)),
            }
        except Exception:
            pass


_load_real()


def list_tracks():
    out = [{"id": "default", "label": "Default Circuit"}]
    for cid, spec in _REAL.items():
        out.append({"id": cid, "label": spec["label"]})
    return out


def build_centerline(name, M=240):
    if name in _REAL:
        spec = _REAL[name]
        return spec["points"].copy(), spec["halfWidth"]
    return _build_default(M)
