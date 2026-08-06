"""Download real F1 circuit geometry and convert to local-coordinate centerlines.

Source: https://github.com/bacinger/f1-circuits (GeoJSON LineStrings, lat/lon).

Each circuit is projected to a local metric frame (equirectangular about its
centroid), resampled to a fixed number of evenly-spaced points, lightly smoothed
to remove GPS jitter, then scaled so every track has the same total centerline
length (so corner difficulty is comparable across circuits for the RL agent).

Run:  python scripts/fetch_tracks.py
Output: backend/rl/track_data/<id>.json
"""

import os
import json
import math
import urllib.request

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl", "track_data")
os.makedirs(OUT_DIR, exist_ok=True)

BASE = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/{}.geojson"

# id -> display label
CIRCUITS = {
    "mc-1929": "Monaco — Monte Carlo",
    "it-1922": "Monza",
    "gb-1948": "Silverstone",
    "be-1925": "Spa-Francorchamps",
    "jp-1962": "Suzuka",
    "bh-2002": "Bahrain",
    "us-2012": "COTA — Austin",
    "hu-1986": "Hungaroring",
    "au-1953": "Melbourne",
    "at-1969": "Red Bull Ring",
}

M = 260                # points per centerline
TARGET_LEN = 1700.0    # world-unit total length (matches the default circuit feel)
HALF_WIDTH = 30.0
R_EARTH = 6371000.0


def project(coords):
    lon = np.array([c[0] for c in coords])
    lat = np.array([c[1] for c in coords])
    lat0 = math.radians(lat.mean())
    lon0 = lon.mean()
    x = R_EARTH * np.radians(lon - lon0) * math.cos(lat0)
    y = R_EARTH * np.radians(lat - lat0)
    return np.stack([x, y], axis=1)


def resample_closed(pts, n):
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2))
    j = 0
    for i, t in enumerate(targets):
        while j < len(cum) - 1 and cum[j + 1] <= t:
            j += 1
        denom = seglen[j] if seglen[j] > 1e-9 else 1.0
        f = (t - cum[j]) / denom
        out[i] = pts[j] + f * (pts[j + 1] - pts[j])
    return out


def smooth_closed(P, iters=1, window=3):
    half = window // 2
    for _ in range(iters):
        acc = np.zeros_like(P)
        for s in range(-half, half + 1):
            acc += np.roll(P, s, axis=0)
        P = acc / window
    return P


def total_length(P):
    seg = np.roll(P, -1, axis=0) - P
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def build(circuit_id):
    url = BASE.format(circuit_id)
    with urllib.request.urlopen(url, timeout=30) as resp:
        gj = json.load(resp)
    feat = gj["features"][0]
    props = feat.get("properties", {})
    coords = feat["geometry"]["coordinates"]
    if feat["geometry"]["type"] == "MultiLineString":
        coords = max(coords, key=len)

    P = project(coords)
    P = resample_closed(P, M)
    P = smooth_closed(P, iters=1, window=3)
    P -= P.mean(axis=0)
    scale = TARGET_LEN / total_length(P)
    P *= scale

    return {
        "id": circuit_id,
        "name": props.get("Name", circuit_id),
        "location": props.get("Location", ""),
        "lengthM": props.get("length"),
        "halfWidth": HALF_WIDTH,
        "points": [[round(float(x), 2), round(float(y), 2)] for x, y in P],
    }


def main():
    index = []
    for cid, label in CIRCUITS.items():
        try:
            data = build(cid)
            data["label"] = label
            with open(os.path.join(OUT_DIR, cid + ".json"), "w", encoding="utf-8") as f:
                json.dump(data, f)
            index.append({"id": cid, "label": label})
            print(f"  ok  {cid:10s} {label}")
        except Exception as e:
            print(f"  FAIL {cid}: {e}")
    with open(os.path.join(OUT_DIR, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {len(index)} circuits to {OUT_DIR}")


if __name__ == "__main__":
    main()
