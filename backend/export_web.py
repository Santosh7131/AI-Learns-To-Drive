"""Export a trained policy + tracks to static assets the browser can run.

The deployed "playground" is static (no backend): the browser loads the exported
policy and runs the SAME physics + Transformer forward pass client-side. This
script is the bridge — run it after training to refresh what the deployed site plays.

    cd backend
    python export_web.py                 # uses best-default (fallback autosave-default)
    python export_web.py --ckpt my-run   # a specific checkpoint
    python export_web.py --track default # track whose geometry drives the parity trace

Outputs (into frontend/public/web/):
    policy.json          actor weights + architecture + physics config + obs layout
    track-<id>.json      centerline / half-width / elevation / terrain for each circuit
    tracks.json          [{id,label}] index
    parity-trace.json    a fixed seeded rollout (obs windows, action means, states)
                         so the TS port can be asserted numerically equal to Python.
"""

import argparse
import json
import os

import numpy as np
import torch

from rl.environment import (
    CarEnv, NUM_RAYS, RAY_FOV, RAY_RANGE, RAY_SAMPLES,
    SEARCH_BACK, SEARCH_FWD, LIDAR_CORR_BACK, LIDAR_CORR_FWD,
    GRID_COLS, GRID_LAT,
)
from rl.model import TransformerActorCritic
from rl.tracks import list_tracks
from rl.trainer import CKPT_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "frontend", "public", "web"))
# the parity trace is a dev artifact (used by frontend/scripts/parity.ts), kept
# OUT of public/ so it isn't bundled into the deployed static site.
PARITY_DIR = os.path.abspath(os.path.join(HERE, "..", "frontend", "scripts"))

# model architecture (must mirror model.py defaults / trainer construction)
ARCH = dict(d_model=64, nhead=4, num_layers=2, dim_ff=128)


def _find_ckpt(name):
    """Resolve a checkpoint name to a path, with sensible fallbacks."""
    candidates = [name] if name else ["best-default", "autosave-default"]
    for c in candidates:
        p = os.path.join(CKPT_DIR, c + ".pt")
        if os.path.exists(p):
            return c, p
    raise FileNotFoundError(
        f"no checkpoint found (looked for {candidates} in {CKPT_DIR}). Train first."
    )


def _tensor(t):
    """Serialize a tensor as {shape, data(flat, row-major)} for easy TS reload."""
    a = t.detach().cpu().float().numpy()
    return {"shape": list(a.shape), "data": a.reshape(-1).tolist()}


def _physics_config(env: CarEnv):
    return {
        "dt": env.dt, "vmax": env.vmax,
        "engineForce": env.engine_force, "brakeForce": env.brake_force,
        "cDrag": env.c_drag, "cRoll": env.c_roll, "gSlope": env.g_slope,
        "steerRate": env.steer_rate, "vTurnFull": env.v_turn_full,
        "gripAlat": env.grip_alat, "vEps": env.v_eps, "maxSteps": env.max_steps,
        # structural / observation layout
        "numRays": NUM_RAYS, "rayFov": RAY_FOV, "rayRange": RAY_RANGE,
        "raySamples": RAY_SAMPLES,
        "searchBack": SEARCH_BACK, "searchFwd": SEARCH_FWD,
        "lidarCorrBack": LIDAR_CORR_BACK, "lidarCorrFwd": LIDAR_CORR_FWD,
        "gridCols": GRID_COLS, "gridLat": GRID_LAT,
        "obsDim": CarEnv.OBS_DIM, "actDim": CarEnv.ACT_DIM,
    }


def _greedy_action(model, obs_window):
    """Deterministic playback action = the actor mean (no sampling), clipped
    exactly as the environment applies it."""
    with torch.no_grad():
        win = torch.from_numpy(obs_window.astype(np.float32))
        mean, _std, _v = model.forward(win)
    mean = mean.cpu().numpy()
    applied = mean.copy()
    applied[:, 0] = np.clip(applied[:, 0], -1.0, 1.0)   # steer
    applied[:, 1] = np.clip(applied[:, 1], 0.0, 1.0)    # accel
    applied[:, 2] = np.clip(applied[:, 2], 0.0, 1.0)    # brake
    return mean, applied


def _state(env: CarEnv):
    return {
        "x": env.x.tolist(), "y": env.y.tolist(), "theta": env.theta.tolist(),
        "v": env.v.tolist(), "progIdx": env.prog_idx.tolist(),
        "progCont": env.prog_cont.tolist(),
        "offtrack": env.offtrack.astype(int).tolist(),
        "sensors": env.sensors.tolist(),
    }


def build_parity_trace(model, track, n_envs=4, steps=40, seed=0):
    """A fully-deterministic seeded rollout under the greedy policy. Records the
    obs window + action mean + applied action + post-step state each step, so the
    TS port can be checked against it exactly (policy AND physics)."""
    env = CarEnv(num_envs=n_envs, seed=seed, track=track)
    obs = env.reset()
    K = model.window
    window = np.zeros((n_envs, K, CarEnv.OBS_DIM), dtype=np.float32)
    window[:, -1, :] = obs

    trace = {"track": track, "seed": seed, "numEnvs": n_envs, "window": K,
             "init": _state(env), "steps": []}

    for _ in range(steps):
        mean, applied = _greedy_action(model, window)
        rec = {
            "obsWindow": window.tolist(),
            "meanAction": mean.tolist(),
            "appliedAction": applied.tolist(),
        }
        obs, _r, done, _info = env.step(applied)
        rec["post"] = _state(env)
        rec["done"] = done.astype(int).tolist()
        trace["steps"].append(rec)
        # roll the window exactly like the trainer does
        window = np.roll(window, -1, axis=1)
        window[:, -1, :] = obs
        if done.any():  # a reset injects RNG; stop so the trace stays reproducible
            window[done, :-1, :] = 0.0
            break
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint name (no .pt)")
    ap.add_argument("--track", default="default", help="track for the parity trace")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ck_name, ck_path = _find_ckpt(args.ckpt)
    try:
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)  # trusted local ckpt only
    win = int(ck.get("window", 8))
    obs_dim = int(ck.get("obs_dim", CarEnv.OBS_DIM))
    trained_track = ck.get("track", "default")

    model = TransformerActorCritic(obs_dim=obs_dim, act_dim=CarEnv.ACT_DIM, window=win, **ARCH)
    model.load_state_dict(ck["model"])
    model.eval()

    # ---- policy.json: only the tensors the actor forward pass needs ----
    sd = model.state_dict()
    want_prefixes = ("embed.", "encoder.", "norm.", "actor_mean.")
    weights = {k: _tensor(v) for k, v in sd.items() if k.startswith(want_prefixes)}
    env0 = CarEnv(num_envs=1, seed=0, track=args.track)
    policy = {
        "checkpoint": ck_name,
        "trainedTrack": trained_track,
        "globalStep": int(ck.get("global_step", 0)),
        "arch": {"obsDim": obs_dim, "actDim": CarEnv.ACT_DIM, "window": win, **ARCH},
        "physics": _physics_config(env0),
        "weights": weights,
    }
    with open(os.path.join(OUT_DIR, "policy.json"), "w") as f:
        json.dump(policy, f, separators=(",", ":"))
    print(f"[export] policy.json  <- {ck_name} @ step {policy['globalStep']}  "
          f"({len(weights)} tensors)")

    # ---- track geometries ----
    tracks = list_tracks()
    with open(os.path.join(OUT_DIR, "tracks.json"), "w") as f:
        json.dump(tracks, f)
    for t in tracks:
        tenv = CarEnv(num_envs=1, seed=0, track=t["id"])
        geo = tenv.track_geometry()
        # full-precision derived arrays the client physics needs — exported (not
        # re-derived in JS) so the slope force and headings match Python exactly.
        geo["grade"] = [float(g) for g in tenv.track_elev_grad]
        geo["tangentAngle"] = [float(a) for a in tenv.tangent_angle]
        with open(os.path.join(OUT_DIR, f"track-{t['id']}.json"), "w") as f:
            json.dump(geo, f, separators=(",", ":"))
    print(f"[export] {len(tracks)} track geometries")

    # ---- parity trace (dev only; not deployed) ----
    os.makedirs(PARITY_DIR, exist_ok=True)
    trace = build_parity_trace(model, args.track)
    with open(os.path.join(PARITY_DIR, "parity-trace.json"), "w") as f:
        json.dump(trace, f)
    print(f"[export] scripts/parity-trace.json  ({len(trace['steps'])} steps on '{args.track}')")
    print(f"[export] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
