# Reinforcement Car 🏎️

A full-stack app that trains a **Transformer-based reinforcement-learning policy** to
drive a car around a race track. The policy outputs three controls — **steering,
acceleration, brake** — and learns via **PPO** across **20 cars training in parallel**,
all visualized live in the browser. You can save / load / delete model checkpoints
from the UI.

```
reinforcement_car/
├── backend/            Flask + PyTorch (the RL brain)
│   ├── app.py          REST + Server-Sent-Events endpoints
│   ├── rl/
│   │   ├── environment.py   Vectorized race-track physics + terrain (20 cars)
│   │   ├── model.py         Transformer actor-critic policy
│   │   ├── trainer.py       PPO trainer + per-track checkpoints (background thread)
│   │   ├── tracks.py        Track registry (default + real F1 circuits)
│   │   └── track_data/      Real circuit geometry (JSON, committed)
│   ├── scripts/fetch_tracks.py   Downloads/derives the F1 circuits
│   └── checkpoints/    Saved .pt models + history.json (generated)
└── frontend/           Vite + React + TypeScript + shadcn/ui + Three.js
    └── src/
        ├── App.tsx
        ├── lib/         api.ts, trackGeometry.ts, carViz.ts, utils.ts
        ├── components/  CarCanvas (2D), CarScene3D (3D), CarInspector,
        │                ControlPanel, MetricsPanel, CheckpointPanel, ErrorBoundary
        └── components/ui/  shadcn primitives
```

## How it works

- **Environment** (`environment.py`): 20 cars are stepped together in NumPy on a closed
  loop track. Each car perceives the world through a **lidar fan of 7 raycasts** (distance
  to the track edge per ray) plus speed and heading error — so the observation is
  `7 rays + 3 = 10` features. On-track and the lidar are **corridor-aware**: they are
  judged only against centerline points near the car's current lap position, so a section
  of the circuit that runs close by in space (but is far away around the lap) reads as a
  wall — a car can't slide onto an overlapping road. Reward = arc-length progress per step;
  leaving the corridor ends the episode with a penalty and respawns the car.
- **Policy** (`model.py`): a small Transformer encoder reads a window of the last 8
  observations and outputs the 3 control means + a value estimate. Actions are sampled
  from a diagonal Gaussian.
- **Training** (`trainer.py`): PPO with GAE, running in a background thread. A thread-safe
  snapshot of all car positions, **per-car control outputs, reward, and sensor readings**
  is streamed to the frontend ~30×/sec over SSE.
- **GPU acceleration**: if CUDA is available the Transformer + PPO update run on the GPU
  automatically (else CPU). The device is shown in the header. The PPO update (Transformer
  backprop) is the GPU-heavy part — on an RTX 4060 it's ~**3.8× faster** than CPU at a
  fleet of 128 cars. Use the **Parallel cars** selector (20 → 150) to scale the fleet and
  load the GPU; the per-car physics stays on CPU (NumPy), which becomes the limiter at very
  large fleets, so ~64–128 is the sweet spot.
- **Persistence**: the trainer auto-saves **per-track** checkpoints — `best-<track>` (by
  mean return) and a rolling `autosave-<track>` — plus a `history.json` of metrics. On
  startup it resumes the current track's autosave (architecture-validated), and switching
  circuits loads that circuit's own saved model (or carries the current weights over as a
  warm start). Checkpoint writes happen off the lock so they never stall the live stream.
- **Frontend**: an HTML canvas renders all 20 cars as proper top-down vehicles (with
  client-side interpolation, motion trails, curbs, and a checkered start line), live
  metrics with a reward sparkline, training controls, and full checkpoint management.

### Real F1 circuits
The circuit dropdown includes the procedural **Default Circuit** plus **10 real F1 tracks**
(Monaco, Monza, Spa, Silverstone, Suzuka, Bahrain, COTA, Hungaroring, Melbourne, Red Bull
Ring). Their geometry is real: downloaded from the open
[`f1-circuits`](https://github.com/bacinger/f1-circuits) dataset (lat/lon), projected to a
local metric frame, resampled, and scaled to a common lap length so corner difficulty is
comparable. Switching circuits keeps the learned model (the observation shape is
track-independent), so you watch the agent adapt.

- Re-download / add circuits: `python backend/scripts/fetch_tracks.py` (writes
  `backend/rl/track_data/*.json`). `tracks.py` auto-registers everything in that folder.
- Real tracks run back close to themselves, so the environment tracks progress with a
  **local arc-length window** (a windowed search around each car's current lap position),
  and on/off-track + lidar are judged against that same local corridor — so a car can't
  drift onto a neighbouring section of the circuit that merely passes close by.

### 3D view
Toggle **3D / 2D** (top-right of the arena). The 3D view is a real Three.js /
react-three-fiber scene with:
- **Gentle elevation** — the world is not flat: a smooth height field gives rolling hills
  and the track follows the terrain, with cars riding the surface and tilting to the slope.
  Real F1 circuits are simple non-crossing loops, so there are no flyovers — except a small
  local **bridge** at a genuine self-crossing (detected with a real segment-intersection
  test), which currently only occurs on **Suzuka's figure-8**.
- **Kerbs** — red/white striped kerbs are generated at the corners (where the centerline
  curvature is high) for a real race-track look.
- **Detailed F1 cars** (nose, wings, sidepods, halo, wheels with rims) that **lean into
  corners and pitch under braking**.
- **Sun + dynamic shadows**, a procedural **sky**, and grass terrain.
- **Orbit camera** (drag to rotate, scroll to zoom) and a **Chase** cam for a selected car.
- **Smooth motion**: the client predicts each car forward from its velocity between network
  updates and gently corrects toward the authoritative state, so motion is continuous at
  the display refresh rate rather than stepping at the ~30 Hz telemetry rate.

### Physics
Longitudinal motion is force-based: engine thrust, braking, **quadratic aerodynamic drag**,
rolling resistance, and **gravity along the local slope** (cars gain speed downhill, bleed
it uphill — real momentum). Top speed emerges from drag balancing thrust (~250–300 km/h).
Cornering is **grip-limited**: lateral acceleration is capped, so a car understeers if it
tries to turn too hard at speed and must brake for corners. Speed is shown in **km/h** in
the car inspector. Tune constants at the top of `backend/rl/environment.py`.

### Inspecting an agent
- **Click any car** to open an inspector showing the Transformer's live outputs (steering /
  acceleration / brake bars), speed, instantaneous reward, and its 7 lidar readings.
- In 2D, the **Sensors** toggle draws every car's lidar fan; in both views the selected
  car's rays are highlighted in cyan so you can see exactly what it perceives.

## Running it

### 1. Backend (Flask, port 5000)

```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1          # (Windows PowerShell)
pip install -r requirements.txt
python app.py
```

### 2. Frontend (Vite, port 5173)

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**, click **Start Training**, and watch the 20 cars learn.
The Vite dev server proxies `/api/*` to the Flask backend, so no CORS setup is needed.

## Using the app

- **Start / Pause / Stop** the PPO training loop.
- **Reset Cars** respawns all cars at the start line (keeps the learned weights).
- **Sliders** tune learning rate, simulation speed (smoothness vs. throughput), and
  exploration on the fly.
- **Checkpoints**: type a name and **Save**; **Load** swaps a checkpoint into the live
  model; **Delete** removes it. Files live in `backend/checkpoints/`.

> Tip: lower the **Sim speed** delay to train faster (cars move in larger jumps), raise it
> for smoother, more watchable motion.

## Deploying

The project has **two deploy shapes**, matching how it's actually used:

- **The playground — what you deploy publicly.** A *static* site that loads the pre-trained
  policy and runs the whole simulation in each visitor's browser (a faithful TS port of the
  physics + Transformer, validated against the Python original). No backend, no server compute,
  no GPU bill — the free tier is plenty.
- **The training console — runs locally on your GPU.** This is where the model is actually
  trained (live PPO + controls); it is not something you need to deploy.

### Deploy the playground (static, recommended)

1. Generate + commit the web assets from a trained checkpoint (one-time, and again whenever you
   want the demo to show a newer model — Render's static build has no Python/GPU, so these are
   committed artifacts):
   ```bash
   cd backend && python export_web.py     # -> frontend/public/web/{policy,track-*,tracks}.json
   git add frontend/public/web && git commit -m "refresh playground model"
   ```
2. Push to GitHub → Render **New → Blueprint** (reads `render.yaml`: a free **Static Site** with
   `VITE_PLAYBACK_ONLY=true`). Also works as-is on Vercel / Netlify / GitHub Pages — build the
   `frontend` with `VITE_PLAYBACK_ONLY=true` and publish `frontend/dist`.

Visitors pick a quality preset (fleet size). On a WebGPU browser the whole fleet's Transformer
runs **batched on the visitor's GPU** (hundreds of cars); without WebGPU it runs on the CPU (a
few dozen). If a backend happens to be configured, a very weak device can fall back to a server
stream — but none is required.

### Host the full training console (optional)

To host the *training* app too, the repo ships a `Dockerfile` (Node builds the UI → Python
serves API + UI same-origin, CPU torch) with the pre-trained default model baked in, so it boots
already-trained. Uncomment the `reinforcement-car-trainer` service in `render.yaml`, or point a
Render **Web Service** (runtime *Docker*) at it. Render injects `$PORT` (honored by `serve.py`).
⚠️ Free tier is CPU-only, 512 MB (PyTorch may OOM — bump `plan` to `starter`), sleeps when idle,
and disk is ephemeral — fine for a demo, not real training. Same-origin, so no CORS needed.

### Manual / any host

```powershell
cd frontend; npm run build      # produces frontend/dist
cd ..\backend; python serve.py  # waitress on 0.0.0.0:5000 — serves API + UI
```

Open **http://<host>:5000**. In production the frontend calls `/api/*` same-origin (no dev
proxy, no CORS needed).

**Run exactly one process.** The trainer is a single stateful in-process object with a
background training thread; multiple workers would each spawn their own trainer. Do **not**
use `gunicorn -w N` / multiple uwsgi workers. `waitress` (one process, many threads) is the
right model and is cross-platform.

**Configuration (environment variables, all optional):**

| Var | Purpose |
|---|---|
| `RLCAR_HOST` / `RLCAR_PORT` | bind address / port (serve.py defaults `0.0.0.0:5000`; dev `app.py` defaults `127.0.0.1:5000`) |
| `RLCAR_THREADS` | waitress worker threads (default 16). Each live SSE viewer holds one thread, so raise this if many people watch at once |
| `RLCAR_TOKEN` | if set, all `POST/PUT/DELETE` require header `X-Auth-Token: <token>` (see security note below) |
| `RLCAR_CORS_ORIGINS` | comma-separated allowed origins for `/api/*` — only needed if the UI is served from a *different* origin than the API |

**GPU:** uses CUDA automatically if the host has an NVIDIA GPU + a CUDA build of torch;
otherwise it falls back to CPU. Install the CUDA wheel explicitly for GPU
(`pip install torch --index-url https://download.pytorch.org/whl/cu121`).

**Behind nginx:** SSE needs proxy buffering off for `/api/telemetry` (`proxy_buffering off;`);
the backend already sends `X-Accel-Buffering: no`.

**Security notes:**
- The API is **unauthenticated by default** — anyone who can reach it can control training and
  manage checkpoints. For a public deployment, either set `RLCAR_TOKEN` **or** put the app
  behind your own auth / network restriction. ⚠️ Setting `RLCAR_TOKEN` protects the raw API but
  the bundled UI does not yet send the header, so it would stop the in-page controls from
  working — use network-level auth, or ask to have the token wired into the UI.
- Checkpoint names are sanitized to a safe in-directory basename (no path traversal).
  Checkpoint files are trusted, server-generated artifacts (`torch.load` uses pickle); do not
  point the checkpoints directory at untrusted files.
