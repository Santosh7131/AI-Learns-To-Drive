"""Vectorized race-track environment: lidar sensors, real circuits, terrain, physics.

The car follows a 2D track (x, y) but the world has **elevation**: a smooth
height field h(x, y) gives the track and surrounding terrain hills. Elevation
feeds back into the physics as a gravity force along the local slope, so cars
gain speed downhill and bleed it uphill (momentum).

Longitudinal motion is force-based (engine, brakes, quadratic aero drag, rolling
resistance, slope gravity) so top speed emerges naturally and momentum feels
real. Cornering is grip-limited: lateral acceleration is capped, so a car cannot
turn sharply at high speed (understeer) and must brake for corners.

Observation (obs_dim = NUM_RAYS + 3):
    [ ray_0 .. ray_{R-1} (normalized lidar distances),
      speed / vmax, sin(heading_error), cos(heading_error) ]
"""

import numpy as np

from .tracks import build_centerline

NUM_RAYS = 7
RAY_FOV = np.radians(150.0)
RAY_RANGE = 220.0
RAY_SAMPLES = 28

# Progress / on-track is judged against centerline points NEAR the car's current
# lap position (a corridor), never the whole track. This is what stops a car from
# "hopping" onto a different track section that merely runs close by in space.
SEARCH_BACK = 5      # progress window (indices) used for the car itself
SEARCH_FWD = 12
LIDAR_CORR_BACK = 8  # wider corridor for the rays (must cover the ray range ahead)
LIDAR_CORR_FWD = 48

# starting grid: cars spawn in 2-wide rows spread EVENLY AROUND THE LAP (by car
# id), so they never stack — and rows stay far apart even through tight corners
# (a compact grid at the line would pile up wherever it crossed a hairpin).
GRID_COLS = 2
GRID_LAT = 0.5       # lateral offset per column, as a fraction of half-width

TWO_PI = 2.0 * np.pi


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class CarEnv:
    NUM_RAYS = NUM_RAYS
    OBS_DIM = NUM_RAYS + 3
    ACT_DIM = 3

    def __init__(self, num_envs=20, seed=0, track="default"):
        self.num_envs = num_envs
        self.track_name = track
        self.rng = np.random.default_rng(seed)

        # ---- physics (force-based; units ~ meters, dt seconds) ----
        self.dt = 0.1
        self.vmax = 85.0            # clamp / normalization (~305 km/h * scale)
        self.engine_force = 42.0
        self.brake_force = 78.0
        self.c_drag = 0.0050        # quadratic aero drag
        self.c_roll = 0.25          # linear rolling resistance
        self.g_slope = 26.0         # gravity strength along slope
        self.steer_rate = 3.0       # max yaw rate (rad/s) at full lock, low speed
        self.v_turn_full = 12.0     # speed for full steering authority
        self.grip_alat = 44.0       # max lateral accel (cornering grip)
        self.v_eps = 3.0
        self.max_steps = 1500

        self.ray_angles = np.linspace(-RAY_FOV / 2, RAY_FOV / 2, NUM_RAYS)
        self.ray_range = RAY_RANGE
        self._offsets = np.arange(-SEARCH_BACK, SEARCH_FWD + 1)

        self._build_track()

        n = num_envs
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.theta = np.zeros(n)
        self.v = np.zeros(n)
        self.prog_idx = np.zeros(n)
        self.prog_cont = np.zeros(n)
        self.laps_done = np.zeros(n, dtype=np.int64)
        self.total_laps_completed = 0
        self.ep_return = np.zeros(n)
        self.ep_len = np.zeros(n, dtype=np.int64)
        self.offtrack = np.zeros(n, dtype=bool)

        self.sensors = np.zeros((n, NUM_RAYS), dtype=np.float32)
        self.last_action = np.zeros((n, 3), dtype=np.float32)
        self.last_reward = np.zeros(n, dtype=np.float32)

        self.finished_returns = []
        self.finished_lengths = []

    # ------------------------------------------------------------------ track
    def _build_track(self):
        self.P, self.half_width = build_centerline(self.track_name)
        self.M = len(self.P)

        nxt = np.roll(self.P, -1, axis=0)
        prv = np.roll(self.P, 1, axis=0)
        tan = nxt - prv
        tan /= (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9)
        self.T = tan
        self.tangent_angle = np.arctan2(self.T[:, 1], self.T[:, 0])

        seg = np.roll(self.P, -1, axis=0) - self.P
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.total_len = float(self.seg_len.sum())

        margin = self.half_width + 30
        self.bounds = {
            "minX": float(self.P[:, 0].min() - margin),
            "maxX": float(self.P[:, 0].max() + margin),
            "minY": float(self.P[:, 1].min() - margin),
            "maxY": float(self.P[:, 1].max() + margin),
        }

        # ---- terrain height field h(x,y) = sum of low-frequency sines ----
        ext = max(self.bounds["maxX"] - self.bounds["minX"],
                  self.bounds["maxY"] - self.bounds["minY"])
        self.ext = ext
        self.terrain = [
            (ext * 0.024, TWO_PI / (ext * 1.30), TWO_PI / (ext * 1.05), 0.6),
            (ext * 0.014, -TWO_PI / (ext * 0.72), TWO_PI / (ext * 0.85), 2.1),
            (ext * 0.008, TWO_PI / (ext * 0.46), -TWO_PI / (ext * 0.52), 4.0),
        ]

        # ---- track surface height: the road simply follows the gentle terrain.
        # The ONLY lift is a small local bridge at a GENUINE self-crossing (e.g.
        # Suzuka's figure-8), detected with a real segment-intersection test, so
        # it never fires on parallel straights or hairpins.
        self.track_elev, self.track_elev_grad = self._compute_track_elevation()

    def _terrain_height(self, x, y):
        h = np.zeros_like(np.asarray(x, dtype=np.float64))
        for a, wx, wy, p in self.terrain:
            h += a * np.sin(wx * x + wy * y + p)
        return h

    def _compute_track_elevation(self):
        """Track height = gentle terrain sampled along the centerline, plus a
        small local bump over the over-pass at any true self-crossing."""
        M = self.M
        seg = max(self.total_len / M, 1e-3)
        z = self._terrain_height(self.P[:, 0], self.P[:, 1])
        z = z + self._crossover_bumps()
        grade = (np.roll(z, -1) - np.roll(z, 1)) / (2.0 * seg)
        return z, grade

    def _crossover_bumps(self):
        """Local lift over the later-lap pass at each TRUE centerline crossing.
        Uses an exact segment-intersection test (orientation signs), so it only
        fires where the track physically crosses itself (0 on most circuits,
        1 on Suzuka) — never on close-but-parallel sections."""
        M = self.M
        a = self.P
        b = np.roll(self.P, -1, axis=0)

        def orient(px, py, qx, qy, rx, ry):
            return (qx - px) * (ry - py) - (qy - py) * (rx - px)

        ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        # segment i (rows) vs segment j (cols)
        d1 = orient(ax[None, :], ay[None, :], bx[None, :], by[None, :], ax[:, None], ay[:, None])
        d2 = orient(ax[None, :], ay[None, :], bx[None, :], by[None, :], bx[:, None], by[:, None])
        d3 = orient(ax[:, None], ay[:, None], bx[:, None], by[:, None], ax[None, :], ay[None, :])
        d4 = orient(ax[:, None], ay[:, None], bx[:, None], by[:, None], bx[None, :], by[None, :])
        crosses = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
        ii = np.arange(M)
        lap = np.minimum(np.abs(ii[:, None] - ii[None, :]), M - np.abs(ii[:, None] - ii[None, :]))
        crosses &= lap > 6  # ignore neighbours that share endpoints
        iu, ju = np.where(np.triu(crosses, k=1))

        bump = np.zeros(M)
        if len(ju) > 0:
            bridge = min(8.0, 0.28 * self.half_width)
            win = max(3, M // 40)
            offs = np.arange(-win, win + 1)
            hump = 0.5 * (1.0 + np.cos(np.pi * offs / (win + 1)))
            for j in ju:  # raise the higher-lap-index pass over the lower
                idxs = (j + offs) % M
                bump[idxs] = np.maximum(bump[idxs], bridge * hump)
        return bump

    def track_geometry(self):
        return {
            "track": self.track_name,
            "centerline": self.P.tolist(),
            "halfWidth": self.half_width,
            "bounds": self.bounds,
            "numPoints": self.M,
            "rayAngles": self.ray_angles.tolist(),
            "rayRange": self.ray_range,
            "numRays": NUM_RAYS,
            "terrain": [{"a": a, "wx": wx, "wy": wy, "p": p} for (a, wx, wy, p) in self.terrain],
            "elevation": [round(float(e), 2) for e in self.track_elev],
        }

    # ------------------------------------------------------------------ reset
    def reset(self):
        self._reset_idx(np.arange(self.num_envs), randomize=True)
        self._resolve_spawn_overlaps()
        return self._observe()

    def _resolve_spawn_overlaps(self, min_gap=20.0, iters=12):
        """Nudge any spawned cars that landed too close (e.g. on a figure-8
        crossover) forward along the track until they are clear."""
        for _ in range(iters):
            pos = np.stack([self.x, self.y], axis=1)
            d = np.hypot(pos[:, None, 0] - pos[None, :, 0],
                         pos[:, None, 1] - pos[None, :, 1])
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] >= min_gap:
                break
            c = max(i, j)                      # move the higher-id car
            new_idx = (int(round(self.prog_idx[c])) + 3) % self.M
            tang = self.T[new_idx]
            sign = -1.0 if (c % GRID_COLS) == 0 else 1.0
            lat = sign * GRID_LAT * self.half_width
            self.x[c] = self.P[new_idx, 0] - tang[1] * lat
            self.y[c] = self.P[new_idx, 1] + tang[0] * lat
            self.theta[c] = self.tangent_angle[new_idx]
            self.prog_idx[c] = float(new_idx)
            self.prog_cont[c] = float(new_idx)

    def _reset_idx(self, idx, randomize=True):
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        n = idx.size
        # slot is fixed per car id; rows are spread evenly around the lap so no
        # two cars stack (robust to hairpins) and respawns land in a clear slot.
        n_rows = (self.num_envs + GRID_COLS - 1) // GRID_COLS
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        along = np.round(row * (self.M / n_rows)).astype(np.int64) % self.M
        base = self.P[along]                                    # (n, 2)
        tang = self.T[along]
        nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)       # left-hand normal
        lat = np.where(col == 0, -1.0, 1.0) * GRID_LAT * self.half_width
        if randomize:
            lat = lat + self.rng.uniform(-0.06, 0.06, n) * self.half_width
        self.x[idx] = base[:, 0] + lat * nrm[:, 0]
        self.y[idx] = base[:, 1] + lat * nrm[:, 1]
        hnoise = (self.rng.uniform(-0.15, 0.15, n) if randomize else np.zeros(n))
        self.theta[idx] = self.tangent_angle[along] + hnoise
        self.v[idx] = 0.0
        self.prog_idx[idx] = along.astype(np.float64)
        self.prog_cont[idx] = along.astype(np.float64)
        self.laps_done[idx] = 0
        self.ep_return[idx] = 0.0
        self.ep_len[idx] = 0
        self.offtrack[idx] = False

    # --------------------------------------------------------------- progress
    def _advance_progress(self):
        """Move each car's lap index within a local window and return the squared
        distance to the closest centerline point in that window (the corridor)."""
        base = np.round(self.prog_idx).astype(np.int64)
        cand = (base[:, None] + self._offsets[None, :]) % self.M
        cp = self.P[cand]
        pos = np.stack([self.x, self.y], axis=1)[:, None, :]
        d2 = ((cp - pos) ** 2).sum(axis=2)
        best = np.argmin(d2, axis=1)
        chosen_off = self._offsets[best]
        new_idx = (base + chosen_off) % self.M
        min_d2 = d2[np.arange(self.num_envs), best]
        return new_idx, chosen_off, min_d2

    def _corridor_points(self, back, fwd):
        """(N, C, 2) centerline points around each car's current lap position."""
        base = np.round(self.prog_idx).astype(np.int64)
        offs = np.arange(-back, fwd)
        cidx = (base[:, None] + offs[None, :]) % self.M
        return self.P[cidx]

    # ---------------------------------------------------------------- sensors
    def _raycast(self):
        """Corridor-aware lidar: a ray sample counts as 'track' only if it is
        within half-width of the car's OWN corridor, so a neighbouring section of
        the circuit that merely runs close by reads as a wall, not open road."""
        t = np.linspace(0.0, self.ray_range, RAY_SAMPLES)
        ang = self.theta[:, None] + self.ray_angles[None, :]
        dx = np.cos(ang)
        dy = np.sin(ang)
        px = (self.x[:, None, None] + t[None, None, :] * dx[:, :, None]).reshape(self.num_envs, -1)
        py = (self.y[:, None, None] + t[None, None, :] * dy[:, :, None]).reshape(self.num_envs, -1)

        cpts = self._corridor_points(LIDAR_CORR_BACK, LIDAR_CORR_FWD)  # (N, C, 2)
        ddx = px[:, :, None] - cpts[:, None, :, 0]                     # (N, R*S, C)
        ddy = py[:, :, None] - cpts[:, None, :, 1]
        d2 = ddx * ddx + ddy * ddy
        on = d2.min(axis=2) <= (self.half_width * self.half_width)     # (N, R*S)
        off = (~on).reshape(self.num_envs, NUM_RAYS, RAY_SAMPLES)
        any_off = off.any(axis=2)
        first = off.argmax(axis=2)
        dist = np.where(any_off, t[first], self.ray_range)
        return dist.astype(np.float32)

    def _observe(self):
        idx = np.round(self.prog_idx).astype(np.int64) % self.M
        ta = self.tangent_angle[idx]
        heading_err = _wrap(self.theta - ta)
        dist = self._raycast()
        self.sensors = (dist / self.ray_range).astype(np.float32)
        obs = np.concatenate([
            self.sensors,
            (self.v / self.vmax)[:, None],
            np.sin(heading_err)[:, None],
            np.cos(heading_err)[:, None],
        ], axis=1).astype(np.float32)
        return obs

    # -------------------------------------------------------------------- step
    def step(self, actions):
        self.finished_returns = []
        self.finished_lengths = []

        steer = np.clip(actions[:, 0], -1.0, 1.0)
        accel = np.clip(actions[:, 1], 0.0, 1.0)
        brake = np.clip(actions[:, 2], 0.0, 1.0)
        self.last_action = np.stack([steer, accel, brake], axis=1).astype(np.float32)

        # --- cornering with grip limit (understeer at speed) ---
        desired_yaw = steer * self.steer_rate * np.minimum(1.0, self.v / self.v_turn_full)
        max_yaw = self.grip_alat / np.maximum(self.v, self.v_eps)
        yaw = np.clip(desired_yaw, -max_yaw, max_yaw)
        self.theta = _wrap(self.theta + yaw * self.dt)

        # --- longitudinal forces incl. slope gravity (momentum) ---
        # slope = how fast the track surface rises along the car's lap position
        cur_idx = np.round(self.prog_idx).astype(np.int64) % self.M
        grade = np.clip(self.track_elev_grad[cur_idx], -0.4, 0.4)
        F = (accel * self.engine_force
             - brake * self.brake_force
             - self.c_drag * self.v * self.v
             - self.c_roll * self.v
             - self.g_slope * grade)
        self.v = np.clip(self.v + F * self.dt, 0.0, self.vmax)

        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt

        new_idx, delta, min_d2 = self._advance_progress()
        self.prog_cont += delta
        self.prog_idx = new_idx.astype(np.float64)

        # lap counting: monotonic, forward-only (jitter across the line can't
        # invent or remove laps)
        new_laps = np.floor(self.prog_cont / self.M).astype(np.int64)
        inc = np.clip(new_laps - self.laps_done, 0, None)
        self.total_laps_completed += int(inc.sum())
        self.laps_done = np.maximum(self.laps_done, new_laps)

        # off-track = too far from the car's OWN corridor (not from any nearby
        # section), so a car can no longer slide onto an overlapping road.
        off = min_d2 > (self.half_width * self.half_width)
        self.offtrack = off

        progress_reward = (delta / self.M) * 12.0
        reward = progress_reward - 0.012
        reward = np.where(off, -2.0, reward)
        self.last_reward = reward.astype(np.float32)

        self.ep_len += 1
        truncated = self.ep_len >= self.max_steps
        terminated = off                 # only a crash is a true terminal state
        done = terminated | truncated    # both end (and reset) the episode
        self.ep_return += reward

        obs = self._observe()

        fin = np.where(done)[0]
        if len(fin) > 0:
            self.finished_returns = self.ep_return[fin].tolist()
            self.finished_lengths = self.ep_len[fin].tolist()
            self._reset_idx(fin, randomize=True)
            obs = self._observe()

        info = {
            "terminated": terminated,
            "total_laps": int(self.total_laps_completed),
            "mean_speed": float(self.v.mean()),
        }
        return obs, reward.astype(np.float32), done, info

    # ------------------------------------------------------------- render data
    def render_state(self):
        cur_idx = np.round(self.prog_idx).astype(np.int64) % self.M
        z = self.track_elev[cur_idx]
        grade = self.track_elev_grad[cur_idx]
        return {
            "cars": [
                {
                    "id": int(i),
                    "x": round(float(self.x[i]), 2),
                    "y": round(float(self.y[i]), 2),
                    "z": round(float(z[i]), 2),
                    "grade": round(float(grade[i]), 4),
                    "theta": round(float(self.theta[i]), 4),
                    "v": round(float(self.v[i]), 2),
                    "offtrack": bool(self.offtrack[i]),
                    "steer": round(float(self.last_action[i, 0]), 3),
                    "accel": round(float(self.last_action[i, 1]), 3),
                    "brake": round(float(self.last_action[i, 2]), 3),
                    "reward": round(float(self.last_reward[i]), 4),
                    "sensors": [round(float(s), 3) for s in self.sensors[i]],
                }
                for i in range(self.num_envs)
            ]
        }
