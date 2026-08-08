/**
 * Client-side car environment — a faithful TS port of the vectorized NumPy
 * `CarEnv` (see backend/rl/environment.py). Same force-based physics, grip-
 * limited cornering, corridor-aware lidar, and local-window progress, so cars
 * driven by the exported policy behave exactly as they did in training.
 *
 * The parity harness (scripts/parity.ts) asserts this matches Python numerically.
 */

export interface PhysicsConfig {
  dt: number; vmax: number;
  engineForce: number; brakeForce: number;
  cDrag: number; cRoll: number; gSlope: number;
  steerRate: number; vTurnFull: number; gripAlat: number; vEps: number;
  maxSteps: number;
  numRays: number; rayFov: number; rayRange: number; raySamples: number;
  searchBack: number; searchFwd: number;
  lidarCorrBack: number; lidarCorrFwd: number;
  gridCols: number; gridLat: number;
  obsDim: number; actDim: number;
}

export interface TrackData {
  centerline: [number, number][];
  halfWidth: number;
  elevation: number[];
  grade: number[];
  tangentAngle: number[];
}

const TWO_PI = 2 * Math.PI;
const posmod = (a: number, n: number) => ((a % n) + n) % n;
const wrap = (a: number) => posmod(a + Math.PI, TWO_PI) - Math.PI;
const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const STUN_STEPS = 12; // ~1.2 sim-seconds frozen after a crash (effects mode)

/** mulberry32 — small seeded PRNG so spawn randomization is reproducible */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class CarEnv {
  readonly n: number;
  readonly cfg: PhysicsConfig;
  readonly M: number;
  readonly obsDim: number;
  readonly numRays: number;

  // track (full precision)
  private Px: Float64Array;
  private Py: Float64Array;
  private halfWidth: number;
  private hw2: number;
  private elev: Float64Array;
  private grade: Float64Array;
  private tanA: Float64Array;
  private rayAngles: Float64Array;
  private tSamples: Float64Array;

  // per-car state
  x: Float64Array;
  y: Float64Array;
  theta: Float64Array;
  v: Float64Array;
  progIdx: Float64Array;
  progCont: Float64Array;
  lapsDone: Int32Array;
  epLen: Int32Array;
  epReturn: Float64Array;
  offtrack: Uint8Array;
  sensors: Float32Array; // [n * numRays]
  lastAction: Float32Array; // [n * 3]
  lastReward: Float32Array; // [n]
  done: Uint8Array; // [n]
  stun: Int32Array; // [n] steps frozen after a crash (effects mode only)
  readonly effects: boolean; // client-side collisions + wall stop-and-recover
  totalLaps = 0;

  private rng: () => number;
  private obsBuf: Float32Array;

  constructor(track: TrackData, cfg: PhysicsConfig, numEnvs: number, seed = 0, effects = false) {
    this.cfg = cfg;
    this.effects = effects;
    this.n = numEnvs;
    this.M = track.centerline.length;
    this.obsDim = cfg.obsDim;
    this.numRays = cfg.numRays;
    this.halfWidth = track.halfWidth;
    this.hw2 = track.halfWidth * track.halfWidth;

    this.Px = new Float64Array(this.M);
    this.Py = new Float64Array(this.M);
    for (let i = 0; i < this.M; i++) {
      this.Px[i] = track.centerline[i][0];
      this.Py[i] = track.centerline[i][1];
    }
    this.elev = Float64Array.from(track.elevation);
    this.grade = Float64Array.from(track.grade);
    this.tanA = Float64Array.from(track.tangentAngle);

    // ray_angles = linspace(-FOV/2, FOV/2, numRays)
    this.rayAngles = new Float64Array(cfg.numRays);
    const denomR = cfg.numRays > 1 ? cfg.numRays - 1 : 1;
    for (let i = 0; i < cfg.numRays; i++)
      this.rayAngles[i] = -cfg.rayFov / 2 + (i * cfg.rayFov) / denomR;
    // t = linspace(0, rayRange, raySamples)
    this.tSamples = new Float64Array(cfg.raySamples);
    const denomS = cfg.raySamples > 1 ? cfg.raySamples - 1 : 1;
    for (let i = 0; i < cfg.raySamples; i++) this.tSamples[i] = (i * cfg.rayRange) / denomS;

    const n = numEnvs;
    this.x = new Float64Array(n);
    this.y = new Float64Array(n);
    this.theta = new Float64Array(n);
    this.v = new Float64Array(n);
    this.progIdx = new Float64Array(n);
    this.progCont = new Float64Array(n);
    this.lapsDone = new Int32Array(n);
    this.epLen = new Int32Array(n);
    this.epReturn = new Float64Array(n);
    this.offtrack = new Uint8Array(n);
    this.sensors = new Float32Array(n * cfg.numRays);
    this.lastAction = new Float32Array(n * 3);
    this.lastReward = new Float32Array(n);
    this.done = new Uint8Array(n);
    this.stun = new Int32Array(n);
    this.obsBuf = new Float32Array(n * this.obsDim);
    this.rng = mulberry32(seed || 1);
  }

  // ------------------------------------------------------------------- reset
  reset(): Float32Array {
    const all = new Int32Array(this.n);
    for (let i = 0; i < this.n; i++) all[i] = i;
    this.resetIdx(all, true);
    this.resolveSpawnOverlaps();
    return this.observe();
  }

  /** overwrite the state of a specific car directly (used by the parity harness) */
  setCar(i: number, x: number, y: number, theta: number, v: number, progIdx: number, progCont: number) {
    this.x[i] = x; this.y[i] = y; this.theta[i] = theta; this.v[i] = v;
    this.progIdx[i] = progIdx; this.progCont[i] = progCont;
    this.lapsDone[i] = 0; this.epLen[i] = 0; this.epReturn[i] = 0; this.offtrack[i] = 0;
  }

  resetIdx(idx: Int32Array | number[], randomize: boolean) {
    const { gridCols, gridLat } = this.cfg;
    const nRows = Math.ceil(this.n / gridCols);
    for (let t = 0; t < idx.length; t++) {
      const i = idx[t];
      const row = Math.floor(i / gridCols);
      const col = i % gridCols;
      const along = posmod(Math.round((row * this.M) / nRows), this.M);
      const ta = this.tanA[along];
      const tx = Math.cos(ta), ty = Math.sin(ta);
      const nx = -ty, ny = tx; // left-hand normal
      let lat = (col === 0 ? -1 : 1) * gridLat * this.halfWidth;
      if (randomize) lat += (this.rng() * 0.12 - 0.06) * this.halfWidth;
      this.x[i] = this.Px[along] + lat * nx;
      this.y[i] = this.Py[along] + lat * ny;
      const hnoise = randomize ? this.rng() * 0.3 - 0.15 : 0;
      this.theta[i] = ta + hnoise;
      this.v[i] = 0;
      this.progIdx[i] = along;
      this.progCont[i] = along;
      this.lapsDone[i] = 0;
      this.epReturn[i] = 0;
      this.epLen[i] = 0;
      this.offtrack[i] = 0;
    }
  }

  private resolveSpawnOverlaps(minGap = 20.0, iters = 12) {
    const { gridCols, gridLat } = this.cfg;
    for (let it = 0; it < iters; it++) {
      // find the closest pair
      let bi = -1, bj = -1, bd = Infinity;
      for (let i = 0; i < this.n; i++) {
        for (let j = i + 1; j < this.n; j++) {
          const dx = this.x[i] - this.x[j], dy = this.y[i] - this.y[j];
          const d = Math.hypot(dx, dy);
          if (d < bd) { bd = d; bi = i; bj = j; }
        }
      }
      if (bd >= minGap || bi < 0) break;
      const c = Math.max(bi, bj);
      const newIdx = posmod(Math.round(this.progIdx[c]) + 3, this.M);
      const ta = this.tanA[newIdx];
      const tx = Math.cos(ta), ty = Math.sin(ta);
      const sign = c % gridCols === 0 ? -1 : 1;
      const lat = sign * gridLat * this.halfWidth;
      this.x[c] = this.Px[newIdx] - ty * lat;
      this.y[c] = this.Py[newIdx] + tx * lat;
      this.theta[c] = ta;
      this.progIdx[c] = newIdx;
      this.progCont[c] = newIdx;
    }
  }

  // --------------------------------------------------------------- progress
  /** returns [newIdx, delta, minD2] for car i within its local window */
  private advanceProgressOne(i: number): [number, number, number] {
    const base = Math.round(this.progIdx[i]);
    let bestOff = -this.cfg.searchBack;
    let bestD2 = Infinity;
    for (let off = -this.cfg.searchBack; off <= this.cfg.searchFwd; off++) {
      const idx = posmod(base + off, this.M);
      const dx = this.Px[idx] - this.x[i];
      const dy = this.Py[idx] - this.y[i];
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; bestOff = off; }
    }
    return [posmod(base + bestOff, this.M), bestOff, bestD2];
  }

  // ---------------------------------------------------------------- sensors
  /** corridor-aware lidar for car i → fills this.sensors[i*R .. ] */
  private raycastOne(i: number) {
    const { lidarCorrBack: cb, lidarCorrFwd: cf, rayRange, raySamples, numRays } = this.cfg;
    const base = Math.round(this.progIdx[i]);
    const th = this.theta[i], xi = this.x[i], yi = this.y[i];
    for (let r = 0; r < numRays; r++) {
      const ang = th + this.rayAngles[r];
      const dx = Math.cos(ang), dy = Math.sin(ang);
      let dist = rayRange;
      for (let s = 0; s < raySamples; s++) {
        const t = this.tSamples[s];
        const px = xi + t * dx, py = yi + t * dy;
        // min squared distance to the car's own corridor points
        let minD2 = Infinity;
        for (let c = -cb; c < cf; c++) {
          const idx = posmod(base + c, this.M);
          const ex = px - this.Px[idx], ey = py - this.Py[idx];
          const d2 = ex * ex + ey * ey;
          if (d2 < minD2) { minD2 = d2; if (minD2 <= this.hw2) break; }
        }
        if (minD2 > this.hw2) { dist = t; break; } // first off-corridor sample
      }
      this.sensors[i * numRays + r] = dist / rayRange;
    }
  }

  // ---------------------------------------------------------------- observe
  observe(): Float32Array {
    const { obsDim, numRays, vmax } = this.cfg;
    for (let i = 0; i < this.n; i++) {
      this.raycastOne(i);
      const idx = posmod(Math.round(this.progIdx[i]), this.M);
      const he = wrap(this.theta[i] - this.tanA[idx]);
      const o = i * obsDim;
      for (let r = 0; r < numRays; r++) this.obsBuf[o + r] = this.sensors[i * numRays + r];
      this.obsBuf[o + numRays] = this.v[i] / vmax;
      this.obsBuf[o + numRays + 1] = Math.sin(he);
      this.obsBuf[o + numRays + 2] = Math.cos(he);
    }
    return this.obsBuf;
  }

  // -------------------------------------------------------------------- step
  /** @param actions flat [n*3] raw actions (will be clipped). @returns obs */
  step(actions: Float32Array): Float32Array {
    const c = this.cfg;
    const finishedIdx: number[] = [];
    for (let i = 0; i < this.n; i++) {
      // effects mode: a crashed car freezes ~1s at the wall, then rejoins the track
      if (this.effects && this.stun[i] > 0) {
        this.v[i] = 0;
        this.offtrack[i] = 1;
        this.lastAction[i * 3] = 0; this.lastAction[i * 3 + 1] = 0; this.lastAction[i * 3 + 2] = 0;
        this.lastReward[i] = 0;
        this.done[i] = 0;
        if (--this.stun[i] === 0) this.recoverToTrack(i);
        continue;
      }

      const steer = clamp(actions[i * 3], -1, 1);
      const accel = clamp(actions[i * 3 + 1], 0, 1);
      const brake = clamp(actions[i * 3 + 2], 0, 1);
      this.lastAction[i * 3] = steer;
      this.lastAction[i * 3 + 1] = accel;
      this.lastAction[i * 3 + 2] = brake;

      // cornering with grip limit
      const desiredYaw = steer * c.steerRate * Math.min(1, this.v[i] / c.vTurnFull);
      const maxYaw = c.gripAlat / Math.max(this.v[i], c.vEps);
      const yaw = clamp(desiredYaw, -maxYaw, maxYaw);
      this.theta[i] = wrap(this.theta[i] + yaw * c.dt);

      // longitudinal forces incl. slope gravity
      const curIdx = posmod(Math.round(this.progIdx[i]), this.M);
      const g = clamp(this.grade[curIdx], -0.4, 0.4);
      const F =
        accel * c.engineForce -
        brake * c.brakeForce -
        c.cDrag * this.v[i] * this.v[i] -
        c.cRoll * this.v[i] -
        c.gSlope * g;
      this.v[i] = clamp(this.v[i] + F * c.dt, 0, c.vmax);

      this.x[i] += this.v[i] * Math.cos(this.theta[i]) * c.dt;
      this.y[i] += this.v[i] * Math.sin(this.theta[i]) * c.dt;

      const [newIdx, delta, minD2] = this.advanceProgressOne(i);
      this.progCont[i] += delta;
      this.progIdx[i] = newIdx;

      const newLaps = Math.floor(this.progCont[i] / this.M);
      const inc = Math.max(0, newLaps - this.lapsDone[i]);
      this.totalLaps += inc;
      if (newLaps > this.lapsDone[i]) this.lapsDone[i] = newLaps;

      const off = minD2 > this.hw2;
      this.offtrack[i] = off ? 1 : 0;

      let reward = (delta / this.M) * 12.0 - 0.012;
      if (off) reward = -2.0;
      this.lastReward[i] = reward;

      this.epLen[i] += 1;
      this.epReturn[i] += reward;

      if (this.effects) {
        // playground: crash → stop at the wall (no teleport), no time-limit resets
        this.done[i] = 0;
        if (off) {
          this.stun[i] = STUN_STEPS;
          this.v[i] = 0;
        }
      } else {
        const truncated = this.epLen[i] >= c.maxSteps;
        const doneCar = off || truncated;
        this.done[i] = doneCar ? 1 : 0;
        if (doneCar) finishedIdx.push(i);
      }
    }

    if (this.effects) this.resolveCollisions();

    let obs = this.observe();
    if (finishedIdx.length) {
      this.resetIdx(finishedIdx, true);
      obs = this.observe();
    }
    return obs;
  }

  /** snap a recovered car back onto the racing surface, facing forward */
  private recoverToTrack(i: number) {
    const idx = posmod(Math.round(this.progIdx[i]), this.M);
    this.x[i] = this.Px[idx];
    this.y[i] = this.Py[idx];
    this.theta[i] = this.tanA[idx];
    this.v[i] = 0;
    this.offtrack[i] = 0;
  }

  /** cheap pairwise car-car collision: separate overlapping cars + bleed speed */
  private resolveCollisions() {
    const R = 4.4;
    const R2 = R * R;
    for (let i = 0; i < this.n; i++) {
      if (this.stun[i] > 0) continue;
      for (let j = i + 1; j < this.n; j++) {
        const dx = this.x[j] - this.x[i];
        const dy = this.y[j] - this.y[i];
        const d2 = dx * dx + dy * dy;
        if (d2 < R2 && d2 > 1e-6) {
          const d = Math.sqrt(d2);
          const ux = dx / d;
          const uy = dy / d;
          if (this.stun[j] === 0) {
            // both mobile: separate fully (half each) with a mild speed scrub
            const push = (R - d) * 0.5;
            this.x[i] -= ux * push;
            this.y[i] -= uy * push;
            this.x[j] += ux * push;
            this.y[j] += uy * push;
            this.v[i] *= 0.85;
            this.v[j] *= 0.85;
          } else {
            // j is frozen: push i clear of it
            const push = R - d;
            this.x[i] -= ux * push;
            this.y[i] -= uy * push;
            this.v[i] *= 0.8;
          }
        }
      }
    }
  }

  // -------------------------------------------------------- render helpers
  /** surface height for car i (for the 3D view) */
  zOf(i: number): number {
    return this.elev[posmod(Math.round(this.progIdx[i]), this.M)];
  }
  gradeOf(i: number): number {
    return this.grade[posmod(Math.round(this.progIdx[i]), this.M)];
  }
}
