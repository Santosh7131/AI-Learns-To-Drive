export interface Car {
  id: number;
  x: number;
  y: number;
  z: number;
  grade: number;
  theta: number;
  v: number;
  offtrack: boolean;
  steer: number;
  accel: number;
  brake: number;
  reward: number;
  sensors: number[];
}

export interface Telemetry {
  cars: Car[];
  step: number;
  status: string;
}

export interface TrackOption {
  id: string;
  label: string;
}

export interface Metrics {
  status: string;
  track: string;
  device: string;
  numEnvs: number;
  globalStep: number;
  updates: number;
  totalLaps: number;
  meanReturn: number;
  meanEpisodeLen: number;
  bestReturn: number;
  fps: number;
  lr: number;
  loss: { policy?: number; value?: number; entropy?: number };
}

export interface TerrainComponent {
  a: number;
  wx: number;
  wy: number;
  p: number;
}

export interface TrackGeometry {
  track: string;
  centerline: [number, number][];
  halfWidth: number;
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  numPoints: number;
  rayAngles: number[];
  rayRange: number;
  numRays: number;
  terrain: TerrainComponent[];
  elevation: number[]; // per-centerline-point track surface height (flyovers)
}

export interface HistoryPoint {
  step: number;
  updates: number;
  meanReturn: number;
  bestReturn: number;
}

export interface Checkpoint {
  name: string;
  globalStep: number;
  updates: number;
  meanReturn: number;
  bestReturn: number;
  created: number;
  sizeKB: number;
}

// API base. Same-origin "/api" by default (backend serves the built UI, or the
// dev Vite proxy handles it). Set VITE_API_URL at build time to point a
// separately-hosted static frontend at a remote backend.
const root = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "");
export const API_BASE = `${root}/api`;
const base = API_BASE;

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  track: () => fetch(`${base}/track`).then(json<TrackGeometry>),
  tracks: () => fetch(`${base}/tracks`).then(json<TrackOption[]>),
  setTrack: (name: string) =>
    fetch(`${base}/track`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then(json<TrackGeometry>),
  status: () => fetch(`${base}/status`).then(json<Metrics>),
  history: () => fetch(`${base}/history`).then(json<HistoryPoint[]>),
  control: (action: string) =>
    fetch(`${base}/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    }).then(json<Metrics>),
  config: (cfg: Record<string, number>) =>
    fetch(`${base}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    }).then(json<Metrics>),
  listCheckpoints: () => fetch(`${base}/checkpoints`).then(json<Checkpoint[]>),
  saveCheckpoint: (name: string) =>
    fetch(`${base}/checkpoints`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then(json<Checkpoint>),
  loadCheckpoint: (name: string) =>
    fetch(`${base}/checkpoints/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then(json<Checkpoint>),
  deleteCheckpoint: (name: string) =>
    fetch(`${base}/checkpoints/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }).then(json<{ deleted: string }>),
};
