/**
 * Playback data sources + capability detection.
 *
 * LocalSimSource runs the in-browser worker simulation and streams telemetry
 * frames into a ref (so the existing 3D/2D renderers work unchanged). The app
 * falls back to the server SSE stream only when a backend is reachable and the
 * device can't sustain local simulation.
 */
import type { Telemetry, TrackOption } from "@/lib/api";
import { API_BASE } from "@/lib/api";
import type { PlaybackMetrics, SimTrack, WorkerIn, WorkerOut } from "./types";
import type { PresetId } from "./presets";

const WEB_BASE = `${import.meta.env.BASE_URL}web/`;

export async function loadTrackList(): Promise<TrackOption[]> {
  return fetch(`${WEB_BASE}tracks.json`).then((r) => r.json());
}

export async function loadSimTrack(id: string): Promise<SimTrack> {
  return fetch(`${WEB_BASE}track-${id}.json`).then((r) => r.json());
}

/** Is a training backend reachable? (short timeout; false on static deploys) */
export async function backendAvailable(timeoutMs = 1500): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(`${API_BASE}/status`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return false;
    // A static host with an SPA rewrite returns index.html (200) for /api/*;
    // require a genuine JSON status so we don't mistake that for a backend.
    if (!(res.headers.get("content-type") || "").includes("application/json")) return false;
    const j = await res.json().catch(() => null);
    return !!j && (typeof j.status === "string" || typeof j.device === "string");
  } catch {
    return false;
  }
}

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export interface SystemScore {
  score: number; // ~0..100, rough
  tier: "low" | "medium" | "high" | "ultra";
  cores: number;
  memoryGB: number | null;
  webgpu: boolean;
  suggestedPreset: PresetId; // capped at "high" on CPU (ultra awaits WebGPU)
}

/** A quick, honest capability estimate: a short arithmetic micro-benchmark
 * combined with core count / memory / WebGPU presence. Used to preselect a
 * preset and to inform the user — it is a rough guide, not a guarantee. */
export function benchmarkSystem(): SystemScore {
  const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
  const memoryGB =
    typeof navigator !== "undefined" && "deviceMemory" in navigator
      ? (navigator as unknown as { deviceMemory: number }).deviceMemory
      : null;
  const webgpu = hasWebGPU();

  // micro-benchmark: iterations of transcendental math per ~40ms
  const start = performance.now();
  let acc = 0;
  let iters = 0;
  while (performance.now() - start < 40) {
    for (let i = 0; i < 20000; i++) acc += Math.sin(i * 0.001 + acc) * Math.exp(-((i % 7) * 0.01));
    iters += 20000;
  }
  const elapsed = performance.now() - start || 1;
  const mops = iters / elapsed / 1000; // million ops/sec (rough)
  void acc;

  // combine into a 0..100 score (weighted toward raw throughput)
  const throughputScore = Math.min(70, (mops / 40) * 70); // ~40 Mops => ~70
  const coreScore = Math.min(20, (cores / 12) * 20);
  const gpuScore = webgpu ? 10 : 0;
  const score = Math.round(throughputScore + coreScore + gpuScore);

  const tier: SystemScore["tier"] =
    score >= 80 ? "ultra" : score >= 60 ? "high" : score >= 35 ? "medium" : "low";
  // With WebGPU the fleet runs on the GPU, so higher tiers are reachable; on CPU
  // cap the auto-pick at High. Ultra stays opt-in either way.
  const suggestedPreset: PresetId = webgpu
    ? score >= 55 ? "high" : score >= 30 ? "medium" : "low"
    : tier === "low" ? "low" : tier === "medium" ? "medium" : "high";
  return { score, tier, cores, memoryGB, webgpu, suggestedPreset };
}

export interface LocalSimOptions {
  trackData: SimTrack;
  numEnvs: number;
  stepsPerSec: number;
  seed?: number;
  onFrame: (t: Telemetry) => void;
  onMetrics: (m: PlaybackMetrics) => void;
  onReady?: (info: { device: string; modelStep: number; bestReturn: number }) => void;
  onError?: (msg: string) => void;
}

export class LocalSimSource {
  private worker: Worker;
  private stopped = false;

  constructor(private opts: LocalSimOptions) {
    this.worker = new Worker(new URL("./sim.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<WorkerOut>) => {
      const m = ev.data;
      if (m.type === "frame") opts.onFrame(m.telemetry);
      else if (m.type === "metrics") opts.onMetrics(m.metrics);
      else if (m.type === "ready") opts.onReady?.(m);
      else if (m.type === "error") opts.onError?.(m.message);
    };
    this.post({
      type: "init",
      trackData: opts.trackData,
      policyUrl: `${WEB_BASE}policy.json`,
      numEnvs: opts.numEnvs,
      seed: opts.seed ?? 1,
      stepsPerSec: opts.stepsPerSec,
    });
  }

  private post(m: WorkerIn) {
    if (!this.stopped) this.worker.postMessage(m);
  }

  setFleet(numEnvs: number) {
    this.post({ type: "config", numEnvs });
  }
  setRate(stepsPerSec: number) {
    this.post({ type: "config", stepsPerSec });
  }
  pause() {
    this.post({ type: "pause" });
  }
  resume() {
    this.post({ type: "resume" });
  }
  stop() {
    this.post({ type: "stop" });
    this.stopped = true;
    this.worker.terminate();
  }
}
