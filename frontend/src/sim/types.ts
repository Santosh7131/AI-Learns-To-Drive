import type { Telemetry, TrackGeometry } from "@/lib/api";
import type { TrackData } from "./env";

/** live summary of the in-browser playback (analogous to training Metrics) */
export interface PlaybackMetrics {
  status: "running" | "paused";
  numEnvs: number;
  stepsPerSec: number;
  totalLaps: number;
  modelStep: number;
  bestReturn: number;
  device: string; // e.g. "Browser · CPU"
  track: string;
}

/** the geometry the renderers need + the extra arrays the physics needs */
export type SimTrack = TrackGeometry & TrackData;

export type WorkerIn =
  | { type: "init"; trackData: SimTrack; policyUrl: string; numEnvs: number; seed: number; stepsPerSec: number }
  | { type: "config"; numEnvs?: number; stepsPerSec?: number }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "stop" };

export type WorkerOut =
  | { type: "ready"; device: string; modelStep: number; bestReturn: number }
  | { type: "frame"; telemetry: Telemetry }
  | { type: "metrics"; metrics: PlaybackMetrics }
  | { type: "error"; message: string };
