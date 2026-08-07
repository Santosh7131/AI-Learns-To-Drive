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

/** live summary of an in-browser training run (the "learn from scratch" mode) */
export interface TrainMetrics {
  learning: boolean;
  updates: number;     // PPO updates so far
  envSteps: number;    // environment steps collected
  avgReturn: number;   // mean episode return over a recent window
  bestReturn: number;  // best recent-window mean seen this run
  entropy: number;     // policy entropy (exploration level)
  laps: number;        // laps completed this run
  history: number[];   // recent avgReturn samples (for a sparkline)
}

/** the geometry the renderers need + the extra arrays the physics needs */
export type SimTrack = TrackGeometry & TrackData;

export type WorkerIn =
  | { type: "init"; trackData: SimTrack; policyUrl: string; numEnvs: number; seed: number; stepsPerSec: number }
  | { type: "config"; numEnvs?: number; stepsPerSec?: number }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "reset" }
  | { type: "setPlayer"; index: number } // -1 = none; else this car is human-driven
  | { type: "playerInput"; steer: number; accel: number; brake: number }
  | { type: "setUntrained"; value: boolean } // random policy (the "before learning" state)
  | { type: "setLearning"; value: boolean }  // live RL: train a brain from scratch in-browser
  | { type: "resetBrain" }                   // wipe the learning brain -> start over from zero
  | { type: "stop" };

export type WorkerOut =
  | { type: "ready"; device: string; modelStep: number; bestReturn: number }
  | { type: "frame"; telemetry: Telemetry }
  | { type: "metrics"; metrics: PlaybackMetrics }
  | { type: "train"; metrics: TrainMetrics }
  | { type: "error"; message: string };
