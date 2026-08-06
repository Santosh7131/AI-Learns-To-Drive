export type PresetId = "low" | "medium" | "high" | "ultra";

export interface Preset {
  id: PresetId;
  label: string;
  fleet: number;
  stepsPerSec: number;
  blurb: string;
}

// CPU tier — the Transformer runs per-car in scalar JS on one worker thread, so
// smooth playback is a few dozen cars.
export const PRESETS_CPU: Preset[] = [
  { id: "low", label: "Low", fleet: 8, stepsPerSec: 12, blurb: "8 cars · runs on anything" },
  { id: "medium", label: "Medium", fleet: 15, stepsPerSec: 12, blurb: "15 cars · smooth on most machines" },
  { id: "high", label: "High", fleet: 25, stepsPerSec: 12, blurb: "25 cars · a strong CPU helps" },
  { id: "ultra", label: "Ultra", fleet: 40, stepsPerSec: 12, blurb: "40 cars · heavy on CPU" },
];

// GPU tier — the whole fleet's policy runs batched on the visitor's GPU (WebGPU),
// so hundreds of cars are feasible. (Physics is still CPU, which caps the very
// top tiers until GPU physics lands.)
export const PRESETS_GPU: Preset[] = [
  { id: "low", label: "Low", fleet: 60, stepsPerSec: 20, blurb: "60 cars · on your GPU" },
  { id: "medium", label: "Medium", fleet: 150, stepsPerSec: 20, blurb: "150 cars · on your GPU" },
  { id: "high", label: "High", fleet: 350, stepsPerSec: 20, blurb: "350 cars · needs a real GPU" },
  { id: "ultra", label: "Ultra", fleet: 750, stepsPerSec: 20, blurb: "750 cars · CPU physics caps the rate for now" },
];

export const presetsFor = (gpu: boolean) => (gpu ? PRESETS_GPU : PRESETS_CPU);
export const presetById = (id: PresetId, gpu: boolean) =>
  presetsFor(gpu).find((p) => p.id === id) ?? presetsFor(gpu)[0];
