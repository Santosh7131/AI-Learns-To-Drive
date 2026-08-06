export type PresetId = "p5" | "p10" | "p20" | "custom";

export interface Preset {
  id: PresetId;
  label: string;
  fleet: number;
}

// Small, watchable fleets. These are all <= the instancing threshold, so the
// detailed car model is what you see.
export const PRESETS: Preset[] = [
  { id: "p5", label: "5", fleet: 5 },
  { id: "p10", label: "10", fleet: 10 },
  { id: "p20", label: "20", fleet: 20 },
];

export const CUSTOM_MIN = 1;
export const CUSTOM_MAX = 120;
export const CUSTOM_DEFAULT = 40;

// sim step-rate target (physics dt = 0.1s → ~1.8x real-time, lively but smooth)
export const STEPS_PER_SEC = 18;

export const presetFleet = (id: PresetId, custom: number) =>
  id === "custom" ? custom : PRESETS.find((p) => p.id === id)?.fleet ?? 10;
