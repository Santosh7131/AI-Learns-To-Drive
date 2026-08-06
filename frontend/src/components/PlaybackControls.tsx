import { Play, Pause, Cpu, Gauge, Sparkles, Boxes } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Preset, PresetId } from "@/sim/presets";
import type { SystemScore } from "@/sim/source";

interface Props {
  presets: Preset[];
  preset: PresetId;
  onPreset: (id: PresetId) => void;
  running: boolean;
  onToggleRun: () => void;
  score: SystemScore | null;
  device: string;
  stepsPerSec: number;
  modelStep: number;
  serverFallback: boolean;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}

const TIER_COLOR: Record<string, string> = {
  low: "text-muted-foreground",
  medium: "text-data",
  high: "text-primary",
  ultra: "text-primary",
};

export function PlaybackControls({
  presets,
  preset,
  onPreset,
  running,
  onToggleRun,
  score,
  device,
  stepsPerSec,
  modelStep,
  serverFallback,
}: Props) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold">
          <Sparkles className="h-4 w-4 text-primary" /> Playground
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* compute mode */}
        <div className={`hairline rounded-xl p-3 ${serverFallback ? "bg-amber-500/10" : "bg-primary/10"}`}>
          <div className="flex items-center gap-2">
            <Cpu className={`h-4 w-4 ${serverFallback ? "text-amber-400" : "text-primary"}`} />
            <span className="text-sm font-medium">{device}</span>
          </div>
          <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
            {serverFallback
              ? "This device is too weak for local simulation — streaming from the server instead."
              : "The cars are simulated live in your browser — the neural net runs on your machine, not a server."}
          </p>
        </div>

        {/* system score */}
        {score && (
          <div className="space-y-1.5">
            <SectionLabel>
              <Gauge className="h-3.5 w-3.5" /> System score
            </SectionLabel>
            <div className="flex items-center gap-3">
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-data to-primary transition-all duration-500"
                  style={{ width: `${Math.max(4, Math.min(100, score.score))}%` }}
                />
              </div>
              <span className={`font-mono text-sm font-semibold tabular-nums ${TIER_COLOR[score.tier]}`}>
                {score.score}
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground">
              {score.cores} cores{score.memoryGB ? ` · ${score.memoryGB} GB` : ""} ·{" "}
              {score.webgpu ? "WebGPU available" : "no WebGPU"} · rough estimate
            </p>
          </div>
        )}

        {/* preset picker */}
        <div className="space-y-2">
          <SectionLabel>
            <Boxes className="h-3.5 w-3.5" /> Quality preset
          </SectionLabel>
          <div className="grid grid-cols-2 gap-2">
            {presets.map((p) => {
              const active = p.id === preset;
              const suggested = score?.suggestedPreset === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => onPreset(p.id)}
                  className={`hairline rounded-xl p-2.5 text-left transition-all ${
                    active
                      ? "bg-primary/15 shadow-glow ring-1 ring-primary/60"
                      : "bg-secondary/40 hover:bg-secondary/70"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold">{p.label}</span>
                    {suggested && !active && (
                      <span className="rounded bg-data/20 px-1 text-[9px] font-medium uppercase text-data">fit</span>
                    )}
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">{p.fleet} cars</div>
                </button>
              );
            })}
          </div>
          <p className="text-[11px] leading-snug text-muted-foreground">
            {presets.find((p) => p.id === preset)?.blurb}
          </p>
        </div>

        {/* transport */}
        <Button onClick={onToggleRun} className="w-full shadow-glow" variant={running ? "secondary" : "default"}>
          {running ? <Pause /> : <Play />} {running ? "Pause" : "Play"}
        </Button>

        {/* live readouts */}
        <div className="grid grid-cols-2 gap-2">
          <div className="hairline rounded-xl bg-secondary/40 p-2.5">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Sim speed</div>
            <div className="mt-0.5 font-mono text-sm font-semibold tabular-nums text-data">
              {stepsPerSec.toFixed(0)}<span className="ml-1 text-[10px] text-muted-foreground">steps/s</span>
            </div>
          </div>
          <div className="hairline rounded-xl bg-secondary/40 p-2.5">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Model</div>
            <div className="mt-0.5 font-mono text-sm font-semibold tabular-nums">
              {(modelStep / 1e6).toFixed(1)}M<span className="ml-1 text-[10px] text-muted-foreground">steps</span>
            </div>
          </div>
        </div>

        <p className="text-[11px] leading-snug text-muted-foreground">
          These cars drive with a Transformer policy trained with reinforcement learning.
          {device.includes("GPU")
            ? " The whole fleet's neural net runs batched on your GPU via WebGPU."
            : " Your browser has no WebGPU, so the fleet runs on the CPU."}
        </p>
      </CardContent>
    </Card>
  );
}
