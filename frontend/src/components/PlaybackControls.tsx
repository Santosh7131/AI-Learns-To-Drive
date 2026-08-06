import { Play, Pause, Cpu, Zap } from "lucide-react";
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

function Label({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{children}</div>;
}

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
  const onGpu = device.includes("GPU");
  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold">Simulation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* compute mode */}
        <div className="flex items-start gap-2.5 rounded-lg border bg-secondary/40 p-3">
          <div className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${onGpu ? "bg-brand/10 text-brand" : "bg-muted text-muted-foreground"}`}>
            {onGpu ? <Zap className="h-4 w-4" /> : <Cpu className="h-4 w-4" />}
          </div>
          <div className="min-w-0">
            <div className="text-sm font-medium">{device}</div>
            <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
              {serverFallback
                ? "Device too weak for local sim — streaming from the server."
                : onGpu
                ? "The whole fleet's neural net runs on your GPU via WebGPU."
                : "The fleet is simulated on your CPU (no WebGPU here)."}
            </p>
          </div>
        </div>

        {/* system score */}
        {score && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label>System score</Label>
              <span className="num text-sm font-semibold">{score.score}</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <div className="h-full rounded-full bg-brand transition-all duration-500" style={{ width: `${Math.max(4, Math.min(100, score.score))}%` }} />
            </div>
            <p className="text-[11px] text-muted-foreground">
              {score.cores} cores{score.memoryGB ? ` · ${score.memoryGB} GB` : ""} · {score.webgpu ? "WebGPU" : "no WebGPU"}
            </p>
          </div>
        )}

        {/* preset picker */}
        <div className="space-y-2">
          <Label>Quality preset</Label>
          <div className="grid grid-cols-2 gap-2">
            {presets.map((p) => {
              const active = p.id === preset;
              const fit = score?.suggestedPreset === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => onPreset(p.id)}
                  className={`rounded-lg border p-2.5 text-left transition-colors ${
                    active ? "border-brand bg-brand/5" : "hover:bg-accent"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold">{p.label}</span>
                    {fit && !active && (
                      <span className="rounded bg-muted px-1 text-[9px] font-medium uppercase text-muted-foreground">fit</span>
                    )}
                  </div>
                  <div className="num mt-0.5 text-[11px] text-muted-foreground">{p.fleet} cars</div>
                </button>
              );
            })}
          </div>
          <p className="text-[11px] leading-snug text-muted-foreground">{presets.find((p) => p.id === preset)?.blurb}</p>
        </div>

        <Button onClick={onToggleRun} variant={running ? "outline" : "default"} className="w-full">
          {running ? <Pause /> : <Play />} {running ? "Pause" : "Play"}
        </Button>

        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-lg border bg-secondary/40 p-2.5">
            <Label>Sim speed</Label>
            <div className="num mt-0.5 text-sm font-semibold">
              {stepsPerSec.toFixed(0)}<span className="ml-1 text-[10px] font-normal text-muted-foreground">steps/s</span>
            </div>
          </div>
          <div className="rounded-lg border bg-secondary/40 p-2.5">
            <Label>Model</Label>
            <div className="num mt-0.5 text-sm font-semibold">
              {(modelStep / 1e6).toFixed(1)}M<span className="ml-1 text-[10px] font-normal text-muted-foreground">steps</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
