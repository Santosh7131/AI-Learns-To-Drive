import { useState } from "react";
import { Play, Pause, RotateCcw, Cpu, Zap, ChevronDown, Gamepad2, FlaskConical, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { PRESETS, CUSTOM_MIN, CUSTOM_MAX, type PresetId } from "@/sim/presets";
import type { SystemScore } from "@/sim/source";
import type { TrainMetrics } from "@/sim/types";

interface Props {
  preset: PresetId;
  onPreset: (id: PresetId) => void;
  customFleet: number;
  onCustomFleet: (n: number) => void;
  fleet: number;
  running: boolean;
  onToggleRun: () => void;
  onReset: () => void;
  speed: number;
  onSpeed: (n: number) => void;
  manual: boolean;
  onManual: (v: boolean) => void;
  untrained: boolean;
  onUntrained: (v: boolean) => void;
  learning: boolean;
  onLearning: (v: boolean) => void;
  onResetBrain: () => void;
  train: TrainMetrics | null;
  score: SystemScore | null;
  device: string;
  stepsPerSec: number;
  modelStep: number;
  serverFallback: boolean;
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{children}</div>;
}

function MiniStat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-lg border bg-secondary/40 p-2.5">
      <Label>{label}</Label>
      <div className={`num mt-0.5 text-sm font-semibold ${accent ? "text-brand" : ""}`}>{value}</div>
    </div>
  );
}

/** tiny reward-over-time sparkline (render-only) */
function Sparkline({ data }: { data: number[] }) {
  const W = 280, H = 46;
  if (data.length < 2) {
    return (
      <div className="flex h-[46px] items-center justify-center rounded-md border bg-secondary/40 text-[11px] text-muted-foreground">
        gathering experience…
      </div>
    );
  }
  const min = Math.min(...data), max = Math.max(...data);
  const span = max - min || 1;
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * W},${H - 5 - ((v - min) / span) * (H - 10)}`)
    .join(" ");
  const zeroY = min < 0 && max > 0 ? H - 5 - ((0 - min) / span) * (H - 10) : null;
  return (
    <div className="rounded-md border bg-secondary/40 p-1.5 text-brand">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-[46px] w-full">
        {zeroY != null && (
          <line x1={0} y1={zeroY} x2={W} y2={zeroY} stroke="currentColor" strokeWidth={1} strokeDasharray="3 3" opacity={0.25} vectorEffect="non-scaling-stroke" />
        )}
        <polyline points={pts} fill="none" stroke="currentColor" strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
    </div>
  );
}

const fmt1 = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(1));

export function PlaybackControls({
  preset,
  onPreset,
  customFleet,
  onCustomFleet,
  fleet,
  running,
  onToggleRun,
  onReset,
  speed,
  onSpeed,
  manual,
  onManual,
  untrained,
  onUntrained,
  learning,
  onLearning,
  onResetBrain,
  train,
  score,
  device,
  stepsPerSec,
  modelStep,
  serverFallback,
}: Props) {
  const [advanced, setAdvanced] = useState(false);
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
                ? `Runs on your GPU via WebGPU${score ? ` · ${score.cores} cores` : ""}.`
                : "The fleet is simulated on your CPU (no WebGPU here)."}
            </p>
          </div>
        </div>

        {/* fleet size */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Cars</Label>
            <span className="num text-sm font-semibold">{fleet}</span>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => onPreset(p.id)}
                className={`rounded-lg border py-2 text-sm font-semibold transition-colors ${
                  preset === p.id ? "border-brand bg-brand/5 text-foreground" : "text-muted-foreground hover:bg-accent"
                }`}
              >
                {p.label}
              </button>
            ))}
            <button
              onClick={() => onPreset("custom")}
              className={`rounded-lg border py-2 text-xs font-semibold transition-colors ${
                preset === "custom" ? "border-brand bg-brand/5 text-foreground" : "text-muted-foreground hover:bg-accent"
              }`}
            >
              Custom
            </button>
          </div>
          {preset === "custom" && (
            <div className="pt-1">
              <Slider
                min={CUSTOM_MIN}
                max={CUSTOM_MAX}
                step={1}
                value={[customFleet]}
                onValueChange={([v]) => onCustomFleet(v)}
              />
              <div className="mt-1.5 flex justify-between text-[11px] text-muted-foreground">
                <span>{CUSTOM_MIN}</span>
                <span className="num text-foreground">{customFleet} cars</span>
                <span>{CUSTOM_MAX}</span>
              </div>
            </div>
          )}
        </div>

        {/* simulation speed */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Sim speed</Label>
            <span className="num text-sm font-semibold">{speed.toFixed(2).replace(/\.00$/, "")}×</span>
          </div>
          <Slider min={0.25} max={8} step={0.25} value={[speed]} onValueChange={([v]) => onSpeed(v)} />
          <p className="text-[11px] leading-snug text-muted-foreground">Fast-forward the simulation to skip ahead, or slow it down to study a corner.</p>
        </div>

        {/* transport */}
        <div className="grid grid-cols-2 gap-2">
          <Button onClick={onToggleRun} variant={running ? "outline" : "default"}>
            {running ? <Pause /> : <Play />} {running ? "Pause" : "Play"}
          </Button>
          <Button onClick={onReset} variant="outline">
            <RotateCcw /> Reset
          </Button>
        </div>

        {/* live readouts */}
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-lg border bg-secondary/40 p-2.5">
            <Label>Rate</Label>
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

        {/* advanced */}
        <div className="border-t pt-4">
          <button
            onClick={() => setAdvanced((a) => !a)}
            className="flex w-full items-center justify-between text-[11px] font-medium uppercase tracking-wide text-muted-foreground transition-colors hover:text-foreground"
          >
            Advanced
            <ChevronDown className={`h-4 w-4 transition-transform ${advanced ? "rotate-180" : ""}`} />
          </button>
          {advanced && (
            <div className="mt-3 space-y-3">
              {/* learn from scratch — real in-browser reinforcement learning */}
              <div className={`rounded-lg border ${learning ? "border-brand bg-brand/5" : ""}`}>
                <button onClick={() => onLearning(!learning)} className="flex w-full items-center gap-2.5 p-3 text-left">
                  <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${learning ? "bg-brand/10 text-brand" : "bg-muted text-muted-foreground"}`}>
                    <Brain className="h-4 w-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium">Train a new brain (live)</div>
                    <div className="text-[11px] leading-snug text-muted-foreground">
                      {learning
                        ? "A blank network is learning to drive right now — reward should climb."
                        : "Wipe the brain to zero and watch a car learn to drive from scratch, live."}
                    </div>
                  </div>
                  <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${learning ? "border-brand/40 text-brand" : "text-muted-foreground"}`}>
                    {learning ? "ON" : "OFF"}
                  </span>
                </button>
                {learning && (
                  <div className="space-y-3 px-3 pb-3">
                    <Sparkline data={train?.history ?? []} />
                    <div className="grid grid-cols-2 gap-2">
                      <MiniStat label="Avg reward" value={fmt1(train?.avgReturn)} accent />
                      <MiniStat label="Best" value={fmt1(train?.bestReturn)} />
                      <MiniStat label="Updates" value={String(train?.updates ?? 0)} />
                      <MiniStat label="Laps" value={String(train?.laps ?? 0)} />
                    </div>
                    <Button onClick={onResetBrain} variant="outline" className="w-full">
                      <RotateCcw /> Reset brain — start from zero
                    </Button>
                    <p className="text-[11px] leading-snug text-muted-foreground">
                      A small network trained by PPO in your browser — not the shipped Transformer. Raise{" "}
                      <span className="font-medium text-foreground">Sim speed</span> to fast-forward the learning, or drop it to watch closely.
                    </p>
                  </div>
                )}
              </div>

              <button
                onClick={() => onManual(!manual)}
                className={`flex w-full items-center gap-2.5 rounded-lg border p-3 text-left transition-colors ${manual ? "border-brand bg-brand/5" : "hover:bg-accent"}`}
              >
                <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${manual ? "bg-brand/10 text-brand" : "bg-muted text-muted-foreground"}`}>
                  <Gamepad2 className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-medium">Drive a car yourself</div>
                  <div className="text-[11px] text-muted-foreground">
                    {manual ? "You're driving — chase cam on." : "Take the wheel among the AI cars."}
                  </div>
                </div>
              </button>
              {manual && (
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                  <span><kbd className="num rounded bg-secondary px-1">W</kbd>/<kbd className="num rounded bg-secondary px-1">↑</kbd> throttle</span>
                  <span><kbd className="num rounded bg-secondary px-1">S</kbd>/<kbd className="num rounded bg-secondary px-1">↓</kbd> brake</span>
                  <span><kbd className="num rounded bg-secondary px-1">A</kbd><kbd className="num rounded bg-secondary px-1">D</kbd> steer</span>
                </div>
              )}

              <button
                onClick={() => onUntrained(!untrained)}
                className={`flex w-full items-center gap-2.5 rounded-lg border p-3 text-left transition-colors ${untrained ? "border-brand bg-brand/5" : "hover:bg-accent"}`}
              >
                <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${untrained ? "bg-brand/10 text-brand" : "bg-muted text-muted-foreground"}`}>
                  <FlaskConical className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-medium">Untrained network</div>
                  <div className="text-[11px] leading-snug text-muted-foreground">
                    {untrained
                      ? "Random flailing — the AI before any training. Toggle off for the trained driver."
                      : "See what the car does before it learns to drive."}
                  </div>
                </div>
              </button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
