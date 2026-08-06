import { useEffect, useRef, useState } from "react";
import { Cpu, Sparkles, Flag, Boxes, Gauge, FlaskConical } from "lucide-react";
import { Arena, type HudStatSpec } from "@/components/Arena";
import { Pill } from "@/components/StatusPill";
import { PlaybackControls } from "@/components/PlaybackControls";
import {
  LocalSimSource,
  benchmarkSystem,
  loadSimTrack,
  hasWebGPU,
  type SystemScore,
} from "@/sim/source";
import { presetById, presetsFor, type PresetId } from "@/sim/presets";
import type { PlaybackMetrics, SimTrack } from "@/sim/types";
import type { Telemetry } from "@/lib/api";

interface Props {
  hasBackend: boolean;
  onGoLive: () => void;
}

export default function PlaybackApp({ hasBackend, onGoLive }: Props) {
  const [track, setTrack] = useState<SimTrack | null>(null);
  const [pb, setPb] = useState<PlaybackMetrics | null>(null);
  const [ready, setReady] = useState<{ device: string; modelStep: number; bestReturn: number } | null>(null);
  const [score, setScore] = useState<SystemScore | null>(null);
  const [preset, setPreset] = useState<PresetId>("medium");
  const [fleet, setFleet] = useState(15);
  const [gpu, setGpu] = useState<boolean>(() => hasWebGPU());
  const [running, setRunning] = useState(false);
  const [serverFallback, setServerFallback] = useState(false);
  const telemetryRef = useRef<Telemetry | null>(null);
  const srcRef = useRef<LocalSimSource | null>(null);

  // benchmark once, load the trained policy + track, start the local sim
  useEffect(() => {
    const s = benchmarkSystem();
    setScore(s);
    const gpuGuess = hasWebGPU();
    const p0 = s.suggestedPreset;
    const cfg0 = presetById(p0, gpuGuess);
    setPreset(p0);
    setFleet(cfg0.fleet);

    let src: LocalSimSource | null = null;
    let alive = true;
    loadSimTrack("default")
      .then((td) => {
        if (!alive) return;
        setTrack(td);
        src = new LocalSimSource({
          trackData: td,
          numEnvs: cfg0.fleet,
          stepsPerSec: cfg0.stepsPerSec,
          onFrame: (t) => {
            telemetryRef.current = t;
          },
          onMetrics: setPb,
          onReady: (info) => {
            setReady(info);
            // the worker reports whether it actually got a GPU device; if that
            // differs from our guess, switch to the correct preset set + fleet.
            const actual = /gpu/i.test(info.device);
            if (actual !== gpuGuess) {
              setGpu(actual);
              const cfg = presetById(p0, actual);
              setFleet(cfg.fleet);
              src?.setFleet(cfg.fleet);
              src?.setRate(cfg.stepsPerSec);
            }
          },
          onError: (msg) => {
            console.error("[sim] worker error:", msg);
            if (hasBackend) {
              setServerFallback(true);
              onGoLive(); // hand off to the server-streamed live view
            }
          },
        });
        srcRef.current = src;
        setRunning(true);
      })
      .catch((e) => console.error("failed to load playground assets", e));

    return () => {
      alive = false;
      src?.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const changePreset = (id: PresetId) => {
    const cfg = presetById(id, gpu);
    setPreset(id);
    setFleet(cfg.fleet);
    srcRef.current?.setFleet(cfg.fleet);
    srcRef.current?.setRate(cfg.stepsPerSec);
  };

  const toggleRun = () => {
    setRunning((r) => {
      const next = !r;
      if (next) srcRef.current?.resume();
      else srcRef.current?.pause();
      return next;
    });
  };

  const device = ready?.device ?? "Browser · CPU";
  const stepsPerSec = pb?.stepsPerSec ?? 0;
  const modelStep = ready?.modelStep ?? pb?.modelStep ?? 0;

  const hud: HudStatSpec[] = [
    { icon: <Flag className="h-3 w-3" />, label: "Laps", value: String(pb?.totalLaps ?? 0), accent: "green" },
    { icon: <Gauge className="h-3 w-3" />, label: "Steps/s", value: stepsPerSec.toFixed(0), accent: "cyan" },
    { icon: <Boxes className="h-3 w-3" />, label: "Cars", value: String(fleet) },
  ];

  return (
    <div className="flex h-full flex-col">
      {/* header */}
      <header className="glass sticky top-0 z-20 flex items-center justify-between gap-3 rounded-none border-x-0 border-t-0 px-4 py-2.5 sm:px-5">
        <div className="flex items-center gap-3">
          <div className="glow-primary flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary/25 to-primary/5 text-primary">
            <Sparkles className="h-5 w-5" />
          </div>
          <div className="leading-tight">
            <h1 className="text-[15px] font-semibold tracking-tight sm:text-base">Reinforcement Car</h1>
            <p className="hidden text-xs text-muted-foreground sm:block">
              Transformer policy · running in your browser
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Pill className="text-primary">
            <Cpu className="h-3.5 w-3.5" />
            <span className="font-semibold">{device}</span>
          </Pill>
          {score && (
            <Pill className="hidden font-mono text-muted-foreground sm:flex">
              score {score.score}
            </Pill>
          )}
          <Pill className={running ? "text-primary" : "text-amber-400"}>
            <span className={`h-2 w-2 rounded-full ${running ? "bg-primary animate-livepulse" : "bg-amber-400"}`} />
            <span className="font-medium">{running ? "Live" : "Paused"}</span>
          </Pill>
          {hasBackend && (
            <button
              onClick={onGoLive}
              title="Switch to live training (backend)"
              className="glass-hud flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-white/70 transition-colors hover:text-white"
            >
              <FlaskConical className="h-3.5 w-3.5" /> Train
            </button>
          )}
        </div>
      </header>

      {/* main */}
      <main className="grid min-h-0 min-w-0 flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_380px]">
        {track ? (
          <Arena
            track={track}
            telemetryRef={telemetryRef}
            numCars={fleet}
            hud={hud}
            caption="Every car is driven by the same Transformer policy, simulated live on your device. Grey = off-track (resetting); click a car to inspect its neural-net outputs."
          />
        ) : (
          <div className="flex min-h-0 items-center justify-center rounded-2xl border bg-[#070a12] text-sm text-muted-foreground">
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-primary/30 border-t-primary" />
            <span className="ml-2">Loading the trained model…</span>
          </div>
        )}

        <aside className="scroll-slim flex min-h-0 flex-col gap-4 overflow-y-auto pb-1">
          <div className="animate-rise" style={{ animationDelay: "40ms" }}>
            <PlaybackControls
              presets={presetsFor(gpu)}
              preset={preset}
              onPreset={changePreset}
              running={running}
              onToggleRun={toggleRun}
              score={score}
              device={device}
              stepsPerSec={stepsPerSec}
              modelStep={modelStep}
              serverFallback={serverFallback}
            />
          </div>
        </aside>
      </main>
    </div>
  );
}
