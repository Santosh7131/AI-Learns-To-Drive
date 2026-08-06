import { useEffect, useRef, useState } from "react";
import { Flag, Boxes, Gauge, Play, Trophy, Waypoints, Cpu, ArrowDown, FlaskConical } from "lucide-react";
import { Arena, type HudStatSpec } from "@/components/Arena";
import { TopNav, REPO_URL } from "@/components/TopNav";
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
              onGoLive();
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

  const device = ready?.device ?? (gpu ? "Browser · GPU" : "Browser · CPU");
  const stepsPerSec = pb?.stepsPerSec ?? 0;
  const modelStep = ready?.modelStep ?? pb?.modelStep ?? 0;

  const hud: HudStatSpec[] = [
    { icon: <Flag className="h-3 w-3" />, label: "Laps", value: String(pb?.totalLaps ?? 0) },
    { icon: <Gauge className="h-3 w-3" />, label: "Steps/s", value: stepsPerSec.toFixed(0), accent: true },
    { icon: <Boxes className="h-3 w-3" />, label: "Cars", value: String(fleet) },
  ];

  return (
    <div id="top" className="min-h-full">
      <TopNav
        right={
          hasBackend ? (
            <button
              onClick={onGoLive}
              className="mr-1 hidden items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground sm:flex"
            >
              <FlaskConical className="h-3.5 w-3.5" /> Training console
            </button>
          ) : undefined
        }
      />

      <main className="mx-auto max-w-6xl px-4 sm:px-6">
        {/* hero */}
        <section className="pb-8 pt-14 sm:pt-20">
          <div className="inline-flex items-center gap-2 rounded-full border bg-secondary/50 px-3 py-1 text-xs text-muted-foreground">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full rounded-full bg-live opacity-60 animate-livepulse" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-live" />
            </span>
            Live · runs entirely in your browser
          </div>
          <h1 className="mt-5 max-w-3xl text-balance text-4xl font-extrabold leading-[1.05] tracking-tight sm:text-6xl">
            An AI that taught itself to drive.
          </h1>
          <p className="mt-5 max-w-2xl text-pretty text-base leading-relaxed text-muted-foreground sm:text-lg">
            No hand-coded rules. A Transformer policy trained with reinforcement learning discovered how to
            steer, accelerate and brake — and the exact trained network is driving below, live, on your own
            hardware.
          </p>
          <div className="mt-7 flex flex-wrap items-center gap-3">
            <a
              href="#demo"
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
            >
              <Play className="h-4 w-4" /> Watch it drive
            </a>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-medium transition-colors hover:bg-accent"
            >
              How it works <ArrowDown className="h-4 w-4" />
            </a>
          </div>
        </section>

        {/* live demo */}
        <section id="demo" className="scroll-mt-20 pb-16">
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="h-[58vh] min-h-[420px]">
              {track ? (
                <Arena track={track} telemetryRef={telemetryRef} numCars={fleet} hud={hud} />
              ) : (
                <div className="flex h-full items-center justify-center rounded-xl border bg-[#0a0e16] text-sm text-white/60">
                  <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />
                  <span className="ml-2">Loading the trained model…</span>
                </div>
              )}
            </div>
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
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
            Every car runs the same policy. Grey cars have gone off-track and are resetting. Click any car to
            inspect the neural network's live steering / throttle / brake outputs and its lidar view.
          </p>
        </section>

        {/* how it works */}
        <section className="border-t py-14">
          <h2 className="text-2xl font-bold tracking-tight">How it works</h2>
          <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-3">
            {[
              {
                icon: <Trophy className="h-5 w-5" />,
                title: "Learns by trial and error",
                body: "With PPO reinforcement learning, cars are rewarded for making progress and penalised for leaving the track. Good driving emerges from millions of attempts — none of it is scripted.",
              },
              {
                icon: <Waypoints className="h-5 w-5" />,
                title: "A Transformer at the wheel",
                body: "The policy attends over a short history of lidar rays and motion to choose continuous steering, throttle and brake — the same architecture behind modern language models.",
              },
              {
                icon: <Cpu className="h-5 w-5" />,
                title: "Runs on your hardware",
                body: "The trained network was ported to run in the browser — batched on your GPU via WebGPU, or on the CPU — and validated to match the original PyTorch model to five decimals.",
              },
            ].map((c) => (
              <div key={c.title} className="rounded-xl border bg-card p-5">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg border bg-secondary/50 text-brand">
                  {c.icon}
                </div>
                <h3 className="mt-4 text-base font-semibold">{c.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{c.body}</p>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer className="border-t">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 py-8 text-sm text-muted-foreground sm:flex-row sm:px-6">
          <div>AI Learns To Drive — a reinforcement-learning project.</div>
          <div className="flex items-center gap-4">
            <span className="text-xs">React · Three.js · PyTorch · WebGPU</span>
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="font-medium text-foreground hover:underline">
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
