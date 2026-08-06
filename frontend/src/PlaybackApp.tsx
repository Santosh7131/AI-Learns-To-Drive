import { useEffect, useRef, useState } from "react";
import { Flag, Gauge, Boxes, Play, Trophy, Waypoints, Cpu, ArrowDown, FlaskConical } from "lucide-react";
import { Arena, type HudStatSpec } from "@/components/Arena";
import { TopNav, REPO_URL } from "@/components/TopNav";
import { PlaybackControls } from "@/components/PlaybackControls";
import {
  LocalSimSource,
  benchmarkSystem,
  loadSimTrack,
  type SystemScore,
} from "@/sim/source";
import { presetFleet, CUSTOM_DEFAULT, STEPS_PER_SEC, type PresetId } from "@/sim/presets";
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
  const [preset, setPreset] = useState<PresetId>("p10");
  const [customFleet, setCustomFleet] = useState(CUSTOM_DEFAULT);
  const [running, setRunning] = useState(false);
  const [serverFallback, setServerFallback] = useState(false);
  const [fs, setFs] = useState(false);
  const [speed, setSpeed] = useState(1);
  const telemetryRef = useRef<Telemetry | null>(null);
  const srcRef = useRef<LocalSimSource | null>(null);
  const demoRef = useRef<HTMLDivElement>(null);

  const fleet = presetFleet(preset, customFleet);

  useEffect(() => {
    const s = benchmarkSystem();
    setScore(s);
    const p0 = s.suggestedPreset;
    setPreset(p0);
    const f0 = presetFleet(p0, CUSTOM_DEFAULT);

    let src: LocalSimSource | null = null;
    let alive = true;
    loadSimTrack("default")
      .then((td) => {
        if (!alive) return;
        setTrack(td);
        src = new LocalSimSource({
          trackData: td,
          numEnvs: f0,
          stepsPerSec: STEPS_PER_SEC,
          onFrame: (t) => {
            telemetryRef.current = t;
          },
          onMetrics: setPb,
          onReady: setReady,
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

  // track native fullscreen state
  useEffect(() => {
    const onChange = () => setFs(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const changePreset = (id: PresetId) => {
    setPreset(id);
    srcRef.current?.setFleet(presetFleet(id, customFleet));
  };
  const changeCustom = (n: number) => {
    setCustomFleet(n);
    if (preset === "custom") srcRef.current?.setFleet(n);
  };
  const toggleRun = () => {
    setRunning((r) => {
      const next = !r;
      if (next) srcRef.current?.resume();
      else srcRef.current?.pause();
      return next;
    });
  };
  const reset = () => srcRef.current?.reset();
  const changeSpeed = (n: number) => {
    setSpeed(n);
    srcRef.current?.setRate(STEPS_PER_SEC * n);
  };
  const toggleFullscreen = () => {
    const el = demoRef.current;
    if (!el) return;
    if (!document.fullscreenElement) el.requestFullscreen?.().catch(() => setFs(true));
    else document.exitFullscreen?.();
  };

  const device = ready?.device ?? "Browser";
  const stepsPerSec = pb?.stepsPerSec ?? 0;
  const modelStep = ready?.modelStep ?? pb?.modelStep ?? 0;

  const hud: HudStatSpec[] = [
    { icon: <Flag className="h-3 w-3" />, label: "Laps", value: String(pb?.totalLaps ?? 0) },
    { icon: <Gauge className="h-3 w-3" />, label: "Steps/s", value: stepsPerSec.toFixed(0), accent: true },
    { icon: <Boxes className="h-3 w-3" />, label: "Cars", value: String(fleet) },
  ];

  const controls = (
    <PlaybackControls
      preset={preset}
      onPreset={changePreset}
      customFleet={customFleet}
      onCustomFleet={changeCustom}
      fleet={fleet}
      running={running}
      onToggleRun={toggleRun}
      onReset={reset}
      speed={speed}
      onSpeed={changeSpeed}
      score={score}
      device={device}
      stepsPerSec={stepsPerSec}
      modelStep={modelStep}
      serverFallback={serverFallback}
    />
  );

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
            <button
              onClick={() => {
                document.getElementById("demo")?.scrollIntoView({ behavior: "smooth" });
                toggleFullscreen();
              }}
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
            >
              <Play className="h-4 w-4" /> Watch it drive
            </button>
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-medium transition-colors hover:bg-accent">
              How it works <ArrowDown className="h-4 w-4" />
            </a>
          </div>
        </section>

        {/* live demo (this subtree is what goes fullscreen) */}
        <section id="demo" className="scroll-mt-20 pb-16">
          <div
            ref={demoRef}
            className={
              fs
                ? "grid h-screen grid-cols-1 gap-4 bg-background p-4 lg:grid-cols-[minmax(0,1fr)_340px]"
                : "grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]"
            }
          >
            <div className={fs ? "min-h-0" : "h-[58vh] min-h-[420px]"}>
              {track ? (
                <Arena
                  track={track}
                  telemetryRef={telemetryRef}
                  numCars={fleet}
                  hud={hud}
                  active={running}
                  fullscreen={fs}
                  onToggleFullscreen={toggleFullscreen}
                />
              ) : (
                <div className="flex h-full items-center justify-center rounded-xl border bg-[#0a0e16] text-sm text-white/60">
                  <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />
                  <span className="ml-2">Loading the trained model…</span>
                </div>
              )}
            </div>
            <div className={fs ? "scroll-slim overflow-y-auto" : ""}>{controls}</div>
          </div>
          {!fs && (
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
              Every car runs the same policy. Grey cars have gone off-track and are resetting. Click any car to
              inspect the neural network's live steering / throttle / brake outputs and its lidar view. Drag to
              orbit, or use the racing-line and fullscreen controls on the scene.
            </p>
          )}
        </section>

        {/* how it works */}
        <section className="border-t py-14">
          <h2 className="text-2xl font-bold tracking-tight">How it works</h2>
          <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-3">
            {[
              { icon: <Trophy className="h-5 w-5" />, title: "Learns by trial and error", body: "With PPO reinforcement learning, cars are rewarded for making progress and penalised for leaving the track. Good driving emerges from millions of attempts — none of it is scripted." },
              { icon: <Waypoints className="h-5 w-5" />, title: "A Transformer at the wheel", body: "The policy attends over a short history of lidar rays and motion to choose continuous steering, throttle and brake — the same architecture behind modern language models." },
              { icon: <Cpu className="h-5 w-5" />, title: "Runs on your hardware", body: "The trained network was ported to run in the browser — batched on your GPU via WebGPU, or on the CPU — and validated to match the original PyTorch model to five decimals." },
            ].map((c) => (
              <div key={c.title} className="rounded-xl border bg-card p-5">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg border bg-secondary/50 text-brand">{c.icon}</div>
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
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="font-medium text-foreground hover:underline">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
