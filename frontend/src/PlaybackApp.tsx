import { useEffect, useRef, useState } from "react";
import { Flag, Gauge, Boxes, Play, ArrowDown, FlaskConical, TrendingUp, Brain } from "lucide-react";
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
import type { PlaybackMetrics, SimTrack, TrainMetrics } from "@/sim/types";
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
  const [manual, setManual] = useState(false);
  const [untrained, setUntrained] = useState(false);
  const [learning, setLearning] = useState(false);
  const [train, setTrain] = useState<TrainMetrics | null>(null);
  const PLAYER_CAR = 0;
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
          onTrain: setTrain,
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
  const toggleManual = (v: boolean) => {
    if (v && learning) { setLearning(false); setTrain(null); srcRef.current?.setLearning(false); }
    setManual(v);
    srcRef.current?.setPlayer(v ? PLAYER_CAR : -1);
  };
  const toggleUntrained = (v: boolean) => {
    if (v && learning) { setLearning(false); setTrain(null); }
    setUntrained(v);
    srcRef.current?.setUntrained(v);
  };
  // live "learn from scratch" mode (real in-browser RL)
  const LEARN_SPEED = 3; // fast-forward so learning is visible in ~a minute (user can slow down to watch)
  const toggleLearning = (v: boolean) => {
    setLearning(v);
    if (v) { setManual(false); setUntrained(false); if (speed < LEARN_SPEED) changeSpeed(LEARN_SPEED); }
    else setTrain(null);
    srcRef.current?.setLearning(v);
  };
  const resetBrain = () => {
    setLearning(true);
    setManual(false);
    setUntrained(false);
    setTrain(null);
    if (speed < LEARN_SPEED) changeSpeed(LEARN_SPEED);
    srcRef.current?.resetBrain();
  };

  // keyboard control for the human-driven car (advanced mode)
  useEffect(() => {
    if (!manual) return;
    const keys = new Set<string>();
    const DRIVE = ["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright"];
    const send = () => {
      const steer = (keys.has("d") || keys.has("arrowright") ? 1 : 0) - (keys.has("a") || keys.has("arrowleft") ? 1 : 0);
      const accel = keys.has("w") || keys.has("arrowup") ? 1 : 0;
      const brake = keys.has("s") || keys.has("arrowdown") ? 1 : 0;
      srcRef.current?.playerInput(steer, accel, brake);
    };
    const down = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (!DRIVE.includes(k)) return;
      e.preventDefault();
      if (!keys.has(k)) { keys.add(k); send(); }
    };
    const up = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (keys.delete(k)) send();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      srcRef.current?.playerInput(0, 0, 0);
    };
  }, [manual]);
  const toggleFullscreen = () => {
    const el = demoRef.current;
    if (!el) return;
    if (!document.fullscreenElement) el.requestFullscreen?.().catch(() => setFs(true));
    else document.exitFullscreen?.();
  };

  const device = ready?.device ?? "Browser";
  const stepsPerSec = pb?.stepsPerSec ?? 0;
  const modelStep = ready?.modelStep ?? pb?.modelStep ?? 0;

  const hud: HudStatSpec[] = learning
    ? [
        { icon: <Brain className="h-3 w-3" />, label: "Learning", value: `${train?.updates ?? 0} upd` },
        { icon: <TrendingUp className="h-3 w-3" />, label: "Avg reward", value: (train?.avgReturn ?? 0).toFixed(1), accent: true },
        { icon: <Boxes className="h-3 w-3" />, label: "Cars", value: String(fleet) },
      ]
    : [
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
      manual={manual}
      onManual={toggleManual}
      untrained={untrained}
      onUntrained={toggleUntrained}
      learning={learning}
      onLearning={toggleLearning}
      onResetBrain={resetBrain}
      train={train}
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
          <h1 className="max-w-3xl text-balance text-4xl font-extrabold leading-[1.05] tracking-tight sm:text-6xl">
            An AI that taught itself to drive.
          </h1>
          <p className="mt-5 max-w-2xl text-pretty text-base leading-relaxed text-muted-foreground sm:text-lg">
            No hand‑coded rules. A Transformer policy trained with reinforcement learning discovered how to
            steer, brake and hold a racing line — from nothing but trial, error and reward. Watch it drive below.
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
            <a href="#how" className="inline-flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-medium transition-colors hover:bg-accent">
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
                  forceChaseCar={manual ? PLAYER_CAR : null}
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

        {/* documentation */}
        <section id="how" className="border-t py-16">
          <div className="max-w-3xl">
            <p className="text-sm font-medium text-brand">How it works</p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">Teaching a car to drive with zero rules</h2>
            <p className="mt-4 leading-relaxed text-muted-foreground">
              The car is never shown a racing line or told how to take a corner. It's given one goal — get as far
              around the track as possible without crashing — and left to work out the rest through reinforcement
              learning. This is the loop it repeated millions of times.
            </p>
          </div>

          <ol className="mt-10 space-y-8">
            {[
              {
                title: "A goal, not instructions",
                body: "The only feedback is a number: a small reward for every bit of forward progress along the track, and a penalty for going off it. There are no example laps to copy and no steering angles to memorise — just the score.",
              },
              {
                title: "What the car senses",
                body: "Each step the car reads seven distance sensors fanned out ahead (how far the track edge is in each direction), plus its own speed and how far its heading has drifted from the track — ten numbers in total. That is its entire view of the world; notably, it cannot see the other cars.",
              },
              {
                title: "The decision — a Transformer",
                body: "Those readings, plus a short memory of the previous few, feed a small Transformer — the same family of model behind today's language models. It outputs three continuous controls every step: steering, throttle and brake.",
              },
              {
                title: "Learning from reward (PPO)",
                body: "Many cars run in parallel to gather experience, and an algorithm called PPO gently shifts the network's weights so that actions leading to more reward become more likely. Braking for corners, carrying speed on the straights and holding a line all emerge on their own — none of it is programmed.",
              },
              {
                title: "The trained driver you see here",
                body: "Once training converges, the finished network is frozen and re-implemented to run this page. Every car above is that same trained policy — validated to reproduce the original model's decisions to five decimal places. Want to see the learning itself? Open the controls' Advanced panel and pick “Train a new brain” — a blank network learns to drive from scratch, live in your browser, with its reward climbing as you watch.",
              },
            ].map((s, i) => (
              <li key={s.title} className="flex gap-4 sm:gap-6">
                <div className="num flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-sm font-semibold text-brand">
                  {i + 1}
                </div>
                <div className="max-w-2xl">
                  <h3 className="text-base font-semibold">{s.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{s.body}</p>
                </div>
              </li>
            ))}
          </ol>

          <div className="mt-12 grid grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border sm:grid-cols-4">
            {[
              { value: "10", label: "sensor inputs" },
              { value: "3", label: "continuous controls" },
              { value: "2.4M", label: "training steps" },
              { value: "0", label: "hand-coded driving rules" },
            ].map((f) => (
              <div key={f.label} className="bg-card p-4">
                <div className="num text-2xl font-bold">{f.value}</div>
                <div className="mt-1 text-xs text-muted-foreground">{f.label}</div>
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
