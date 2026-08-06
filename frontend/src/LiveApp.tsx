import { useEffect, useRef, useState } from "react";
import { Cpu, Wifi, WifiOff, Flag, Gauge, Boxes, Sparkles } from "lucide-react";
import { Arena, type HudStatSpec } from "@/components/Arena";
import { Pill } from "@/components/StatusPill";
import { ControlPanel } from "@/components/ControlPanel";
import { MetricsPanel } from "@/components/MetricsPanel";
import { CheckpointPanel } from "@/components/CheckpointPanel";
import {
  api,
  API_BASE,
  type Metrics,
  type Telemetry,
  type TrackGeometry,
  type TrackOption,
} from "@/lib/api";

const STATUS = {
  running: { label: "Running", dot: "bg-primary", text: "text-primary", live: true },
  paused: { label: "Paused", dot: "bg-amber-400", text: "text-amber-400", live: false },
  stopped: { label: "Stopped", dot: "bg-destructive", text: "text-destructive", live: false },
  idle: { label: "Idle", dot: "bg-muted-foreground", text: "text-muted-foreground", live: false },
} as const;

function isGpu(device?: string) {
  return /cuda|nvidia|gpu/i.test(device ?? "");
}

interface Props {
  onGoPlayground: () => void;
}

/** The full training console: live server telemetry + training controls. */
export default function LiveApp({ onGoPlayground }: Props) {
  const [track, setTrack] = useState<TrackGeometry | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [connected, setConnected] = useState(false);
  const [trackOptions, setTrackOptions] = useState<TrackOption[]>([]);
  const [trackName, setTrackName] = useState("default");
  const telemetryRef = useRef<Telemetry | null>(null);

  useEffect(() => {
    let alive = true;
    const tryLoad = () =>
      api
        .track()
        .then((t) => {
          if (!alive) return;
          setTrack(t);
          setTrackName(t.track);
        })
        .catch(() => setTimeout(tryLoad, 1500));
    tryLoad();
    api.tracks().then((opts) => alive && setTrackOptions(opts)).catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const handleTrackChange = (name: string) => {
    if (name === trackName) return;
    setTrackName(name);
    api.setTrack(name).then((geo) => setTrack(geo)).catch(console.error);
  };

  useEffect(() => {
    const es = new EventSource(`${API_BASE}/telemetry`);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as { telemetry: Telemetry; metrics: Metrics };
        telemetryRef.current = data.telemetry;
        setMetrics(data.metrics);
      } catch {
        /* ignore malformed frame */
      }
    };
    return () => es.close();
  }, []);

  const status = metrics?.status ?? "idle";
  const st = STATUS[status as keyof typeof STATUS] ?? STATUS.idle;
  const gpu = isGpu(metrics?.device);
  const numCars = metrics?.numEnvs ?? 20;

  const hud: HudStatSpec[] = [
    { icon: <Flag className="h-3 w-3" />, label: "Laps", value: String(metrics?.totalLaps ?? 0), accent: "green" },
    { icon: <Gauge className="h-3 w-3" />, label: "Best", value: metrics ? metrics.bestReturn.toFixed(1) : "—" },
    { icon: <Boxes className="h-3 w-3" />, label: "Steps/s", value: metrics ? metrics.fps.toFixed(0) : "—", accent: "cyan" },
  ];

  return (
    <div className="flex h-full flex-col">
      <header className="glass sticky top-0 z-20 flex items-center justify-between gap-3 rounded-none border-x-0 border-t-0 px-4 py-2.5 sm:px-5">
        <div className="flex items-center gap-3">
          <div className="glow-primary flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary/25 to-primary/5 text-primary">
            <Cpu className="h-5 w-5" />
          </div>
          <div className="leading-tight">
            <h1 className="text-[15px] font-semibold tracking-tight sm:text-base">Reinforcement Car</h1>
            <p className="hidden text-xs text-muted-foreground sm:block">
              Transformer · PPO · {numCars} agents in parallel
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {metrics && (
            <Pill className={gpu ? "text-primary" : "text-muted-foreground"}>
              <Cpu className="h-3.5 w-3.5" />
              <span className="font-semibold">{gpu ? "GPU" : "CPU"}</span>
              <span className="hidden max-w-[150px] truncate text-muted-foreground md:inline">
                {metrics.device.replace(/NVIDIA GeForce |Laptop GPU/g, "").trim()}
              </span>
            </Pill>
          )}
          <Pill className="hidden font-mono text-muted-foreground sm:flex">
            {metrics ? metrics.globalStep.toLocaleString() : "0"}
            <span className="text-muted-foreground/60">steps</span>
          </Pill>
          <Pill className={st.text}>
            <span className={`h-2 w-2 rounded-full ${st.dot} ${st.live ? "animate-livepulse" : ""}`} />
            <span className="font-medium">{st.label}</span>
          </Pill>
          <Pill className={connected ? "text-primary" : "text-destructive"}>
            {connected ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
            <span className="hidden font-medium sm:inline">{connected ? "live" : "offline"}</span>
          </Pill>
          <button
            onClick={onGoPlayground}
            title="Preview the in-browser playground"
            className="glass-hud flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-white/70 transition-colors hover:text-white"
          >
            <Sparkles className="h-3.5 w-3.5" /> Playground
          </button>
        </div>
      </header>

      <main className="grid min-h-0 min-w-0 flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_380px]">
        {track ? (
          <Arena
            key={trackName}
            track={track}
            telemetryRef={telemetryRef}
            numCars={numCars}
            hud={hud}
            caption={`Each car is one of ${numCars} agents learning to drive. Grey = off-track (resetting); trails show recent paths. Steering, acceleration and brake are all produced by the Transformer policy.`}
          />
        ) : (
          <div className="flex min-h-0 items-center justify-center rounded-2xl border bg-[#070a12] text-sm text-muted-foreground">
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-primary/30 border-t-primary" />
            <span className="ml-2">Connecting to the training backend…</span>
          </div>
        )}

        <aside className="scroll-slim flex min-h-0 flex-col gap-4 overflow-y-auto pb-1">
          <div className="animate-rise" style={{ animationDelay: "40ms" }}>
            <ControlPanel
              status={status}
              trackName={trackName}
              trackOptions={trackOptions}
              onTrackChange={handleTrackChange}
              fleet={numCars}
              gpu={gpu}
            />
          </div>
          <div className="animate-rise" style={{ animationDelay: "100ms" }}>
            <MetricsPanel metrics={metrics} />
          </div>
          <div className="flex min-h-[280px] flex-1 animate-rise flex-col" style={{ animationDelay: "160ms" }}>
            <CheckpointPanel />
          </div>
        </aside>
      </main>
    </div>
  );
}
