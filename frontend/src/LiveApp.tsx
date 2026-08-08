import { useEffect, useRef, useState } from "react";
import { Cpu, Wifi, WifiOff, Sparkles } from "lucide-react";
import { Arena, type HudStatSpec } from "@/components/Arena";
import { TopNav } from "@/components/TopNav";
import { Pill } from "@/components/StatusPill";
import { ControlPanel } from "@/components/ControlPanel";
import { MetricsPanel } from "@/components/MetricsPanel";
import { CheckpointPanel } from "@/components/CheckpointPanel";
import { Flag, Gauge, Boxes } from "lucide-react";
import {
  api,
  API_BASE,
  type Metrics,
  type Telemetry,
  type TrackGeometry,
  type TrackOption,
} from "@/lib/api";

const STATUS: Record<string, { label: string; dot: string; text: string; live?: boolean }> = {
  running: { label: "Running", dot: "bg-live", text: "text-live", live: true },
  paused: { label: "Paused", dot: "bg-amber-500", text: "text-amber-500" },
  stopped: { label: "Stopped", dot: "bg-destructive", text: "text-destructive" },
  idle: { label: "Idle", dot: "bg-muted-foreground", text: "text-muted-foreground" },
};

const isGpu = (d?: string) => /cuda|nvidia|gpu/i.test(d ?? "");

interface Props {
  onGoPlayground: () => void;
}

/** The local training console: live server telemetry + training controls. */
export default function LiveApp({ onGoPlayground }: Props) {
  const [track, setTrack] = useState<TrackGeometry | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [connected, setConnected] = useState(false);
  const [trackOptions, setTrackOptions] = useState<TrackOption[]>([]);
  const [trackName, setTrackName] = useState("default");
  const telemetryRef = useRef<Telemetry | null>(null);

  useEffect(() => {
    let alive = true;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const tryLoad = () =>
      api.track().then((t) => {
        if (!alive) return;
        setTrack(t);
        setTrackName(t.track);
      }).catch(() => { if (alive) retry = setTimeout(tryLoad, 1500); });
    tryLoad();
    api.tracks().then((opts) => alive && setTrackOptions(opts)).catch(() => {});
    return () => { alive = false; if (retry) clearTimeout(retry); };
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
      } catch { /* ignore */ }
    };
    return () => es.close();
  }, []);

  const status = metrics?.status ?? "idle";
  const st = STATUS[status] ?? STATUS.idle;
  const gpu = isGpu(metrics?.device);
  const numCars = metrics?.numEnvs ?? 20;

  const hud: HudStatSpec[] = [
    { icon: <Flag className="h-3 w-3" />, label: "Laps", value: String(metrics?.totalLaps ?? 0) },
    { icon: <Gauge className="h-3 w-3" />, label: "Best", value: metrics ? metrics.bestReturn.toFixed(1) : "—" },
    { icon: <Boxes className="h-3 w-3" />, label: "Steps/s", value: metrics ? metrics.fps.toFixed(0) : "—", accent: true },
  ];

  return (
    <div className="flex h-full flex-col">
      <TopNav
        compact
        right={
          <button
            onClick={onGoPlayground}
            className="mr-1 flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <Sparkles className="h-3.5 w-3.5" /> Playground
          </button>
        }
      />

      {/* status strip */}
      <div className="flex flex-wrap items-center gap-2 border-b bg-secondary/30 px-4 py-2 sm:px-6">
        {metrics && (
          <Pill className={gpu ? "text-brand" : "text-muted-foreground"}>
            <Cpu className="h-3.5 w-3.5" />
            <span className="font-medium">{gpu ? "GPU" : "CPU"}</span>
            <span className="hidden max-w-[160px] truncate text-muted-foreground md:inline">
              {metrics.device.replace(/NVIDIA GeForce |Laptop GPU/g, "").trim()}
            </span>
          </Pill>
        )}
        <Pill className="num text-muted-foreground">
          {metrics ? metrics.globalStep.toLocaleString() : "0"} steps
        </Pill>
        <Pill className={st.text}>
          <span className={`h-2 w-2 rounded-full ${st.dot} ${st.live ? "animate-livepulse" : ""}`} />
          <span className="font-medium">{st.label}</span>
        </Pill>
        <Pill className={connected ? "text-live" : "text-destructive"}>
          {connected ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
          <span className="hidden font-medium sm:inline">{connected ? "connected" : "offline"}</span>
        </Pill>
      </div>

      <main className="grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_360px] sm:p-6">
        <section className="min-h-0 min-w-0">
          {track ? (
            <Arena key={trackName} track={track} telemetryRef={telemetryRef} numCars={numCars} hud={hud} active={status === "running"} />
          ) : (
            <div className="flex h-full min-h-[420px] items-center justify-center rounded-xl border bg-[#0a0e16] text-sm text-white/60">
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />
              <span className="ml-2">Connecting to the training backend…</span>
            </div>
          )}
        </section>

        <aside className="scroll-slim flex min-h-0 flex-col gap-4 overflow-y-auto pb-1">
          <ControlPanel status={status} trackName={trackName} trackOptions={trackOptions} onTrackChange={handleTrackChange} fleet={numCars} gpu={gpu} />
          <MetricsPanel metrics={metrics} />
          <div className="flex min-h-[280px] flex-1 flex-col">
            <CheckpointPanel />
          </div>
        </aside>
      </main>
    </div>
  );
}
