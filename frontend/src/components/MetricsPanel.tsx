import { useEffect, useRef, useState } from "react";
import { Activity } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type Metrics } from "@/lib/api";

interface Props {
  metrics: Metrics | null;
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: "green" | "cyan" }) {
  const tone = accent === "green" ? "text-primary" : accent === "cyan" ? "text-data" : "text-foreground";
  return (
    <div className="hairline rounded-xl bg-secondary/40 p-2.5">
      <div className="truncate text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-0.5 font-mono text-sm font-semibold leading-tight tabular-nums ${tone}`}>{value}</div>
    </div>
  );
}

function Sparkline({ data }: { data: number[] }) {
  if (data.length < 2) {
    return (
      <div className="flex h-16 items-center justify-center text-[11px] text-muted-foreground">
        collecting data…
      </div>
    );
  }
  const w = 100;
  const h = 40;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const step = w / (data.length - 1);
  const pts = data
    .map((v, i) => `${(i * step).toFixed(2)},${(h - ((v - min) / range) * h).toFixed(2)}`)
    .join(" ");
  const last = data[data.length - 1];
  const lastX = (data.length - 1) * step;
  const lastY = h - ((last - min) / range) * h;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="h-16 w-full">
      <defs>
        <linearGradient id="spark" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity="0.4" />
          <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${h} ${pts} ${w},${h}`} fill="url(#spark)" />
      <polyline
        points={pts}
        fill="none"
        stroke="hsl(var(--primary))"
        strokeWidth="1.75"
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx={lastX} cy={lastY} r="2.5" fill="hsl(var(--primary))" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function MetricsPanel({ metrics }: Props) {
  const m = metrics;
  const [history, setHistory] = useState<number[]>([]);
  const lastSample = useRef(0);

  // seed the sparkline from persisted training history on mount
  useEffect(() => {
    api
      .history()
      .then((h) => {
        if (!h.length) return;
        // don't clobber live points that may have already arrived
        setHistory((cur) => (cur.length ? cur : h.slice(-80).map((p) => p.meanReturn)));
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!m) return;
    const now = performance.now();
    if (now - lastSample.current < 500) return; // sample ~2 Hz
    lastSample.current = now;
    setHistory((h) => [...h.slice(-79), m.meanReturn]);
  }, [m]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold">
          <Activity className="h-4 w-4 text-primary" /> Live Metrics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* reward trend hero */}
        <div className="hairline rounded-xl bg-secondary/30 p-3">
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Mean return</span>
            <span className="font-mono text-lg font-semibold tabular-nums text-primary">
              {m ? m.meanReturn.toFixed(2) : "—"}
            </span>
          </div>
          <Sparkline data={history} />
        </div>

        <div className="grid grid-cols-3 gap-2">
          <Stat label="Env steps" value={m ? m.globalStep.toLocaleString() : "—"} />
          <Stat label="Updates" value={m ? String(m.updates) : "—"} />
          <Stat label="Steps/s" value={m ? m.fps.toFixed(0) : "—"} accent="cyan" />
          <Stat label="Best" value={m ? m.bestReturn.toFixed(1) : "—"} accent="green" />
          <Stat label="Laps" value={m ? String(m.totalLaps) : "—"} accent="green" />
          <Stat label="Ep. len" value={m ? m.meanEpisodeLen.toFixed(0) : "—"} />
        </div>
      </CardContent>
    </Card>
  );
}
