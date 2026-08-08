import { useEffect, useState } from "react";
import { X, Gauge, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Car, Telemetry } from "@/lib/api";
import { carColorCss } from "@/lib/carViz";

interface Props {
  telemetryRef: React.MutableRefObject<Telemetry | null>;
  carId: number;
  onClose: () => void;
}

function Bar({ value, className, label, display }: { value: number; className: string; label: string; display: string }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="num text-foreground">{display}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={`h-full rounded-full transition-[width] duration-100 ${className}`}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
      </div>
    </div>
  );
}

function BipolarBar({ value, label }: { value: number; label: string }) {
  const v = Math.max(-1, Math.min(1, value));
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="num text-foreground">{v >= 0 ? "+" : ""}{v.toFixed(2)}</span>
      </div>
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div className="absolute left-1/2 top-0 h-full w-px bg-border" />
        <div
          className="absolute top-0 h-full bg-brand transition-all duration-100"
          style={{ left: v >= 0 ? "50%" : `${50 + v * 50}%`, width: `${Math.abs(v) * 50}%` }}
        />
      </div>
    </div>
  );
}

export function CarInspector({ telemetryRef, carId, onClose }: Props) {
  const [car, setCar] = useState<Car | null>(null);

  useEffect(() => {
    const id = setInterval(() => {
      const t = telemetryRef.current;
      setCar(t?.cars.find((c) => c.id === carId) ?? null);
    }, 80);
    return () => clearInterval(id);
  }, [telemetryRef, carId]);

  return (
    <div className="pointer-events-auto w-64 rounded-xl border bg-card p-3.5 shadow-md">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-sm ring-1 ring-border" style={{ background: carColorCss(carId) }} />
          <span className="text-sm font-semibold">Car #{carId}</span>
          {car?.offtrack && (
            <span className="rounded bg-destructive/10 px-1.5 py-0.5 text-[10px] font-medium text-destructive">off-track</span>
          )}
        </div>
        <Button variant="ghost" size="icon" aria-label="Close inspector" title="Close" className="h-6 w-6 text-muted-foreground" onClick={onClose}>
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div className="mb-2 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        <Cpu className="h-3 w-3" /> Transformer outputs
      </div>
      <div className="space-y-2.5">
        <BipolarBar value={car?.steer ?? 0} label="Steering" />
        <Bar value={car?.accel ?? 0} className="bg-live" label="Acceleration" display={(car?.accel ?? 0).toFixed(2)} />
        <Bar value={car?.brake ?? 0} className="bg-destructive" label="Brake" display={(car?.brake ?? 0).toFixed(2)} />
      </div>

      <div className="my-3 h-px bg-border" />

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-lg border bg-secondary/50 p-2">
          <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            <Gauge className="h-3 w-3" /> Speed
          </div>
          <div className="num text-sm text-foreground">
            {((car?.v ?? 0) * 3.6).toFixed(0)}
            <span className="ml-1 text-[10px] text-muted-foreground">km/h</span>
          </div>
        </div>
        <div className="rounded-lg border bg-secondary/50 p-2">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Reward</div>
          <div className={`num text-sm ${(car?.reward ?? 0) >= 0 ? "text-live" : "text-destructive"}`}>
            {(car?.reward ?? 0).toFixed(3)}
          </div>
        </div>
      </div>

      <div className="mb-1.5 mt-3 text-[10px] uppercase tracking-wide text-muted-foreground">
        Lidar ({car?.sensors?.length ?? 0} rays)
      </div>
      <div className="flex h-10 items-end gap-1">
        {(car?.sensors ?? []).map((s, i) => (
          <div key={i} className="flex h-full flex-1 items-end rounded-sm bg-secondary">
            <div className="w-full rounded-sm bg-brand/70" style={{ height: `${Math.max(4, s * 100)}%` }} />
          </div>
        ))}
      </div>
    </div>
  );
}
