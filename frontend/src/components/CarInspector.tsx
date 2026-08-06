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

// 0..1 bar
function Bar({ value, color, label, display }: { value: number; color: string; label: string; display: string }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-white/60">{label}</span>
        <span className="font-mono tabular-nums text-white/90">{display}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full transition-[width] duration-100"
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }}
        />
      </div>
    </div>
  );
}

// bipolar -1..1 bar (fills from center)
function BipolarBar({ value, label }: { value: number; label: string }) {
  const v = Math.max(-1, Math.min(1, value));
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-white/60">{label}</span>
        <span className="font-mono tabular-nums text-white/90">{v >= 0 ? "+" : ""}{v.toFixed(2)}</span>
      </div>
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        <div className="absolute left-1/2 top-0 h-full w-px bg-white/30" />
        <div
          className="absolute top-0 h-full bg-data transition-all duration-100"
          style={{
            left: v >= 0 ? "50%" : `${50 + v * 50}%`,
            width: `${Math.abs(v) * 50}%`,
          }}
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

  const color = carColorCss(carId);

  return (
    <div className="glass-hud pointer-events-auto w-64 rounded-2xl p-3.5">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-sm ring-1 ring-white/20" style={{ background: color }} />
          <span className="text-sm font-semibold text-white">Car #{carId}</span>
          {car?.offtrack && (
            <span className="rounded bg-destructive/25 px-1.5 py-0.5 text-[10px] font-medium text-red-300">
              off-track
            </span>
          )}
        </div>
        <Button variant="ghost" size="icon" className="h-6 w-6 text-white/60 hover:text-white" onClick={onClose}>
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div className="mb-2 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-white/40">
        <Cpu className="h-3 w-3" /> Transformer outputs
      </div>
      <div className="space-y-2.5">
        <BipolarBar value={car?.steer ?? 0} label="Steering" />
        <Bar value={car?.accel ?? 0} color="hsl(var(--primary))" label="Acceleration" display={(car?.accel ?? 0).toFixed(2)} />
        <Bar value={car?.brake ?? 0} color="hsl(var(--destructive))" label="Brake" display={(car?.brake ?? 0).toFixed(2)} />
      </div>

      <div className="my-3 h-px bg-white/10" />

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-lg bg-white/5 p-2">
          <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-white/40">
            <Gauge className="h-3 w-3" /> Speed
          </div>
          <div className="font-mono text-sm tabular-nums text-white">
            {((car?.v ?? 0) * 3.6).toFixed(0)}
            <span className="ml-1 text-[10px] text-white/40">km/h</span>
          </div>
        </div>
        <div className="rounded-lg bg-white/5 p-2">
          <div className="text-[10px] uppercase tracking-wider text-white/40">Reward</div>
          <div className={`font-mono text-sm tabular-nums ${(car?.reward ?? 0) >= 0 ? "text-primary" : "text-red-400"}`}>
            {(car?.reward ?? 0).toFixed(3)}
          </div>
        </div>
      </div>

      <div className="mb-1.5 mt-3 text-[10px] uppercase tracking-wider text-white/40">
        Lidar sensors ({car?.sensors?.length ?? 0} rays)
      </div>
      <div className="flex h-12 items-end gap-1">
        {(car?.sensors ?? []).map((s, i) => (
          <div key={i} className="flex flex-1 flex-col items-center gap-1">
            <div className="flex h-9 w-full items-end rounded-sm bg-white/5">
              <div className="w-full rounded-sm bg-data/80" style={{ height: `${Math.max(4, s * 100)}%` }} />
            </div>
            <span className="text-[8px] text-white/40">{i}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
