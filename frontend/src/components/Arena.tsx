import { useEffect, useState, lazy, Suspense } from "react";
import { Radar, Camera, MousePointerClick } from "lucide-react";
import { CarCanvas } from "@/components/CarCanvas";
import { CarInspector } from "@/components/CarInspector";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import type { Telemetry, TrackGeometry } from "@/lib/api";

// Three.js (~1 MB) loads only when the 3D view is used.
const CarScene3D = lazy(() =>
  import("@/components/CarScene3D").then((m) => ({ default: m.CarScene3D }))
);

export interface HudStatSpec {
  icon: React.ReactNode;
  label: string;
  value: string;
  accent?: "green" | "cyan";
}

/** floating stat chip over the arena */
export function HudStat({ icon, label, value, accent }: HudStatSpec) {
  const tone = accent === "green" ? "text-primary" : accent === "cyan" ? "text-data" : "text-foreground";
  return (
    <div className="glass-hud rounded-xl px-3 py-2">
      <div className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {icon} {label}
      </div>
      <div className={`mt-0.5 font-mono text-base font-semibold leading-none tabular-nums ${tone}`}>{value}</div>
    </div>
  );
}

interface ArenaProps {
  track: TrackGeometry;
  telemetryRef: React.MutableRefObject<Telemetry | null>;
  numCars: number;
  hud: HudStatSpec[];
  caption: string;
}

/**
 * The shared live 3D/2D race view: renderers, floating HUD, view/camera
 * controls, click-to-inspect. Owns view-only state (view mode, chase, sensors,
 * selection) so both the training and playback shells reuse it identically.
 */
export function Arena({ track, telemetryRef, numCars, hud, caption }: ArenaProps) {
  const [selectedCar, setSelectedCar] = useState<number | null>(null);
  const [showSensors, setShowSensors] = useState(false);
  const [view, setView] = useState<"3d" | "2d">("3d");
  const [chase, setChase] = useState(false);

  // drop a now-out-of-range selection if the fleet shrank
  useEffect(() => {
    if (selectedCar !== null && selectedCar >= numCars) setSelectedCar(null);
  }, [numCars, selectedCar]);

  return (
    <section className="flex min-h-0 min-w-0 flex-col gap-2">
      <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden rounded-2xl">
        {view === "3d" ? (
          <ErrorBoundary
            fallback={
              <CarCanvas track={track} telemetryRef={telemetryRef} selectedId={selectedCar} onSelect={setSelectedCar} showSensors={showSensors} />
            }
          >
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center bg-[#070a12] text-sm text-muted-foreground">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary/30 border-t-primary" />
                  <span className="ml-2">Loading 3D scene…</span>
                </div>
              }
            >
              <CarScene3D track={track} telemetryRef={telemetryRef} selectedId={selectedCar} onSelect={setSelectedCar} chase={chase} numCars={numCars} />
            </Suspense>
          </ErrorBoundary>
        ) : (
          <CarCanvas track={track} telemetryRef={telemetryRef} selectedId={selectedCar} onSelect={setSelectedCar} showSensors={showSensors} />
        )}

        {/* live HUD */}
        <div className="pointer-events-none absolute left-3 top-3 flex gap-2">
          {hud.map((h) => (
            <HudStat key={h.label} {...h} />
          ))}
        </div>

        {/* view + camera controls */}
        <div className="absolute right-3 top-3 flex items-center gap-2">
          {view === "2d" && (
            <button
              onClick={() => setShowSensors((s) => !s)}
              className={`glass-hud flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${showSensors ? "text-data ring-1 ring-data/50" : "text-white/70 hover:text-white"}`}
            >
              <Radar className="h-3.5 w-3.5" /> Sensors
            </button>
          )}
          {view === "3d" && (
            <button
              onClick={() => setChase((c) => !c)}
              disabled={selectedCar === null}
              title={selectedCar === null ? "Select a car first" : "Chase camera"}
              className={`glass-hud flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-40 ${chase ? "text-data ring-1 ring-data/50" : "text-white/70 hover:text-white"}`}
            >
              <Camera className="h-3.5 w-3.5" /> Chase
            </button>
          )}
          <div className="glass-hud flex overflow-hidden rounded-lg p-0.5">
            {(["3d", "2d"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`rounded-md px-3 py-1 text-xs font-semibold uppercase transition-colors ${view === v ? "bg-primary text-primary-foreground shadow-glow" : "text-white/60 hover:text-white"}`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* car inspector */}
        {selectedCar !== null && (
          <div className="absolute bottom-3 right-3 animate-rise">
            <CarInspector telemetryRef={telemetryRef} carId={selectedCar} onClose={() => setSelectedCar(null)} />
          </div>
        )}

        {selectedCar === null && (
          <div className="glass-hud pointer-events-none absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-1.5 rounded-full px-3 py-1 text-[11px] text-white/70">
            <MousePointerClick className="h-3 w-3" />
            Click a car to inspect{view === "3d" ? " · drag to orbit, scroll to zoom" : " its Transformer outputs"}
          </div>
        )}
      </div>
      <p className="px-1 text-xs leading-relaxed text-muted-foreground">{caption}</p>
    </section>
  );
}
