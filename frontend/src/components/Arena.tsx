import { useEffect, useState, lazy, Suspense } from "react";
import { Radar, Camera, MousePointerClick, Spline, Maximize2, Minimize2 } from "lucide-react";
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
  accent?: boolean;
}

export function HudStat({ icon, label, value, accent }: HudStatSpec) {
  return (
    <div className="rounded-lg border bg-card/95 px-3 py-1.5 shadow-sm">
      <div className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {icon} {label}
      </div>
      <div className={`num mt-0.5 text-[15px] font-semibold leading-none ${accent ? "text-brand" : "text-foreground"}`}>
        {value}
      </div>
    </div>
  );
}

function OverlayToggle({
  active,
  onClick,
  children,
  title,
  disabled,
}: {
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
  title?: string;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium shadow-sm transition-colors disabled:opacity-40 ${
        active ? "border-brand/40 bg-brand/10 text-brand" : "bg-card text-muted-foreground hover:text-foreground"
      }`}
    >
      {children}
    </button>
  );
}

interface ArenaProps {
  track: TrackGeometry;
  telemetryRef: React.MutableRefObject<Telemetry | null>;
  numCars: number;
  hud: HudStatSpec[];
  active?: boolean;
  fullscreen?: boolean;
  onToggleFullscreen?: () => void;
  forceChaseCar?: number | null; // manual-drive: select + chase this car
}

/** Shared live 3D/2D race view: renderers, HUD, view/camera controls, racing
 * line, click-to-inspect, fullscreen. Owns view-only state. */
export function Arena({ track, telemetryRef, numCars, hud, active = true, fullscreen, onToggleFullscreen, forceChaseCar }: ArenaProps) {
  const [selectedCar, setSelectedCar] = useState<number | null>(null);
  const [showSensors, setShowSensors] = useState(false);
  const [view, setView] = useState<"3d" | "2d">("3d");
  const [chase, setChase] = useState(false);
  const [racingLine, setRacingLine] = useState(false);

  useEffect(() => {
    if (selectedCar !== null && selectedCar >= numCars) setSelectedCar(null);
  }, [numCars, selectedCar]);

  // manual-drive: follow the player's car
  useEffect(() => {
    if (forceChaseCar != null && forceChaseCar >= 0) {
      setSelectedCar(forceChaseCar);
      setChase(true);
      setView("3d");
    } else if (forceChaseCar === null) {
      setChase(false);
    }
  }, [forceChaseCar]);

  // Chase: if nothing is selected, grab a random car and follow it
  const startChase = () => {
    if (chase) {
      setChase(false);
      return;
    }
    if (selectedCar === null) setSelectedCar(Math.floor(Math.random() * Math.max(1, numCars)));
    setChase(true);
  };

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border bg-[#0a0e16]">
      {view === "3d" ? (
        <ErrorBoundary
          fallback={
            <CarCanvas track={track} telemetryRef={telemetryRef} selectedId={selectedCar} onSelect={setSelectedCar} showSensors={showSensors} showRacingLine={racingLine} active={active} />
          }
        >
          <Suspense
            fallback={
              <div className="flex h-full items-center justify-center bg-[#0a0e16] text-sm text-white/60">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />
                <span className="ml-2">Loading 3D scene…</span>
              </div>
            }
          >
            <CarScene3D
              track={track}
              telemetryRef={telemetryRef}
              selectedId={selectedCar}
              onSelect={setSelectedCar}
              chase={chase}
              numCars={numCars}
              active={active}
              showRacingLine={racingLine}
            />
          </Suspense>
        </ErrorBoundary>
      ) : (
        <CarCanvas track={track} telemetryRef={telemetryRef} selectedId={selectedCar} onSelect={setSelectedCar} showSensors={showSensors} showRacingLine={racingLine} />
      )}

      {/* live HUD */}
      <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2">
        {hud.map((h) => (
          <HudStat key={h.label} {...h} />
        ))}
      </div>

      {/* view + camera controls */}
      <div className="absolute right-3 top-3 flex flex-wrap items-center justify-end gap-2">
        <OverlayToggle active={racingLine} onClick={() => setRacingLine((s) => !s)} title="Show the ideal racing line">
          <Spline className="h-3.5 w-3.5" /> Line
        </OverlayToggle>
        {view === "2d" && (
          <OverlayToggle active={showSensors} onClick={() => setShowSensors((s) => !s)}>
            <Radar className="h-3.5 w-3.5" /> Sensors
          </OverlayToggle>
        )}
        {view === "3d" && (
          <OverlayToggle active={chase} onClick={startChase} title="Chase a car (picks one if none selected)">
            <Camera className="h-3.5 w-3.5" /> Chase
          </OverlayToggle>
        )}
        {onToggleFullscreen && (
          <OverlayToggle active={fullscreen} onClick={onToggleFullscreen} title="Fullscreen">
            {fullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </OverlayToggle>
        )}
        <div className="flex rounded-lg border bg-card p-0.5 shadow-sm">
          {(["3d", "2d"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`rounded-md px-3 py-1 text-xs font-semibold uppercase transition-colors ${
                view === v ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
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
        <div className="pointer-events-none absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-1.5 rounded-full border bg-card/95 px-3 py-1 text-[11px] text-muted-foreground shadow-sm">
          <MousePointerClick className="h-3 w-3" />
          Click a car to inspect{view === "3d" ? " · drag to orbit, right-drag to pan" : ""}
        </div>
      )}
    </div>
  );
}
