import { useEffect, useState } from "react";
import LiveApp from "@/LiveApp";
import PlaybackApp from "@/PlaybackApp";
import { backendAvailable } from "@/sim/source";

function Splash({ label }: { label: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-muted-foreground">
      <span className="h-6 w-6 animate-spin rounded-full border-2 border-primary/30 border-t-primary" />
      {label}
    </div>
  );
}

/**
 * Routes between two shells:
 *   • a training backend is reachable   → the full live training console
 *   • no backend (e.g. the static deploy) → the in-browser playback playground
 * When a backend exists you can still switch to the playground to preview it.
 */
export default function App() {
  const [hasBackend, setHasBackend] = useState<boolean | null>(null);
  const [mode, setMode] = useState<"live" | "playback" | null>(null);

  useEffect(() => {
    // A static playground deploy can force playback and skip backend probing.
    const forced = import.meta.env.VITE_PLAYBACK_ONLY;
    if (forced === "true" || forced === "1") {
      setHasBackend(false);
      setMode("playback");
      return;
    }
    backendAvailable().then((ok) => {
      setHasBackend(ok);
      setMode(ok ? "live" : "playback");
    });
  }, []);

  if (mode === null) return <Splash label="Starting…" />;

  return mode === "playback" ? (
    <PlaybackApp hasBackend={!!hasBackend} onGoLive={() => setMode("live")} />
  ) : (
    <LiveApp onGoPlayground={() => setMode("playback")} />
  );
}
