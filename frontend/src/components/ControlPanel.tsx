import { useState } from "react";
import { Play, Pause, Square, RotateCcw, Map, Boxes, Trash2, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api, type TrackOption } from "@/lib/api";

interface Props {
  status: string;
  trackName: string;
  trackOptions: TrackOption[];
  onTrackChange: (name: string) => void;
  fleet: number;
  gpu: boolean;
}

const FLEET_OPTIONS = [20, 48, 96, 150];

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}

export function ControlPanel({ status, trackName, trackOptions, onTrackChange, fleet, gpu }: Props) {
  const [lr, setLr] = useState(3e-4);
  const [simDelay, setSimDelay] = useState(0.012);
  const [ent, setEnt] = useState(0.005);
  const running = status === "running";

  const call = (action: string) => api.control(action).catch(console.error);
  const pushConfig = (cfg: Record<string, number>) => api.config(cfg).catch(console.error);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold">Training Control</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* setup */}
        <div className="space-y-3">
          <div className="space-y-1.5">
            <SectionLabel><Map className="h-3.5 w-3.5" /> Circuit</SectionLabel>
            <Select value={trackName} onValueChange={onTrackChange}>
              <SelectTrigger><SelectValue placeholder="Select a track" /></SelectTrigger>
              <SelectContent>
                {trackOptions.map((t) => (
                  <SelectItem key={t.id} value={t.id}>{t.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <SectionLabel><Boxes className="h-3.5 w-3.5" /> Parallel cars</SectionLabel>
            <Select value={String(fleet)} onValueChange={(v) => pushConfig({ fleet: Number(v) })}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {[...new Set([fleet, ...FLEET_OPTIONS])].sort((a, b) => a - b).map((n) => (
                  <SelectItem key={n} value={String(n)}>{n} cars</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[11px] leading-snug text-muted-foreground">
              {gpu ? "More cars = more parallel work for the GPU." : "Running on CPU — large fleets train slowly."}
            </p>
          </div>
        </div>

        {/* transport */}
        <div className="space-y-2">
          {!running ? (
            <Button onClick={() => call("start")} className="w-full">
              <Play /> {status === "paused" ? "Resume training" : "Start training"}
            </Button>
          ) : (
            <Button variant="secondary" onClick={() => call("pause")} className="w-full">
              <Pause /> Pause
            </Button>
          )}
          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" onClick={() => call("reset")}>
              <RotateCcw /> Reset cars
            </Button>
            <Button variant="outline" onClick={() => call("stop")}>
              <Square /> Stop
            </Button>
          </div>
          <Button
            variant="ghost"
            className="w-full text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={() => {
              if (
                window.confirm(
                  "Reset ALL training progress?\n\nThis wipes the learned model, metrics and reward history, and the auto-saved checkpoint for this circuit. Your named checkpoints are kept and can still be loaded."
                )
              )
                call("resetProgress");
            }}
          >
            <Trash2 /> Reset progress
          </Button>
        </div>

        <Separator />

        {/* tuning */}
        <div className="space-y-4">
          <SectionLabel><SlidersHorizontal className="h-3.5 w-3.5" /> Hyperparameters</SectionLabel>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <Label className="text-muted-foreground">Learning rate</Label>
              <span className="font-mono text-foreground">{lr.toExponential(1)}</span>
            </div>
            <Slider min={-5} max={-2.5} step={0.1} value={[Math.log10(lr)]}
              onValueChange={([v]) => setLr(Math.pow(10, v))}
              onValueCommit={([v]) => pushConfig({ lr: Math.pow(10, v) })} />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <Label className="text-muted-foreground">Sim speed</Label>
              <span className="font-mono text-foreground">{simDelay === 0 ? "max" : `${(simDelay * 1000).toFixed(0)}ms/step`}</span>
            </div>
            <Slider min={0} max={0.05} step={0.002} value={[0.05 - simDelay]}
              onValueChange={([v]) => setSimDelay(0.05 - v)}
              onValueCommit={([v]) => pushConfig({ simDelay: 0.05 - v })} />
            <p className="text-[11px] leading-snug text-muted-foreground">Lower delay = faster learning, less smooth motion.</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <Label className="text-muted-foreground">Exploration (entropy)</Label>
              <span className="font-mono text-foreground">{ent.toFixed(4)}</span>
            </div>
            <Slider min={0} max={0.03} step={0.001} value={[ent]}
              onValueChange={([v]) => setEnt(v)}
              onValueCommit={([v]) => pushConfig({ entCoef: v })} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
