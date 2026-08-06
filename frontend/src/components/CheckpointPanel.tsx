import { useEffect, useState } from "react";
import { Save, Download, Trash2, RefreshCw, Archive } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { api, type Checkpoint } from "@/lib/api";

export function CheckpointPanel() {
  const [items, setItems] = useState<Checkpoint[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = async () => {
    try {
      setItems(await api.listCheckpoints());
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const flash = (t: string) => {
    setMsg(t);
    setTimeout(() => setMsg(null), 2500);
  };

  const save = async () => {
    setBusy(true);
    try {
      const info = await api.saveCheckpoint(name.trim());
      flash(`Saved "${info.name}"`);
      setName("");
      await refresh();
    } catch (e) {
      flash("Save failed");
    } finally {
      setBusy(false);
    }
  };

  const load = async (n: string) => {
    try {
      await api.loadCheckpoint(n);
      flash(`Loaded "${n}" into the live model`);
    } catch {
      flash("Load failed");
    }
  };

  const del = async (n: string) => {
    try {
      await api.deleteCheckpoint(n);
      flash(`Deleted "${n}"`);
      await refresh();
    } catch {
      flash("Delete failed");
    }
  };

  return (
    <Card className="flex min-h-0 flex-1 flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold">
            <Archive className="h-4 w-4 text-data" /> Checkpoints
          </CardTitle>
          <Button variant="ghost" size="icon" onClick={refresh} title="Refresh" className="h-7 w-7">
            <RefreshCw />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="flex gap-2">
          <Input
            placeholder="checkpoint name (optional)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && save()}
          />
          <Button onClick={save} disabled={busy}>
            <Save /> Save
          </Button>
        </div>

        {msg && (
          <div className="animate-rise rounded-lg bg-primary/10 px-2.5 py-1.5 text-xs text-primary">{msg}</div>
        )}

        <div className="scroll-slim min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
          {items.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center gap-1 py-8 text-center">
              <Archive className="h-6 w-6 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground">No checkpoints yet</p>
              <p className="text-xs text-muted-foreground/70">Train a bit, then save one.</p>
            </div>
          )}
          {items.map((c) => (
            <div
              key={c.name}
              className="hairline group flex items-center justify-between gap-2 rounded-xl bg-secondary/30 p-2.5 transition-colors hover:bg-secondary/60"
            >
              <div className="min-w-0">
                <div className="truncate text-sm font-medium">{c.name}</div>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  <Badge variant="secondary" className="font-mono text-[10px]">
                    {c.globalStep.toLocaleString()} steps
                  </Badge>
                  <Badge variant="outline" className="font-mono text-[10px] text-primary">
                    ret {c.meanReturn.toFixed(1)}
                  </Badge>
                  <Badge variant="outline" className="font-mono text-[10px] text-muted-foreground">
                    {c.sizeKB}KB
                  </Badge>
                </div>
              </div>
              <div className="flex shrink-0 gap-1">
                <Button variant="outline" size="icon" title="Load into live model" onClick={() => load(c.name)} className="h-8 w-8">
                  <Download />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  title="Delete"
                  className="h-8 w-8 text-destructive hover:bg-destructive/10 hover:text-destructive"
                  onClick={() => del(c.name)}
                >
                  <Trash2 />
                </Button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
