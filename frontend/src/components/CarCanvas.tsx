import { useEffect, useRef } from "react";
import type { Telemetry, TrackGeometry } from "@/lib/api";
import { computeTrackEdges, racingLine, type Pt } from "@/lib/trackGeometry";
import { carColorCss, lerpAngle } from "@/lib/carViz";

interface Props {
  track: TrackGeometry;
  telemetryRef: React.MutableRefObject<Telemetry | null>;
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  showSensors: boolean;
  showRacingLine?: boolean;
  active?: boolean; // false when paused/idle — skip redraws to save CPU
}

interface RenderCar {
  x: number;
  y: number;
  svx: number; // measured on-screen velocity (world units/sec) — sim-speed independent
  svy: number;
  ltx: number; // last authoritative (truth) position
  lty: number;
  theta: number;
  v: number;
  offtrack: boolean;
  init: boolean;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  const rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

// world units for a car; scaled to screen at draw time
const CAR_L = 22;
const CAR_W = 11;

// draw a top-down car centered at origin, nose pointing +x (local space)
function drawCar(
  ctx: CanvasRenderingContext2D,
  s: number,
  color: string,
  offtrack: boolean,
  speedFrac: number
) {
  const L = CAR_L * s;
  const W = CAR_W * s;
  const base = offtrack ? "#5b606b" : color;

  // ground shadow
  ctx.save();
  ctx.globalAlpha = 0.35;
  ctx.fillStyle = "#000";
  roundRect(ctx, -L * 0.5 + 1.5 * s, -W * 0.5 + 1.5 * s, L, W, W * 0.35);
  ctx.fill();
  ctx.restore();

  // wheels (dark, slightly outside the body)
  ctx.fillStyle = "#15161a";
  const wl = L * 0.26;
  const ww = W * 0.16;
  const wy = W * 0.52;
  const wx = L * 0.28;
  for (const [cx, cy] of [
    [wx, wy],
    [wx, -wy],
    [-wx, wy],
    [-wx, -wy],
  ]) {
    roundRect(ctx, cx - wl / 2, cy - ww / 2, wl, ww, ww * 0.4);
    ctx.fill();
  }

  // body with a soft length-wise gradient + speed glow
  if (!offtrack && speedFrac > 0.05) {
    ctx.shadowColor = base;
    ctx.shadowBlur = 6 + speedFrac * 16;
  }
  const grad = ctx.createLinearGradient(-L / 2, 0, L / 2, 0);
  grad.addColorStop(0, base);
  grad.addColorStop(0.55, base);
  grad.addColorStop(1, "rgba(255,255,255,0.35)");
  ctx.fillStyle = grad;
  ctx.strokeStyle = "rgba(0,0,0,0.45)";
  ctx.lineWidth = Math.max(0.5, 0.8 * s);
  roundRect(ctx, -L * 0.5, -W * 0.5, L, W, W * 0.32);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.stroke();

  // cabin / windshield
  ctx.fillStyle = "rgba(12,14,20,0.85)";
  roundRect(ctx, -L * 0.08, -W * 0.34, L * 0.34, W * 0.68, W * 0.2);
  ctx.fill();

  // headlights
  ctx.fillStyle = "rgba(255,247,210,0.95)";
  for (const hy of [W * 0.28, -W * 0.28]) {
    roundRect(ctx, L * 0.42, hy - W * 0.07, L * 0.06, W * 0.14, W * 0.05);
    ctx.fill();
  }
  // tail lights
  ctx.fillStyle = "rgba(255,60,60,0.9)";
  for (const ty of [W * 0.28, -W * 0.28]) {
    roundRect(ctx, -L * 0.48, ty - W * 0.07, L * 0.05, W * 0.14, W * 0.05);
    ctx.fill();
  }
}

export function CarCanvas({ track, telemetryRef, selectedId, onSelect, showSensors, showRacingLine, active = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const rendered = useRef<RenderCar[]>([]);
  const trails = useRef<{ x: number; y: number }[][]>([]);
  // refs let the persistent draw loop read the latest props without restarting
  const selectedRef = useRef<number | null>(selectedId);
  selectedRef.current = selectedId;
  const showSensorsRef = useRef(showSensors);
  showSensorsRef.current = showSensors;
  const showRacingLineRef = useRef(showRacingLine);
  showRacingLineRef.current = showRacingLine;
  const activeRef = useRef(active);
  activeRef.current = active;
  const transformRef = useRef<{
    scale: number;
    offX: number;
    offY: number;
    H: number;
    minX: number;
    minY: number;
    dpr: number;
  } | null>(null);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const tf = transformRef.current;
    const tele = telemetryRef.current;
    if (!tf || !tele) return;
    const bx = e.nativeEvent.offsetX * tf.dpr;
    const by = e.nativeEvent.offsetY * tf.dpr;
    const wx = tf.minX + (bx - tf.offX) / tf.scale;
    const wy = tf.minY + (tf.H - tf.offY - by) / tf.scale;
    let best = -1;
    let bestD = 30 / tf.scale + CAR_L; // generous pick radius in world units
    for (const c of tele.cars) {
      const d = Math.hypot(c.x - wx, c.y - wy);
      if (d < bestD) {
        bestD = d;
        best = c.id;
      }
    }
    onSelect(best === selectedRef.current ? null : best === -1 ? null : best);
  };

  useEffect(() => {
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let dpr = 1;

    // reset interpolation/trail state when the track changes
    rendered.current = [];
    trails.current = [];

    // Keep the drawing buffer in sync with the wrapper's CSS size every frame.
    // This is resilient to first-paint timing, layout changes and browser zoom,
    // which ResizeObserver alone handled inconsistently.
    const syncSize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = wrap.clientWidth;
      const ch = wrap.clientHeight;
      if (cw === 0 || ch === 0) return false;
      const bw = Math.round(cw * dpr);
      const bh = Math.round(ch * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
      return true;
    };

    const { minX, maxX, minY, maxY } = track.bounds;
    const worldW = maxX - minX;
    const worldH = maxY - minY;
    const pts = track.centerline;
    const hw = track.halfWidth;
    const M = pts.length;

    // precompute inner / outer track boundaries (curvature-clamped so edges
    // don't fold over at corners tighter than the track width)
    const { left: inner, right: outer } = computeTrackEdges(pts as Pt[], hw);
    const racePts = racingLine(pts as Pt[], hw); // ideal line (render-only)

    let lastT = performance.now();
    let frameStep = -1;              // last telemetry step ingested
    let frameTime = performance.now(); // wall-clock when it arrived (for velocity from truth deltas)
    let drawnOnce = false;
    const draw = (now: number) => {
      const delta = Math.min((now - lastT) / 1000, 0.05);
      lastT = now;
      if (!syncSize()) {
        raf = requestAnimationFrame(draw);
        return;
      }
      // paused/idle: the sim isn't advancing, so retain the last frame and skip
      // the redraw work (canvas keeps its pixels). Draw once so it's never blank.
      if (!activeRef.current && drawnOnce) {
        raf = requestAnimationFrame(draw);
        return;
      }
      const W = canvas.width;
      const H = canvas.height;
      const pad = 28 * dpr;
      const scale = Math.min((W - pad * 2) / worldW, (H - pad * 2) / worldH);
      const offX = (W - worldW * scale) / 2;
      const offY = (H - worldH * scale) / 2;

      const sx = (x: number) => offX + (x - minX) * scale;
      const sy = (y: number) => H - (offY + (y - minY) * scale);
      transformRef.current = { scale, offX, offY, H, minX, minY, dpr };

      // ---- background: dark radial + faint grid ----
      const bg = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.7);
      bg.addColorStop(0, "#101218");
      bg.addColorStop(1, "#070708");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = "rgba(255,255,255,0.03)";
      ctx.lineWidth = 1;
      const gridStep = 40 * dpr;
      for (let gx = (offX % gridStep); gx < W; gx += gridStep) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, H);
        ctx.stroke();
      }
      for (let gy = (offY % gridStep); gy < H; gy += gridStep) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(W, gy);
        ctx.stroke();
      }

      const centerPath = () => {
        ctx.beginPath();
        pts.forEach(([x, y], i) => {
          const px = sx(x);
          const py = sy(y);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
      };

      // ---- asphalt (thick stroke along centerline) ----
      centerPath();
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = "#23242b";
      ctx.lineWidth = hw * 2 * scale;
      ctx.stroke();

      // subtle inner asphalt sheen
      centerPath();
      ctx.strokeStyle = "rgba(255,255,255,0.025)";
      ctx.lineWidth = hw * 1.4 * scale;
      ctx.stroke();

      // ---- curbs: red/white striped edges ----
      const drawCurb = (edge: [number, number][]) => {
        // white base
        ctx.beginPath();
        edge.forEach(([x, y], i) => {
          const px = sx(x);
          const py = sy(y);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.strokeStyle = "rgba(235,235,240,0.9)";
        ctx.lineWidth = Math.max(2, 3 * dpr);
        ctx.setLineDash([]);
        ctx.stroke();
        // red dashes over it
        ctx.strokeStyle = "rgba(220,40,40,0.95)";
        ctx.setLineDash([10 * dpr, 10 * dpr]);
        ctx.stroke();
        ctx.setLineDash([]);
      };
      drawCurb(inner);
      drawCurb(outer);

      // ---- faint centre lane marking ----
      centerPath();
      ctx.setLineDash([7 * dpr, 11 * dpr]);
      ctx.strokeStyle = "rgba(255,255,255,0.14)";
      ctx.lineWidth = Math.max(1, 1.5 * dpr);
      ctx.stroke();
      ctx.setLineDash([]);

      // ---- ideal racing line (red, toggled) ----
      if (showRacingLineRef.current) {
        ctx.beginPath();
        racePts.forEach(([x, y], i) => {
          const px = sx(x);
          const py = sy(y);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.strokeStyle = "rgba(255,59,48,0.95)";
        ctx.lineWidth = Math.max(2, 2.5 * dpr);
        ctx.lineJoin = "round";
        ctx.stroke();
      }

      // ---- checkered start/finish line ----
      {
        const [ix, iy] = inner[0];
        const [ox, oy] = outer[0];
        const cells = 8;
        for (let c = 0; c < cells; c++) {
          const t0 = c / cells;
          const t1 = (c + 1) / cells;
          const ax = sx(ix + (ox - ix) * t0);
          const ay = sy(iy + (oy - iy) * t0);
          const bx = sx(ix + (ox - ix) * t1);
          const by = sy(iy + (oy - iy) * t1);
          // thickness along the tangent
          const dx = (bx - ax) * 0.9;
          const dy = (by - ay) * 0.9;
          const tnx = -(by - ay);
          const tny = bx - ax;
          const tl = Math.hypot(tnx, tny) || 1;
          const depth = ((tnx / tl) * (hw * 0.18 * scale));
          const depthY = ((tny / tl) * (hw * 0.18 * scale));
          ctx.fillStyle = c % 2 === 0 ? "rgba(245,245,245,0.95)" : "rgba(20,20,22,0.95)";
          ctx.beginPath();
          ctx.moveTo(ax - depth, ay - depthY);
          ctx.lineTo(ax + dx - depth, ay + dy - depthY);
          ctx.lineTo(ax + dx + depth, ay + dy + depthY);
          ctx.lineTo(ax + depth, ay + depthY);
          ctx.closePath();
          ctx.fill();
        }
      }

      // ---- velocity-predicted, frame-rate-independent smoothing ----
      const tele = telemetryRef.current;
      if (tele) {
        const target = tele.cars;
        if (rendered.current.length !== target.length) {
          rendered.current = target.map((c) => ({
            x: c.x, y: c.y, svx: 0, svy: 0, ltx: c.x, lty: c.y, theta: c.theta, v: c.v, offtrack: c.offtrack, init: true,
          }));
          trails.current = target.map(() => []);
        }
        // Measure each car's REAL on-screen speed from how far the truth moved per
        // wall-second, then chase a predicted current position. Tracks correctly
        // at any sim-speed (no infield corner-cutting at 8x) and if the CPU lags.
        const isNew = tele.step !== frameStep;
        const dtWall = isNew ? Math.max(0.004, (now - frameTime) / 1000) : 0;
        if (isNew) { frameStep = tele.step; frameTime = now; }
        const age = Math.min(0.2, (now - frameTime) / 1000);
        const corr = 1 - Math.exp(-14 * delta);
        for (let i = 0; i < target.length; i++) {
          const r = rendered.current[i];
          const t = target[i];
          if (isNew) {
            const jump = Math.hypot(t.x - r.ltx, t.y - r.lty);
            if (r.init || jump > track.halfWidth * 4) {
              r.x = t.x; r.y = t.y; r.theta = t.theta; r.svx = 0; r.svy = 0; r.init = false;
              trails.current[i] = [];
            } else {
              r.svx += ((t.x - r.ltx) / dtWall - r.svx) * 0.5; // mild smoothing
              r.svy += ((t.y - r.lty) / dtWall - r.svy) * 0.5;
            }
            r.ltx = t.x; r.lty = t.y;
          }
          const predX = t.x + r.svx * age;
          const predY = t.y + r.svy * age;
          r.x += (predX - r.x) * corr;
          r.y += (predY - r.y) * corr;
          r.theta = lerpAngle(r.theta, t.theta, corr);
          r.v = t.v;
          r.offtrack = t.offtrack;

          const tr = trails.current[i];
          tr.push({ x: r.x, y: r.y });
          if (tr.length > 22) tr.shift();
        }
      }

      // ---- trails ----
      rendered.current.forEach((c, i) => {
        const tr = trails.current[i];
        if (!tr || tr.length < 2 || c.offtrack) return;
        const color = carColorCss(i);
        for (let j = 1; j < tr.length; j++) {
          const a = tr[j - 1];
          const b = tr[j];
          ctx.beginPath();
          ctx.moveTo(sx(a.x), sy(a.y));
          ctx.lineTo(sx(b.x), sy(b.y));
          ctx.strokeStyle = color;
          ctx.globalAlpha = (j / tr.length) * 0.35;
          ctx.lineWidth = Math.max(1, 2 * dpr);
          ctx.lineCap = "round";
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      });

      // ---- sensor rays (lidar) ----
      const sel = selectedRef.current;
      const tele2 = telemetryRef.current;
      const angles = track.rayAngles;
      const range = track.rayRange;
      if (tele2 && angles) {
        rendered.current.forEach((c, i) => {
          const isSel = i === sel;
          if (!showSensorsRef.current && !isSel) return;
          if (c.offtrack) return;
          const sensors = tele2.cars[i]?.sensors;
          if (!sensors) return;
          const cx = sx(c.x);
          const cy = sy(c.y);
          for (let j = 0; j < angles.length; j++) {
            const a = c.theta + angles[j];
            const dist = sensors[j] * range;
            const ex = sx(c.x + Math.cos(a) * dist);
            const ey = sy(c.y + Math.sin(a) * dist);
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(ex, ey);
            ctx.strokeStyle = isSel ? "rgba(56,229,255,0.85)" : "rgba(120,200,255,0.18)";
            ctx.lineWidth = isSel ? 1.5 * dpr : 1 * dpr;
            ctx.stroke();
            if (isSel) {
              ctx.beginPath();
              ctx.arc(ex, ey, 2.5 * dpr, 0, Math.PI * 2);
              ctx.fillStyle = "rgba(56,229,255,0.95)";
              ctx.fill();
            }
          }
        });
      }

      // ---- cars ----
      rendered.current.forEach((c, i) => {
        const cx = sx(c.x);
        const cy = sy(c.y);
        const ang = -c.theta; // y is flipped
        const speedFrac = Math.min(1, Math.abs(c.v) / 160);

        // selection ring
        if (i === sel) {
          ctx.beginPath();
          ctx.arc(cx, cy, CAR_L * scale * 0.95, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(56,229,255,0.9)";
          ctx.lineWidth = 2 * dpr;
          ctx.stroke();
        }

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(ang);
        drawCar(ctx, scale, carColorCss(i), c.offtrack, speedFrac);
        ctx.restore();

        // id label above the car
        ctx.fillStyle = i === sel ? "rgba(56,229,255,1)" : "rgba(255,255,255,0.6)";
        ctx.font = `${(i === sel ? 11 : 10) * dpr}px ui-monospace, monospace`;
        ctx.textAlign = "center";
        ctx.fillText(String(i), cx, cy - CAR_W * scale - 5 * dpr);
      });

      drawnOnce = true;
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
    };
  }, [track, telemetryRef]);

  return (
    <div
      ref={wrapRef}
      className="relative h-full w-full min-h-0 min-w-0 overflow-hidden rounded-xl border bg-[#070708]"
    >
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        className="absolute inset-0 block h-full w-full cursor-pointer"
      />
    </div>
  );
}
