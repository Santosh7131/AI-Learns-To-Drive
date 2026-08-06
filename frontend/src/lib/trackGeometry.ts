// Build the left/right edges of a track ribbon from its centerline.
//
// A naive constant-width offset folds over itself at corners tighter than the
// half-width (real circuits have many such corners, e.g. COTA hairpins). To
// avoid the fold, we clamp the *inner* edge's offset to a fraction of the local
// radius of curvature, so an edge can never cross past the centerline. The outer
// edge keeps full width, so corners still look natural.

export type Pt = [number, number];

export interface TerrainComp {
  a: number;
  wx: number;
  wy: number;
  p: number;
}

// Terrain height field — MUST match backend environment._terrain_height.
export function terrainHeight(terrain: TerrainComp[], x: number, y: number) {
  let h = 0;
  for (const t of terrain) h += t.a * Math.sin(t.wx * x + t.wy * y + t.p);
  return h;
}

// Gradient (dh/dx, dh/dy) of the terrain height field.
export function terrainGrad(terrain: TerrainComp[], x: number, y: number): [number, number] {
  let gx = 0;
  let gy = 0;
  for (const t of terrain) {
    const c = t.a * Math.cos(t.wx * x + t.wy * y + t.p);
    gx += c * t.wx;
    gy += c * t.wy;
  }
  return [gx, gy];
}

function wrap(a: number) {
  return ((a + Math.PI) % (2 * Math.PI)) - Math.PI;
}

// uniform Catmull-Rom interpolation of a scalar between p1 and p2
function cr(p0: number, p1: number, p2: number, p3: number, t: number) {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

/**
 * Resample a closed centerline (+ its per-point elevation) to `factor`x density
 * with a Catmull-Rom spline, so the rendered road/edges follow smooth curves
 * instead of visible facets. Visual only — the physics keeps the coarse points.
 */
export function resampleClosed(
  pts: Pt[],
  elev: number[],
  factor: number
): { pts: Pt[]; elev: number[] } {
  if (factor <= 1) return { pts: pts.slice(), elev: elev.slice() };
  const M = pts.length;
  const outPts: Pt[] = [];
  const outElev: number[] = [];
  for (let i = 0; i < M; i++) {
    const a = pts[(i - 1 + M) % M];
    const b = pts[i];
    const c = pts[(i + 1) % M];
    const d = pts[(i + 2) % M];
    const ea = elev[(i - 1 + M) % M];
    const eb = elev[i];
    const ec = elev[(i + 1) % M];
    const ed = elev[(i + 2) % M];
    for (let s = 0; s < factor; s++) {
      const t = s / factor;
      outPts.push([cr(a[0], b[0], c[0], d[0], t), cr(a[1], b[1], c[1], d[1], t)]);
      outElev.push(cr(ea, eb, ec, ed, t));
    }
  }
  return { pts: outPts, elev: outElev };
}

export function computeTrackEdges(
  pts: Pt[],
  halfWidth: number
): { left: Pt[]; right: Pt[] } {
  const M = pts.length;
  const left: Pt[] = [];
  const right: Pt[] = [];
  for (let i = 0; i < M; i++) {
    const [x, y] = pts[i];
    const [ax, ay] = pts[(i - 2 + M) % M];
    const [cx, cy] = pts[(i + 2) % M];

    // smoothed tangent + left-hand normal
    let tx = cx - ax;
    let ty = cy - ay;
    const tl = Math.hypot(tx, ty) || 1;
    tx /= tl;
    ty /= tl;
    const nx = -ty;
    const ny = tx;

    // local turn + radius of curvature
    const aIn = Math.atan2(y - ay, x - ax);
    const aOut = Math.atan2(cy - y, cx - x);
    const dphi = wrap(aOut - aIn);
    const ds = (Math.hypot(x - ax, y - ay) + Math.hypot(cx - x, cy - y)) / 2;
    const radius = ds / (Math.abs(dphi) + 1e-6);

    const clamp = Math.min(halfWidth, 0.8 * radius);
    // dphi > 0 => turning left => inner edge is the +normal (left) side
    const leftOff = dphi > 0 ? clamp : halfWidth;
    const rightOff = dphi > 0 ? halfWidth : clamp;

    left.push([x + nx * leftOff, y + ny * leftOff]);
    right.push([x - nx * rightOff, y - ny * rightOff]);
  }
  return { left, right };
}
