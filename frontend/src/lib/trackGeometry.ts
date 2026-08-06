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
