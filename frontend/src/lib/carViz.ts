// Shared car-visual helpers so 2D canvas, 3D scene, and the inspector all agree
// on per-car color and angle interpolation. (No Three.js import here — keeps this
// usable from the lightweight 2D view without pulling WebGL into that bundle.)

export function carHue(id: number, total = 20): number {
  return Math.round((id / total) * 360);
}

// CSS hsl string (commas so it parses in canvas, CSS, and THREE.Color alike)
export function carColorCss(id: number, total = 20): string {
  return `hsl(${carHue(id, total)}, 85%, 58%)`;
}

export function lerpAngle(a: number, b: number, t: number): number {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}
