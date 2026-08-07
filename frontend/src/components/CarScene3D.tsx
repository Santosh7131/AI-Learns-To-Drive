import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line, Sky, RoundedBox } from "@react-three/drei";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three-stdlib";
import type { Telemetry, TrackGeometry } from "@/lib/api";
import {
  computeTrackEdges,
  resampleClosed,
  racingLine,
  terrainHeight,
  type Pt,
  type TerrainComp,
} from "@/lib/trackGeometry";
import { carColorCss, lerpAngle } from "@/lib/carViz";

interface Props {
  track: TrackGeometry;
  telemetryRef: React.MutableRefObject<Telemetry | null>;
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  chase: boolean;
  numCars?: number;
  active?: boolean;         // running → continuous render; paused → on-demand (GPU idles)
  showRacingLine?: boolean; // render-only ideal line (cars never observe it)
}

const TRACK_LIFT = 0.45; // flat track surface height
const CAR_LIFT = 0.5; // detailed-car ride height so the wheels sit planted on the road
const INSTANCED_LIFT = TRACK_LIFT + 0.6; // simple body box rests its floor on the road

function carColor(id: number, total = 20) {
  return new THREE.Color(carColorCss(id, total));
}

// sim (x, y) -> world (x, height, y); heading rotates about +Y by -theta.

// ---------------------------------------------------------------- terrain
function Terrain({ track }: { track: TrackGeometry }) {
  const geom = useMemo(() => {
    const { minX, maxX, minY, maxY } = track.bounds;
    const cx = (minX + maxX) / 2;
    const cz = (minY + maxY) / 2;
    const ext = Math.max(maxX - minX, maxY - minY);
    const size = ext * 2.6;
    const seg = 150;
    const g = new THREE.PlaneGeometry(size, size, seg, seg);
    g.rotateX(-Math.PI / 2);
    const pos = g.attributes.position as THREE.BufferAttribute;
    const terrain = track.terrain as TerrainComp[];

    // centerline as flat arrays for a fast nearest-point search
    const cl = track.centerline;
    const M = cl.length;
    const clx = new Float64Array(M);
    const clz = new Float64Array(M);
    for (let k = 0; k < M; k++) {
      clx[k] = cl[k][0];
      clz[k] = cl[k][1];
    }
    const hw = track.halfWidth;
    const cell = size / seg;                 // terrain grid cell size
    const flatR = hw + Math.max(10, cell * 1.6); // fully-flat apron radius (covers grid coarseness)
    const shoulder = hw * 2.4;               // long, gentle verge out to the hills
    const reach = flatR + shoulder;
    const reach2 = reach * reach;
    const footprint = TRACK_LIFT - 0.12;     // flat grass, just below the flat road

    // The road is flat, so flatten the grass around it to road level and ramp
    // smoothly out to the natural rolling hills further away. No per-point road
    // elevation, so nothing can poke through and there are no seams.
    for (let i = 0; i < pos.count; i++) {
      const wx = pos.getX(i) + cx;
      const wz = pos.getZ(i) + cz;
      const natural = terrainHeight(terrain, wx, wz) - 0.15;
      let best = Infinity;
      for (let k = 0; k < M; k++) {
        const dx = wx - clx[k];
        const dz = wz - clz[k];
        const d2 = dx * dx + dz * dz;
        if (d2 < best) best = d2;
      }
      if (best > reach2) {
        pos.setY(i, natural);
        continue;
      }
      const d = Math.sqrt(best);
      let h;
      if (d <= flatR) {
        h = footprint;
      } else {
        const tt = (d - flatR) / shoulder;
        const sm = tt * tt * (3 - 2 * tt); // smoothstep verge up to natural grass
        h = footprint + (natural - footprint) * sm;
      }
      pos.setY(i, h);
    }
    g.computeVertexNormals();
    return g;
  }, [track]);

  useEffect(() => () => geom.dispose(), [geom]); // free GPU buffers on track change/unmount

  const center: [number, number, number] = [
    (track.bounds.minX + track.bounds.maxX) / 2,
    0,
    (track.bounds.minY + track.bounds.maxY) / 2,
  ];
  return (
    <mesh geometry={geom} position={center} receiveShadow>
      <meshStandardMaterial color="#3a5a2c" roughness={1} metalness={0} />
    </mesh>
  );
}

// ---------------------------------------------------------------- track mesh
// The road is rendered FLAT (constant Y). Elevation drove two whole classes of
// bug — edge/curb geometry spiking into triangles at sharp corners, and the car
// grounding at the coarse physics height while the mesh used the smoothed one,
// so cars sank into slopes. A flat surface removes both, permanently.
function TrackMesh({ track }: { track: TrackGeometry }) {
  const { geometry, left, right, centerLine } = useMemo(() => {
    const hw = track.halfWidth;
    // resample to a smooth, dense curve for rendering (physics keeps coarse pts)
    const { pts } = resampleClosed(track.centerline as Pt[], track.elevation, 6);
    const M = pts.length;
    const { left: eL, right: eR } = computeTrackEdges(pts, hw);
    const Y = TRACK_LIFT; // one flat plane — no per-vertex elevation
    const left: [number, number, number][] = eL.map(([x, y]) => [x, Y, y]);
    const right: [number, number, number][] = eR.map(([x, y]) => [x, Y, y]);

    // ---- asphalt ribbon (flat) ----
    const positions: number[] = [];
    const idx: number[] = [];
    for (let i = 0; i < M; i++) {
      positions.push(left[i][0], left[i][1], left[i][2]);
      positions.push(right[i][0], right[i][1], right[i][2]);
    }
    for (let i = 0; i < M; i++) {
      const a = 2 * i, b = 2 * i + 1, c = 2 * ((i + 1) % M), d = 2 * ((i + 1) % M) + 1;
      idx.push(a, b, c, b, d, c);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    g.setIndex(idx);
    g.computeVertexNormals();

    const centerLine: [number, number, number][] = pts.map(([x, y]) => [x, Y + 0.04, y]);
    left.push(left[0]);
    right.push(right[0]);
    return { geometry: g, left, right, centerLine };
  }, [track]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  const startLine = useMemo<[number, number, number][]>(() => [left[0], right[0]], [left, right]);

  // Kerbs = a solid white edge line with a red DASHED line laid over it, so the
  // red dashes + white gaps read as an alternating red/white kerb. Pure lines
  // that follow the edge exactly — no offset geometry, so nothing can fold into
  // the triangle spikes we had before. Mirrors the 2D renderer's curb.
  return (
    <group>
      <mesh geometry={geometry} receiveShadow castShadow>
        <meshStandardMaterial color="#26272c" roughness={0.97} metalness={0} side={THREE.DoubleSide} />
      </mesh>
      <Line points={left} color="#eef0f4" lineWidth={4} />
      <Line points={right} color="#eef0f4" lineWidth={4} />
      <Line points={left} color="#c81e1e" lineWidth={4} dashed dashSize={5} gapSize={5} />
      <Line points={right} color="#c81e1e" lineWidth={4} dashed dashSize={5} gapSize={5} />
      <Line points={centerLine} color="#f4d03f" lineWidth={1.4} dashed dashSize={7} gapSize={10} transparent opacity={0.55} />
      <Line points={startLine} color="#ffffff" lineWidth={6} />
    </group>
  );
}

// ---------------------------------------------------------- detailed F1 car
// Smooth, rounded single-seater. Rounded panels + restrained metallic paint;
// dark carbon for floor/wings/wheels. Body length along local +X, cabin +Y.
const CARBON = "#16171c";
const RIM = "#c7ccd6";
const TIRE = "#0c0d10";
function CarMesh({ color, selected }: { color: THREE.Color; selected: boolean }) {
  const paint = (
    <meshStandardMaterial color={color} roughness={0.3} metalness={0.6} emissive={color} emissiveIntensity={selected ? 0.35 : 0.05} />
  );
  const carbon = <meshStandardMaterial color={CARBON} roughness={0.55} metalness={0.2} />;
  return (
    <group>
      {/* floor / chassis */}
      <RoundedBox args={[9.2, 0.4, 3.0]} radius={0.14} smoothness={3} position={[-0.2, 0.5, 0]} castShadow>
        {carbon}
      </RoundedBox>
      {/* nose cone */}
      <mesh position={[3.9, 0.85, 0]} rotation={[0, 0, -Math.PI / 2]} castShadow>
        <cylinderGeometry args={[0.16, 0.7, 3.2, 20]} />
        {paint}
      </mesh>
      {/* cockpit tub */}
      <RoundedBox args={[4.8, 0.95, 1.9]} radius={0.3} smoothness={3} position={[-0.4, 1.05, 0]} castShadow>
        {paint}
      </RoundedBox>
      {/* engine cover, tapering back */}
      <RoundedBox args={[3.4, 1.15, 1.0]} radius={0.28} smoothness={3} position={[-2.6, 1.2, 0]} castShadow>
        {paint}
      </RoundedBox>
      {/* airbox scoop above the driver */}
      <RoundedBox args={[1.2, 0.7, 0.7]} radius={0.2} smoothness={3} position={[-1.5, 1.85, 0]} castShadow>
        {carbon}
      </RoundedBox>
      {/* side pods */}
      {[1.35, -1.35].map((z, i) => (
        <RoundedBox key={i} args={[3.2, 0.9, 0.95]} radius={0.28} smoothness={3} position={[-1.1, 0.95, z]} castShadow>
          {paint}
        </RoundedBox>
      ))}
      {/* cockpit opening + halo */}
      <mesh position={[0.35, 1.55, 0]}>
        <sphereGeometry args={[0.55, 20, 14]} />
        <meshStandardMaterial color="#0a0c12" roughness={0.12} metalness={0.75} />
      </mesh>
      <mesh position={[0.9, 1.75, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.72, 0.07, 12, 28, Math.PI]} />
        {carbon}
      </mesh>
      {/* front wing + endplates */}
      <RoundedBox args={[1.4, 0.12, 4.1]} radius={0.05} smoothness={2} position={[4.85, 0.42, 0]} castShadow>
        {carbon}
      </RoundedBox>
      {[2.02, -2.02].map((z, i) => (
        <RoundedBox key={i} args={[1.4, 0.78, 0.12]} radius={0.05} smoothness={2} position={[4.85, 0.68, z]}>
          {paint}
        </RoundedBox>
      ))}
      {/* rear wing + endplates */}
      <RoundedBox args={[1.5, 0.16, 3.4]} radius={0.06} smoothness={2} position={[-4.7, 2.35, 0]} castShadow>
        {paint}
      </RoundedBox>
      {[1.62, -1.62].map((z, i) => (
        <RoundedBox key={i} args={[1.6, 1.25, 0.12]} radius={0.06} smoothness={2} position={[-4.7, 1.85, z]}>
          {carbon}
        </RoundedBox>
      ))}
      {/* wheels */}
      {([
        [2.9, 1.95],
        [2.9, -1.95],
        [-3.0, 1.95],
        [-3.0, -1.95],
      ] as const).map(([wx, wz], i) => (
        <group key={i} position={[wx, 0.9, wz]} rotation={[Math.PI / 2, 0, 0]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.92, 0.92, 0.9, 24]} />
            <meshStandardMaterial color={TIRE} roughness={0.85} metalness={0.05} />
          </mesh>
          <mesh>
            <cylinderGeometry args={[0.5, 0.5, 0.92, 16]} />
            <meshStandardMaterial color={RIM} metalness={0.85} roughness={0.28} />
          </mesh>
        </group>
      ))}
      {selected && (
        <mesh position={[0, 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[6.4, 7.2, 48]} />
          <meshBasicMaterial color="#2563eb" transparent opacity={0.9} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  );
}

// ---------------------------------------------------------------- all cars
interface RState {
  x: number;
  y: number;
  svx: number; // measured on-screen velocity (world units/sec) — sim-speed independent
  svy: number;
  ltx: number; // last authoritative (truth) position, used to measure svx
  lty: number;
  theta: number;
  roll: number;
  pitch: number;
  ry: number; // render height (suspension spring)
  rvy: number; // vertical velocity
  init: boolean;
}
function Cars({ telemetryRef, track, selectedId, onSelect, numCars = 20 }: Omit<Props, "chase">) {
  const outer = useRef<(THREE.Group | null)[]>([]);
  const inner = useRef<(THREE.Group | null)[]>([]);
  const state = useRef<RState[]>([]);
  const fm = useRef({ step: -1, time: 0 }); // last telemetry frame seen + when (for velocity from truth deltas)

  useFrame((_, rawDelta) => {
    const tele = telemetryRef.current;
    if (!tele) return;
    const cars = tele.cars;
    const delta = Math.min(rawDelta, 0.05);
    if (state.current.length !== cars.length) {
      state.current = cars.map((c) => ({
        x: c.x, y: c.y, svx: 0, svy: 0, ltx: c.x, lty: c.y, theta: c.theta, roll: 0, pitch: 0,
        ry: CAR_LIFT, rvy: 0, init: true,
      }));
    }
    // A new telemetry frame arrives every sim step; at high sim-speed many steps
    // pass per rendered frame. Measure each car's REAL on-screen speed from how
    // far the truth moved per wall-second, then chase a predicted current
    // position. This tracks correctly at any sim-speed (no more corner-cutting
    // across the infield at 8x), and degrades gracefully if the CPU can't keep up.
    const now = performance.now();
    const isNew = tele.step !== fm.current.step;
    const dtWall = isNew ? Math.max(0.004, (now - fm.current.time) / 1000) : 0;
    if (isNew) { fm.current.step = tele.step; fm.current.time = now; }
    const age = Math.min(0.2, (now - fm.current.time) / 1000); // time since that frame (for prediction)
    const corr = 1 - Math.exp(-14 * delta); // reconcile toward the predicted truth
    const tilt = 1 - Math.exp(-6 * delta); // body roll/pitch smoothing

    for (let i = 0; i < cars.length; i++) {
      const o = outer.current[i];
      const inn = inner.current[i];
      const r = state.current[i];
      const t = cars[i];
      if (!o || !r) continue;

      if (isNew) {
        const jump = Math.hypot(t.x - r.ltx, t.y - r.lty);
        if (r.init || jump > track.halfWidth * 4) {
          // first frame or a respawn teleport: snap, drop any carried velocity
          r.x = t.x; r.y = t.y; r.theta = t.theta; r.svx = 0; r.svy = 0;
          r.ry = CAR_LIFT; r.rvy = 0; r.init = false;
        } else {
          const nvx = (t.x - r.ltx) / dtWall;
          const nvy = (t.y - r.lty) / dtWall;
          r.svx += (nvx - r.svx) * 0.5; // mild smoothing of the measurement
          r.svy += (nvy - r.svy) * 0.5;
        }
        r.ltx = t.x; r.lty = t.y;
      }
      // follow the predicted current position (truth + measured velocity * age)
      const predX = t.x + r.svx * age;
      const predY = t.y + r.svy * age;
      r.x += (predX - r.x) * corr;
      r.y += (predY - r.y) * corr;
      r.theta = lerpAngle(r.theta, t.theta, corr);

      // vertical suspension spring: subtle travel that settles onto the flat road
      const groundY = CAR_LIFT;
      if (r.ry > groundY + 0.2) {
        r.rvy -= 30 * delta; // airborne — gentle gravity
      } else {
        r.rvy += (groundY - r.ry) * 80 * delta; // grounded — soft suspension
        r.rvy *= Math.exp(-13 * delta);
      }
      r.ry += r.rvy * delta;
      if (r.ry < groundY) {
        r.ry = groundY;
        if (r.rvy < 0) r.rvy = -r.rvy * 0.12;
      }
      if (r.ry > groundY + 2.5) r.ry = groundY + 2.5; // cap air — stays car-like

      o.position.set(r.x, r.ry, r.y);
      o.rotation.y = -r.theta;

      // flat road: pitch comes only from throttle/brake weight transfer, roll from steering
      const targetPitch =
        ((t.accel ?? 0) - (t.brake ?? 0)) * 0.05 -
        Math.max(-0.1, Math.min(0.1, r.rvy * 0.012));
      const targetRoll = -(t.steer ?? 0) * 0.18;
      r.pitch += (targetPitch - r.pitch) * tilt;
      r.roll += (targetRoll - r.roll) * tilt;
      if (inn) {
        inn.rotation.z = r.pitch;
        inn.rotation.x = r.roll;
      }
    }
  });

  return (
    <>
      {Array.from({ length: numCars }).map((_, i) => (
        <group key={i} ref={(el) => (outer.current[i] = el)}>
          <group
            ref={(el) => (inner.current[i] = el)}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(i === selectedId ? null : i);
            }}
          >
            <group scale={1.3}>
              <CarMesh color={carColor(i, numCars)} selected={i === selectedId} />
            </group>
          </group>
        </group>
      ))}
    </>
  );
}

// ------------------------------------------------- instanced cars (big fleets)
// For hundreds–thousands of cars, one InstancedMesh (2 draw calls) replaces the
// per-car detailed groups. Same velocity-extrapolation smoothing as Cars.
interface ISt { x: number; y: number; svx: number; svy: number; ltx: number; lty: number; theta: number; ry: number; rvy: number; init: boolean }

function InstancedCars({ track, telemetryRef, selectedId, onSelect, numCars = 20 }: Omit<Props, "chase">) {
  const bodyRef = useRef<THREE.InstancedMesh>(null);
  const cabinRef = useRef<THREE.InstancedMesh>(null);
  const state = useRef<ISt[]>([]);
  const fm = useRef({ step: -1, time: 0 });
  const off = useRef<Uint8Array>(new Uint8Array(0));
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const grey = useMemo(() => new THREE.Color("#5b606b"), []);
  // two-tone car silhouette: a rounded body + a darker cabin baked slightly up/back
  const bodyGeo = useMemo(() => new RoundedBoxGeometry(8.0, 1.2, 3.4, 3, 0.34), []);
  const cabinGeo = useMemo(() => {
    const g = new RoundedBoxGeometry(3.7, 0.95, 2.6, 3, 0.3);
    g.translate(-0.4, 1.0, 0);
    return g;
  }, []);
  useEffect(() => () => { bodyGeo.dispose(); cabinGeo.dispose(); }, [bodyGeo, cabinGeo]);

  // (re)assign per-instance colors when the fleet size changes
  useEffect(() => {
    const bm = bodyRef.current;
    if (!bm) return;
    for (let i = 0; i < numCars; i++) bm.setColorAt(i, carColor(i, numCars));
    if (bm.instanceColor) bm.instanceColor.needsUpdate = true;
    off.current = new Uint8Array(numCars);
  }, [numCars]);

  useFrame((_, rawDelta) => {
    const tele = telemetryRef.current;
    const bm = bodyRef.current;
    const cm = cabinRef.current;
    if (!tele || !bm || !cm) return;
    const cars = tele.cars;
    const delta = Math.min(rawDelta, 0.05);
    if (state.current.length !== cars.length) {
      state.current = cars.map((c) => ({ x: c.x, y: c.y, svx: 0, svy: 0, ltx: c.x, lty: c.y, theta: c.theta, ry: INSTANCED_LIFT, rvy: 0, init: true }));
    }
    // measure real on-screen velocity from truth deltas (see Cars) so cars track
    // correctly at any sim-speed instead of cutting across the infield at 8x.
    const now = performance.now();
    const isNew = tele.step !== fm.current.step;
    const dtWall = isNew ? Math.max(0.004, (now - fm.current.time) / 1000) : 0;
    if (isNew) { fm.current.step = tele.step; fm.current.time = now; }
    const age = Math.min(0.2, (now - fm.current.time) / 1000);
    const corr = 1 - Math.exp(-14 * delta);
    const n = Math.min(numCars, cars.length);
    let colorDirty = false;
    for (let i = 0; i < n; i++) {
      const r = state.current[i];
      const t = cars[i];
      if (!r) continue;
      if (isNew) {
        const jump = Math.hypot(t.x - r.ltx, t.y - r.lty);
        if (r.init || jump > track.halfWidth * 4) {
          r.x = t.x; r.y = t.y; r.theta = t.theta; r.svx = 0; r.svy = 0;
          r.ry = INSTANCED_LIFT; r.rvy = 0; r.init = false;
        } else {
          r.svx += ((t.x - r.ltx) / dtWall - r.svx) * 0.5;
          r.svy += ((t.y - r.lty) / dtWall - r.svy) * 0.5;
        }
        r.ltx = t.x; r.lty = t.y;
      }
      const predX = t.x + r.svx * age;
      const predY = t.y + r.svy * age;
      r.x += (predX - r.x) * corr;
      r.y += (predY - r.y) * corr;
      r.theta = lerpAngle(r.theta, t.theta, corr);
      const groundY = INSTANCED_LIFT;
      if (r.ry > groundY + 0.2) r.rvy -= 30 * delta;
      else { r.rvy += (groundY - r.ry) * 80 * delta; r.rvy *= Math.exp(-13 * delta); }
      r.ry += r.rvy * delta;
      if (r.ry < groundY) { r.ry = groundY; if (r.rvy < 0) r.rvy = -r.rvy * 0.12; }
      if (r.ry > groundY + 2.5) r.ry = groundY + 2.5;
      dummy.position.set(r.x, r.ry, r.y);
      dummy.rotation.set(0, -r.theta, 0);
      dummy.updateMatrix();
      bm.setMatrixAt(i, dummy.matrix);
      cm.setMatrixAt(i, dummy.matrix);
      const o = t.offtrack ? 1 : 0;
      if (off.current[i] !== o) {
        off.current[i] = o;
        bm.setColorAt(i, o ? grey : carColor(i, numCars));
        colorDirty = true;
      }
    }
    bm.count = n;
    cm.count = n;
    bm.instanceMatrix.needsUpdate = true;
    cm.instanceMatrix.needsUpdate = true;
    if (colorDirty && bm.instanceColor) bm.instanceColor.needsUpdate = true;
  });

  return (
    <group>
      <instancedMesh
        ref={bodyRef}
        args={[bodyGeo, undefined, numCars]}
        castShadow
        receiveShadow
        onClick={(e) => {
          e.stopPropagation();
          if (e.instanceId != null) onSelect(e.instanceId === selectedId ? null : e.instanceId);
        }}
      >
        <meshStandardMaterial metalness={0.5} roughness={0.4} />
      </instancedMesh>
      <instancedMesh ref={cabinRef} args={[cabinGeo, undefined, numCars]} castShadow>
        <meshStandardMaterial color="#0b0d13" metalness={0.6} roughness={0.25} />
      </instancedMesh>
    </group>
  );
}

// ---------------------------------------------------------------- chase cam
function CameraRig({
  telemetryRef,
  followId,
}: {
  telemetryRef: React.MutableRefObject<Telemetry | null>;
  followId: number | null;
}) {
  const { camera } = useThree();
  const camPos = useRef(new THREE.Vector3());
  const lookAt = useRef(new THREE.Vector3());
  const desired = useRef(new THREE.Vector3());
  const target = useRef(new THREE.Vector3());
  const inited = useRef(false);
  useFrame((_, rawDelta) => {
    if (followId === null) {
      inited.current = false;
      return;
    }
    const c = telemetryRef.current?.cars[followId];
    if (!c) return;
    const dt = Math.min(rawDelta, 0.05);
    const fx = Math.cos(c.theta);
    const fz = Math.sin(c.theta);
    const h = TRACK_LIFT; // flat road — constant camera height, no terrain jitter
    desired.current.set(c.x - fx * 30, h + 13, c.y - fz * 30);
    target.current.set(c.x + fx * 9, h + 2.5, c.y + fz * 9); // look slightly ahead
    if (!inited.current) {
      camPos.current.copy(desired.current);
      lookAt.current.copy(target.current);
      inited.current = true;
    }
    // frame-rate-independent critical damping — smooth, no terrain jitter
    camPos.current.lerp(desired.current, 1 - Math.exp(-3.5 * dt));
    lookAt.current.lerp(target.current, 1 - Math.exp(-6 * dt));
    camera.position.copy(camPos.current);
    camera.lookAt(lookAt.current);
  });
  return null;
}

// ---------------------------------------------------------------- sensor rays
function SensorRays({
  track,
  telemetryRef,
  selectedId,
}: {
  track: TrackGeometry;
  telemetryRef: React.MutableRefObject<Telemetry | null>;
  selectedId: number | null;
}) {
  const ref = useRef<THREE.LineSegments>(null);
  const positions = useMemo(() => new Float32Array(track.numRays * 2 * 3), [track.numRays]);
  useFrame(() => {
    const seg = ref.current;
    if (!seg) return;
    const c = selectedId === null ? null : telemetryRef.current?.cars[selectedId];
    if (!c || !c.sensors) {
      seg.visible = false;
      return;
    }
    seg.visible = true;
    const h0 = TRACK_LIFT + 1.4; // flat road
    const n = Math.min(track.numRays, c.sensors.length);
    for (let j = 0; j < track.numRays; j++) {
      const s = j < n ? c.sensors[j] : 0;
      const a = c.theta + track.rayAngles[j];
      const dist = Number.isFinite(s) ? s * track.rayRange : 0;
      const ex = c.x + Math.cos(a) * dist;
      const ez = c.y + Math.sin(a) * dist;
      positions[j * 6 + 0] = c.x;
      positions[j * 6 + 1] = h0;
      positions[j * 6 + 2] = c.y;
      positions[j * 6 + 3] = ex;
      positions[j * 6 + 4] = TRACK_LIFT + 1.0;
      positions[j * 6 + 5] = ez;
    }
    const attr = seg.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.array.set(positions);
    attr.needsUpdate = true;
  });
  return (
    <lineSegments ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#38e5ff" transparent opacity={0.9} />
    </lineSegments>
  );
}

// ---------------------------------------------------------------- scene root
export function CarScene3D({ track, telemetryRef, selectedId, onSelect, chase, numCars = 20, active = true, showRacingLine = false }: Props) {
  const { minX, maxX, minY, maxY } = track.bounds;
  const cx = (minX + maxX) / 2;
  const cz = (minY + maxY) / 2;
  const extent = Math.max(maxX - minX, maxY - minY);
  const dist = extent * 0.8;
  const followId = chase ? selectedId : null;
  const shadow = extent * 0.62;
  const sun: [number, number, number] = [cx - extent * 0.3, extent * 0.55, cz - extent * 0.22];

  // ideal racing line (render-only; the cars never observe it)
  const racePts = useMemo<[number, number, number][]>(() => {
    const rl = racingLine(track.centerline as Pt[], track.halfWidth);
    const pts = rl.map(([x, y]) => [x, TRACK_LIFT + 0.06, y] as [number, number, number]);
    pts.push(pts[0]);
    return pts;
  }, [track]);

  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = containerRef.current;
    const fire = () => window.dispatchEvent(new Event("resize"));
    // r3f's initial auto-measure can miss the container's real size inside a
    // scrolling grid layout, leaving the canvas at its 300x150 default. Bridge
    // any container resize to r3f, plus a few delayed nudges for first paint.
    const ro = el ? new ResizeObserver(fire) : null;
    ro?.observe(el as Element);
    const raf = requestAnimationFrame(fire);
    const t1 = setTimeout(fire, 300);
    const t2 = setTimeout(fire, 900);
    return () => {
      ro?.disconnect();
      cancelAnimationFrame(raf);
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden rounded-xl border bg-[#0a0e16]">
      <Canvas
        shadows
        dpr={[1, 1.75]}
        frameloop={active ? "always" : "demand"}
        gl={{ preserveDrawingBuffer: true, antialias: true }}
        camera={{ position: [cx, dist, cz + dist * 0.85], fov: 50, near: 1, far: extent * 8 }}
        onPointerMissed={() => onSelect(null)}
      >
        <fogExp2 attach="fog" args={["#9fb6c9", 0.00042]} />
        <Sky distance={45000} sunPosition={sun} turbidity={6} rayleigh={2.4} mieCoefficient={0.006} />
        <hemisphereLight args={["#cfe2ff", "#3a4a32", 0.55]} />
        <ambientLight intensity={0.25} />
        <directionalLight
          position={sun}
          intensity={2.2}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-bias={-0.0004}
          shadow-camera-near={1}
          shadow-camera-far={extent * 3}
          shadow-camera-left={-shadow}
          shadow-camera-right={shadow}
          shadow-camera-top={shadow}
          shadow-camera-bottom={-shadow}
        />

        <Terrain track={track} />
        <TrackMesh track={track} />
        {showRacingLine && <Line points={racePts} color="#ff3b30" lineWidth={3} />}
        {numCars > 60 ? (
          <InstancedCars track={track} telemetryRef={telemetryRef} selectedId={selectedId} onSelect={onSelect} numCars={numCars} />
        ) : (
          <Cars track={track} telemetryRef={telemetryRef} selectedId={selectedId} onSelect={onSelect} numCars={numCars} />
        )}
        <SensorRays track={track} telemetryRef={telemetryRef} selectedId={selectedId} />

        <CameraRig telemetryRef={telemetryRef} followId={followId} />
        <OrbitControls
          enabled={followId === null}
          makeDefault
          target={[cx, 0, cz]}
          enablePan
          screenSpacePanning={false}
          maxPolarAngle={Math.PI * 0.495}
          minDistance={8}
          maxDistance={extent * 4}
          enableDamping
          dampingFactor={0.08}
        />
      </Canvas>
    </div>
  );
}
