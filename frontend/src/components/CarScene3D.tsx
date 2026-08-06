import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line, Sky } from "@react-three/drei";
import * as THREE from "three";
import type { Telemetry, TrackGeometry } from "@/lib/api";
import {
  computeTrackEdges,
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
}

const TRACK_LIFT = 0.45; // track surface above terrain
const CAR_LIFT = 0.5; // car sits on the track surface

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
    const elev = track.elevation;
    const M = cl.length;
    const clx = new Float64Array(M);
    const clz = new Float64Array(M);
    const naturalAtCl = new Float64Array(M);
    for (let k = 0; k < M; k++) {
      clx[k] = cl[k][0];
      clz[k] = cl[k][1];
      naturalAtCl[k] = terrainHeight(terrain, cl[k][0], cl[k][1]);
    }
    const hw = track.halfWidth;
    const cell = size / seg;                 // terrain grid cell size
    const flatR = hw + Math.max(10, cell * 1.6); // fully-flat apron radius (covers grid coarseness)
    const shoulder = hw * 1.0;               // graded grass verge beyond the apron
    const reach = flatR + shoulder;
    const reach2 = reach * reach;

    // Carve the terrain to the road: flatten to just below the road surface
    // inside the track footprint, then ramp smoothly out to natural grass — so
    // the green never poked up through the asphalt. Bridge spans are left alone.
    for (let i = 0; i < pos.count; i++) {
      const wx = pos.getX(i) + cx;
      const wz = pos.getZ(i) + cz;
      const natural = terrainHeight(terrain, wx, wz) - 0.15;

      const flatR2 = flatR * flatR;
      let best = Infinity;
      let bk = 0;
      let minRoad = Infinity;   // lowest road surface among ALL nearby sections
      let bridgeOnly = true;    // every nearby section here is a flyover span
      for (let k = 0; k < M; k++) {
        const dx = wx - clx[k];
        const dz = wz - clz[k];
        const d2 = dx * dx + dz * dz;
        if (d2 < best) { best = d2; bk = k; }
        if (d2 < flatR2) {
          const isBridgeK = elev[k] - naturalAtCl[k] > 2.0;
          if (!isBridgeK) {
            bridgeOnly = false;
            if (elev[k] < minRoad) minRoad = elev[k]; // carve below the lowest road
          }
        }
      }
      if (best > reach2 || bridgeOnly) {
        pos.setY(i, natural); // open ground / under a bridge span
        continue;
      }
      const d = Math.sqrt(best);
      const road = (minRoad === Infinity ? elev[bk] : minRoad) + TRACK_LIFT;
      const footprint = road - 1.0; // flat apron, safely below every nearby road
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
function wrapPi(a: number) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}
function norm2(x: number, y: number): [number, number] {
  const L = Math.hypot(x, y) || 1;
  return [x / L, y / L];
}

function TrackMesh({ track }: { track: TrackGeometry }) {
  const { geometry, kerbGeom, left, right } = useMemo(() => {
    const pts = track.centerline as Pt[];
    const elev = track.elevation;
    const hw = track.halfWidth;
    const M = pts.length;
    const { left: eL, right: eR } = computeTrackEdges(pts, hw);
    const yAt = (i: number) => elev[i] + TRACK_LIFT;
    const left: [number, number, number][] = eL.map(([x, y], i) => [x, yAt(i), y]);
    const right: [number, number, number][] = eR.map(([x, y], i) => [x, yAt(i), y]);

    // ---- asphalt ribbon ----
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

    // ---- red/white kerbs at corners only (where the track actually turns) ----
    const dphi = new Array(M);
    for (let i = 0; i < M; i++) {
      const [px, py] = pts[(i - 1 + M) % M];
      const [x, y] = pts[i];
      const [nx, ny] = pts[(i + 1) % M];
      dphi[i] = Math.abs(wrapPi(Math.atan2(ny - y, nx - x) - Math.atan2(y - py, x - px)));
    }
    const kerbW = Math.min(4.5, hw * 0.16);
    const cornerThr = 0.02;
    const kp: number[] = [];
    const kc: number[] = [];
    const ki: number[] = [];
    let v = 0;
    const addKerb = (edge: [number, number, number][]) => {
      let stripe = 0;
      for (let i = 0; i < M; i++) {
        const i2 = (i + 1) % M;
        if (dphi[i] < cornerThr && dphi[i2] < cornerThr) { stripe = 0; continue; }
        // outward normal = (edge - centerline), normalized
        const oA = norm2(edge[i][0] - pts[i][0], edge[i][2] - pts[i][1]);
        const oB = norm2(edge[i2][0] - pts[i2][0], edge[i2][2] - pts[i2][1]);
        const yA = edge[i][1] + 0.06;
        const yB = edge[i2][1] + 0.06;
        kp.push(edge[i][0], yA, edge[i][2]);
        kp.push(edge[i][0] + oA[0] * kerbW, yA, edge[i][2] + oA[1] * kerbW);
        kp.push(edge[i2][0], yB, edge[i2][2]);
        kp.push(edge[i2][0] + oB[0] * kerbW, yB, edge[i2][2] + oB[1] * kerbW);
        const red = stripe % 2 === 0;
        const col = red ? [0.82, 0.13, 0.14] : [0.92, 0.92, 0.94];
        for (let k = 0; k < 4; k++) kc.push(col[0], col[1], col[2]);
        ki.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
        v += 4;
        stripe++;
      }
    };
    addKerb(left);
    addKerb(right);
    let kerbGeom: THREE.BufferGeometry | null = null;
    if (kp.length) {
      kerbGeom = new THREE.BufferGeometry();
      kerbGeom.setAttribute("position", new THREE.Float32BufferAttribute(kp, 3));
      kerbGeom.setAttribute("color", new THREE.Float32BufferAttribute(kc, 3));
      kerbGeom.setIndex(ki);
      kerbGeom.computeVertexNormals();
    }

    left.push(left[0]);
    right.push(right[0]);
    return { geometry: g, kerbGeom, left, right };
  }, [track]);

  useEffect(() => () => {
    geometry.dispose();
    kerbGeom?.dispose();
  }, [geometry, kerbGeom]);

  const centerLine = useMemo(() => {
    const elev = track.elevation;
    return track.centerline.map(
      ([x, y], i) => [x, elev[i] + TRACK_LIFT + 0.05, y] as [number, number, number]
    );
  }, [track]);

  const startLine = useMemo<[number, number, number][]>(() => [left[0], right[0]], [left, right]);

  return (
    <group>
      <mesh geometry={geometry} receiveShadow castShadow>
        <meshStandardMaterial color="#2e2f35" roughness={0.92} metalness={0.04} side={THREE.DoubleSide} />
      </mesh>
      {kerbGeom && (
        <mesh geometry={kerbGeom} receiveShadow>
          <meshStandardMaterial vertexColors roughness={0.6} metalness={0.0} side={THREE.DoubleSide} />
        </mesh>
      )}
      <Line points={left} color="#e9e9ee" lineWidth={2} transparent opacity={0.85} />
      <Line points={right} color="#e9e9ee" lineWidth={2} transparent opacity={0.85} />
      <Line points={centerLine} color="#ffffff" lineWidth={1.2} dashed dashSize={6} gapSize={9} transparent opacity={0.35} />
      <Line points={startLine} color="#ffffff" lineWidth={6} />
    </group>
  );
}

// ---------------------------------------------------------- detailed F1 car
const DARK = "#15161b";
const RIM = "#cfd3da";
function CarMesh({ color, selected }: { color: THREE.Color; selected: boolean }) {
  const body = (
    <meshStandardMaterial color={color} roughness={0.35} metalness={0.55} emissive={color} emissiveIntensity={selected ? 0.5 : 0.08} />
  );
  return (
    <group>
      <mesh position={[-0.2, 0.55, 0]} castShadow>
        <boxGeometry args={[9, 0.35, 3]} />
        <meshStandardMaterial color={DARK} roughness={0.7} />
      </mesh>
      <mesh position={[-0.6, 1.05, 0]} castShadow>
        <boxGeometry args={[4.6, 1.0, 2.1]} />
        {body}
      </mesh>
      <mesh position={[3.6, 0.95, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
        <cylinderGeometry args={[0.2, 0.85, 3.4, 16]} />
        {body}
      </mesh>
      <mesh position={[-2.7, 1.25, 0]} castShadow>
        <boxGeometry args={[3.2, 1.3, 1.2]} />
        {body}
      </mesh>
      <mesh position={[-3.4, 2.0, 0]} castShadow>
        <boxGeometry args={[2.0, 1.0, 0.12]} />
        {body}
      </mesh>
      {[1.35, -1.35].map((z, i) => (
        <mesh key={i} position={[-1.2, 0.95, z]} castShadow>
          <boxGeometry args={[3.0, 0.9, 0.9]} />
          {body}
        </mesh>
      ))}
      <mesh position={[0.4, 1.65, 0]}>
        <sphereGeometry args={[0.62, 16, 12]} />
        <meshStandardMaterial color="#0b0d14" roughness={0.15} metalness={0.7} />
      </mesh>
      <mesh position={[1.0, 1.9, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.7, 0.08, 8, 24, Math.PI]} />
        <meshStandardMaterial color={DARK} metalness={0.5} roughness={0.4} />
      </mesh>
      <mesh position={[4.7, 0.45, 0]} castShadow>
        <boxGeometry args={[1.3, 0.12, 4.0]} />
        <meshStandardMaterial color={DARK} roughness={0.5} />
      </mesh>
      {[2.0, -2.0].map((z, i) => (
        <mesh key={i} position={[4.7, 0.7, z]}>
          <boxGeometry args={[1.3, 0.8, 0.12]} />
          {body}
        </mesh>
      ))}
      <mesh position={[-4.6, 2.5, 0]} castShadow>
        <boxGeometry args={[1.3, 0.16, 3.4]} />
        {body}
      </mesh>
      {[1.6, -1.6].map((z, i) => (
        <mesh key={i} position={[-4.6, 2.0, z]}>
          <boxGeometry args={[1.5, 1.3, 0.12]} />
          <meshStandardMaterial color={DARK} roughness={0.5} />
        </mesh>
      ))}
      {([
        [2.9, 1.9],
        [2.9, -1.9],
        [-3.0, 1.9],
        [-3.0, -1.9],
      ] as const).map(([wx, wz], i) => (
        <group key={i} position={[wx, 0.9, wz]} rotation={[Math.PI / 2, 0, 0]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.95, 0.95, 0.85, 18]} />
            <meshStandardMaterial color="#0c0c0f" roughness={0.85} />
          </mesh>
          <mesh>
            <cylinderGeometry args={[0.52, 0.52, 0.88, 12]} />
            <meshStandardMaterial color={RIM} metalness={0.8} roughness={0.3} />
          </mesh>
        </group>
      ))}
      {selected && (
        <mesh position={[0, 0.12, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <torusGeometry args={[7.2, 0.45, 10, 44]} />
          <meshStandardMaterial color="#38e5ff" emissive="#38e5ff" emissiveIntensity={1.2} />
        </mesh>
      )}
    </group>
  );
}

// ---------------------------------------------------------------- all cars
interface RState {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  theta: number;
  roll: number;
  pitch: number;
  init: boolean;
}
function Cars({ telemetryRef, track, selectedId, onSelect, numCars = 20 }: Omit<Props, "chase">) {
  const outer = useRef<(THREE.Group | null)[]>([]);
  const inner = useRef<(THREE.Group | null)[]>([]);
  const state = useRef<RState[]>([]);

  useFrame((_, rawDelta) => {
    const tele = telemetryRef.current;
    if (!tele) return;
    const cars = tele.cars;
    const delta = Math.min(rawDelta, 0.05);
    if (state.current.length !== cars.length) {
      state.current = cars.map((c) => ({
        x: c.x, y: c.y, z: c.z ?? 0, vx: 0, vy: 0, theta: c.theta, roll: 0, pitch: 0, init: true,
      }));
    }
    const corr = 1 - Math.exp(-7 * delta); // position/heading correction toward truth
    const tilt = 1 - Math.exp(-6 * delta); // body roll/pitch smoothing

    for (let i = 0; i < cars.length; i++) {
      const o = outer.current[i];
      const inn = inner.current[i];
      const r = state.current[i];
      const t = cars[i];
      if (!o || !r) continue;

      const tvx = t.v * Math.cos(t.theta);
      const tvy = t.v * Math.sin(t.theta);
      const tz = t.z ?? 0;

      if (r.init || Math.hypot(t.x - r.x, t.y - r.y) > track.halfWidth * 5) {
        // first frame or a respawn teleport: snap
        r.x = t.x; r.y = t.y; r.z = tz; r.vx = tvx; r.vy = tvy; r.theta = t.theta; r.init = false;
      } else {
        // predict forward by velocity, then gently correct toward authoritative state
        r.x += r.vx * delta;
        r.y += r.vy * delta;
        r.x += (t.x - r.x) * corr;
        r.y += (t.y - r.y) * corr;
        r.z += (tz - r.z) * corr;
        r.vx += (tvx - r.vx) * corr;
        r.vy += (tvy - r.vy) * corr;
        r.theta = lerpAngle(r.theta, t.theta, corr);
      }

      o.position.set(r.x, r.z + CAR_LIFT, r.y);
      o.rotation.y = -r.theta;

      // tilt the body to the track slope (grade) + driving forces
      const targetPitch = Math.atan(t.grade ?? 0) + ((t.accel ?? 0) - (t.brake ?? 0)) * 0.04;
      const targetRoll = -(t.steer ?? 0) * 0.16;
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
            <CarMesh color={carColor(i, numCars)} selected={i === selectedId} />
          </group>
        </group>
      ))}
    </>
  );
}

// ------------------------------------------------- instanced cars (big fleets)
// For hundreds–thousands of cars, one InstancedMesh (2 draw calls) replaces the
// per-car detailed groups. Same velocity-extrapolation smoothing as Cars.
interface ISt { x: number; y: number; z: number; vx: number; vy: number; theta: number; init: boolean }

function InstancedCars({ track, telemetryRef, selectedId, onSelect, numCars = 20 }: Omit<Props, "chase">) {
  const bodyRef = useRef<THREE.InstancedMesh>(null);
  const state = useRef<ISt[]>([]);
  const off = useRef<Uint8Array>(new Uint8Array(0));
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const grey = useMemo(() => new THREE.Color("#5b606b"), []);

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
    if (!tele || !bm) return;
    const cars = tele.cars;
    const delta = Math.min(rawDelta, 0.05);
    if (state.current.length !== cars.length) {
      state.current = cars.map((c) => ({ x: c.x, y: c.y, z: c.z ?? 0, vx: 0, vy: 0, theta: c.theta, init: true }));
    }
    const corr = 1 - Math.exp(-7 * delta);
    const n = Math.min(numCars, cars.length);
    let colorDirty = false;
    for (let i = 0; i < n; i++) {
      const r = state.current[i];
      const t = cars[i];
      if (!r) continue;
      const tvx = t.v * Math.cos(t.theta);
      const tvy = t.v * Math.sin(t.theta);
      const tz = t.z ?? 0;
      if (r.init || Math.hypot(t.x - r.x, t.y - r.y) > track.halfWidth * 5) {
        r.x = t.x; r.y = t.y; r.z = tz; r.vx = tvx; r.vy = tvy; r.theta = t.theta; r.init = false;
      } else {
        r.x += r.vx * delta; r.y += r.vy * delta;
        r.x += (t.x - r.x) * corr; r.y += (t.y - r.y) * corr; r.z += (tz - r.z) * corr;
        r.vx += (tvx - r.vx) * corr; r.vy += (tvy - r.vy) * corr;
        r.theta = lerpAngle(r.theta, t.theta, corr);
      }
      dummy.position.set(r.x, r.z + CAR_LIFT, r.y);
      dummy.rotation.set(0, -r.theta, 0);
      dummy.updateMatrix();
      bm.setMatrixAt(i, dummy.matrix);
      const o = t.offtrack ? 1 : 0;
      if (off.current[i] !== o) {
        off.current[i] = o;
        bm.setColorAt(i, o ? grey : carColor(i, numCars));
        colorDirty = true;
      }
    }
    bm.count = n;
    bm.instanceMatrix.needsUpdate = true;
    if (colorDirty && bm.instanceColor) bm.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={bodyRef}
      args={[undefined, undefined, numCars]}
      castShadow
      receiveShadow
      onClick={(e) => {
        e.stopPropagation();
        if (e.instanceId != null) onSelect(e.instanceId === selectedId ? null : e.instanceId);
      }}
    >
      <boxGeometry args={[6.8, 1.2, 2.9]} />
      <meshStandardMaterial metalness={0.35} roughness={0.5} />
    </instancedMesh>
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
  const tmp = useRef(new THREE.Vector3());
  useFrame(() => {
    if (followId === null) return;
    const c = telemetryRef.current?.cars[followId];
    if (!c) return;
    const fx = Math.cos(c.theta);
    const fz = Math.sin(c.theta);
    const h = c.z ?? 0;
    tmp.current.set(c.x - fx * 42, h + 22, c.y - fz * 42);
    camera.position.lerp(tmp.current, 0.09);
    camera.lookAt(c.x, h + 3, c.y);
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
    const cz = c.z ?? 0;
    const h0 = cz + 1.4;
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
      positions[j * 6 + 4] = cz + 1.0;
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
export function CarScene3D({ track, telemetryRef, selectedId, onSelect, chase, numCars = 20 }: Props) {
  const { minX, maxX, minY, maxY } = track.bounds;
  const cx = (minX + maxX) / 2;
  const cz = (minY + maxY) / 2;
  const extent = Math.max(maxX - minX, maxY - minY);
  const dist = extent * 0.8;
  const followId = chase ? selectedId : null;
  const shadow = extent * 0.62;
  const sun: [number, number, number] = [cx - extent * 0.3, extent * 0.55, cz - extent * 0.22];

  useEffect(() => {
    const fire = () => window.dispatchEvent(new Event("resize"));
    const raf = requestAnimationFrame(fire);
    const t = setTimeout(fire, 250);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(t);
    };
  }, []);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border bg-[#0a0e16]">
      <Canvas
        shadows
        dpr={[1, 1.75]}
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
        {numCars > 60 ? (
          <InstancedCars track={track} telemetryRef={telemetryRef} selectedId={selectedId} onSelect={onSelect} numCars={numCars} />
        ) : (
          <Cars track={track} telemetryRef={telemetryRef} selectedId={selectedId} onSelect={onSelect} numCars={numCars} />
        )}
        <SensorRays track={track} telemetryRef={telemetryRef} selectedId={selectedId} />

        <CameraRig telemetryRef={telemetryRef} followId={followId} />
        <OrbitControls
          enabled={followId === null}
          target={[cx, 0, cz]}
          maxPolarAngle={Math.PI / 2.15}
          minDistance={20}
          maxDistance={extent * 2.5}
          enableDamping
          dampingFactor={0.08}
        />
      </Canvas>
    </div>
  );
}
