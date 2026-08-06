/**
 * JS↔Python parity harness. Asserts the TS policy + environment reproduce the
 * Python trace (backend/export_web.py -> parity-trace.json) within tolerance.
 *
 *   cd frontend && npx --yes tsx scripts/parity.ts
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Policy, type PolicyFile } from "../src/sim/policy.ts";
import { CarEnv, type PhysicsConfig, type TrackData } from "../src/sim/env.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, "..", "public", "web");
const read = (f: string) => JSON.parse(readFileSync(join(WEB, f), "utf8"));

const policyFile = read("policy.json") as PolicyFile;
const trace = read("parity-trace.json");
const track = read(`track-${trace.track}.json`) as TrackData;

const policy = new Policy(policyFile);
const cfg = policyFile.physics as unknown as PhysicsConfig;
const K = policyFile.arch.window;
const O = policyFile.arch.obsDim;
const N = trace.numEnvs as number;

let ok = true;
const check = (label: string, maxErr: number, tol: number) => {
  const pass = maxErr <= tol && Number.isFinite(maxErr);
  ok = ok && pass;
  console.log(`  ${pass ? "PASS" : "FAIL"}  ${label.padEnd(34)} max|err|=${maxErr.toExponential(3)}  (tol ${tol.toExponential(1)})`);
};

// ---------------------------------------------------------------- 1) policy
// For every recorded (window -> mean) pair, run the TS forward pass.
{
  let maxErr = 0;
  for (const step of trace.steps) {
    for (let i = 0; i < N; i++) {
      const win = Float32Array.from(step.obsWindow[i].flat() as number[]); // [K*O]
      const mean = policy.forward(win);
      for (let a = 0; a < 3; a++) maxErr = Math.max(maxErr, Math.abs(mean[a] - step.meanAction[i][a]));
    }
  }
  check("policy forward (action mean)", maxErr, 2e-3);
}

// ------------------------------------------------------------- 2) observe
// Load the recorded spawn state, observe, compare to the newest window row.
{
  const env = new CarEnv(track, cfg, N, trace.seed);
  const init = trace.init;
  for (let i = 0; i < N; i++)
    env.setCar(i, init.x[i], init.y[i], init.theta[i], init.v[i], init.progIdx[i], init.progCont[i]);
  const obs = env.observe();
  let maxErr = 0;
  const first = trace.steps[0];
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < O; d++) {
      const got = obs[i * O + d];
      const exp = first.obsWindow[i][K - 1][d]; // newest row = spawn observation
      maxErr = Math.max(maxErr, Math.abs(got - exp));
    }
  }
  check("observe (lidar + heading)", maxErr, 1e-4);
}

// ------------------------------------------------------- 3) physics rollout
// Drive the env with the RECORDED applied actions from the recorded init and
// compare the full trajectory (isolates physics from the policy).
{
  const env = new CarEnv(track, cfg, N, trace.seed);
  const init = trace.init;
  for (let i = 0; i < N; i++)
    env.setCar(i, init.x[i], init.y[i], init.theta[i], init.v[i], init.progIdx[i], init.progCont[i]);

  let maxPos = 0, maxVel = 0, maxTheta = 0, maxSensor = 0, offMismatch = 0;
  for (const step of trace.steps) {
    if (step.done.some((d: number) => d === 1)) break; // trace stops at first reset
    const act = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      act[i * 3] = step.appliedAction[i][0];
      act[i * 3 + 1] = step.appliedAction[i][1];
      act[i * 3 + 2] = step.appliedAction[i][2];
    }
    env.step(act);
    const post = step.post;
    for (let i = 0; i < N; i++) {
      maxPos = Math.max(maxPos, Math.abs(env.x[i] - post.x[i]), Math.abs(env.y[i] - post.y[i]));
      maxVel = Math.max(maxVel, Math.abs(env.v[i] - post.v[i]));
      maxTheta = Math.max(maxTheta, Math.abs(env.theta[i] - post.theta[i]));
      if (env.offtrack[i] !== post.offtrack[i]) offMismatch++;
      for (let r = 0; r < cfg.numRays; r++)
        maxSensor = Math.max(maxSensor, Math.abs(env.sensors[i * cfg.numRays + r] - post.sensors[i][r]));
    }
  }
  check("physics rollout: position", maxPos, 1e-3);
  check("physics rollout: velocity", maxVel, 1e-4);
  check("physics rollout: heading", maxTheta, 1e-5);
  check("physics rollout: lidar", maxSensor, 1e-4);
  check("physics rollout: offtrack flags", offMismatch, 0);
}

console.log(ok ? "\n✅ PARITY OK — TS port matches Python\n" : "\n❌ PARITY FAILED\n");
process.exit(ok ? 0 : 1);
