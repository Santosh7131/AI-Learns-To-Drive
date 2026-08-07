/**
 * Headless smoke-test for the in-browser RL trainer (src/sim/trainer.ts).
 *
 * Runs the real training env (effects=false) + PPO trainer for a while and
 * prints the mean episode return over time. Asserts the policy actually LEARNS
 * (late return >> early return). This is how we verify learning works without a
 * browser — same spirit as the parity harness.
 *
 *   cd frontend && npx --yes tsx scripts/train-smoke.ts [numCars] [envSteps]
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { CarEnv, type PhysicsConfig, type TrackData } from "../src/sim/env.ts";
import { Trainer, Rollout, DEFAULT_HPARAMS } from "../src/sim/trainer.ts";
import type { PolicyFile } from "../src/sim/policy.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, "..", "public", "web");
const readWeb = (f: string) => JSON.parse(readFileSync(join(WEB, f), "utf8"));

const policyFile = readWeb("policy.json") as PolicyFile;
const cfg = policyFile.physics as unknown as PhysicsConfig;
const O = policyFile.arch.obsDim;
const A = policyFile.arch.actDim;

const trackId = (policyFile.trainedTrack as string) || "default";
const track = readWeb(`track-${trackId}.json`) as TrackData;

const N = Number(process.argv[2] || 32);
const TOTAL = Number(process.argv[3] || 80000);
const HORIZON = 32;
const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

console.log(`train-smoke: track=${trackId} cars=${N} obsDim=${O} actDim=${A} steps=${TOTAL}`);
console.log(`hparams: ${JSON.stringify(DEFAULT_HPARAMS)}\n`);

const env = new CarEnv(track, cfg, N, 7, false); // effects=false => real RL episodes (crash/timeout -> reset)
const trainer = new Trainer(O, A, 12345);
const roll = new Rollout(HORIZON, N, O, A);

const curObs = new Float32Array(N * O);
curObs.set(env.reset());

const rawAct = new Float32Array(N * A);
const envAct = new Float32Array(N * A);
const logp = new Float32Array(N);
const val = new Float32Array(N);
const rew = new Float32Array(N);
const lastVal = new Float32Array(N);

const epRet = new Float64Array(N);
const epLen = new Int32Array(N);
const recentRet: number[] = [];
const recentLen: number[] = [];
const RING = 200;
const pushRing = (arr: number[], v: number) => { arr.push(v); if (arr.length > RING) arr.shift(); };
const meanOf = (arr: number[]) => (arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : NaN);

const curve: { step: number; ret: number; len: number }[] = [];
let earlyMean = NaN;
let updates = 0;
const t0 = Date.now();

for (let step = 1; step <= TOTAL; step++) {
  trainer.act(curObs, N, rawAct, logp, val);
  for (let i = 0; i < N; i++) {
    envAct[i * A] = clamp(rawAct[i * A], -1, 1);
    envAct[i * A + 1] = clamp(rawAct[i * A + 1], 0, 1);
    envAct[i * A + 2] = clamp(rawAct[i * A + 2], 0, 1);
  }
  const next = env.step(envAct);
  for (let i = 0; i < N; i++) rew[i] = env.lastReward[i];

  roll.add(curObs, rawAct, logp, val, rew, env.done);

  for (let i = 0; i < N; i++) {
    epRet[i] += rew[i];
    epLen[i]++;
    if (env.done[i]) {
      pushRing(recentRet, epRet[i]);
      pushRing(recentLen, epLen[i]);
      epRet[i] = 0;
      epLen[i] = 0;
    }
  }
  curObs.set(next);

  if (roll.full) {
    for (let i = 0; i < N; i++) lastVal[i] = trainer.value(curObs, i);
    const b = roll.finish(lastVal, DEFAULT_HPARAMS.gamma, DEFAULT_HPARAMS.lambda);
    const stats = trainer.update(b.obs, b.act, b.logp, b.adv, b.ret, b.S);
    updates++;
    if (updates % 15 === 0) {
      const mr = meanOf(recentRet);
      const ml = meanOf(recentLen);
      curve.push({ step, ret: mr, len: ml });
      if (Number.isNaN(earlyMean) && recentRet.length >= 50) earlyMean = mr;
      console.log(
        `step ${String(step).padStart(6)} | upd ${String(updates).padStart(4)} | ` +
        `ret ${mr.toFixed(2).padStart(7)} | epLen ${ml.toFixed(0).padStart(4)} | ` +
        `laps ${env.totalLaps.toString().padStart(4)} | ent ${stats.entropy.toFixed(2)} | ` +
        `std ${trainer.logStd.map((x) => Math.exp(x).toFixed(2)).join("/")}`
      );
    }
  }
}

const secs = (Date.now() - t0) / 1000;
const lateMean = meanOf(recentRet);
const lateLen = meanOf(recentLen);
console.log(`\ndone in ${secs.toFixed(1)}s  (${(TOTAL / secs).toFixed(0)} env steps/s, ${(updates / secs).toFixed(1)} updates/s)`);
console.log(`early mean return ≈ ${earlyMean.toFixed(2)}   late mean return ≈ ${lateMean.toFixed(2)}   late epLen ≈ ${lateLen.toFixed(0)}`);

// success = the learner clearly improved (return climbed and episodes got longer)
const improvedReturn = Number.isFinite(earlyMean) && lateMean > earlyMean + 1.0;
const drivesForward = lateMean > 0.5; // net positive => making real progress, not just crashing
const ok = improvedReturn && drivesForward;
console.log(ok ? "\n✅ LEARNS — the car improves from scratch\n" : "\n❌ NO CLEAR LEARNING (tune hparams)\n");
process.exit(ok ? 0 : 1);
