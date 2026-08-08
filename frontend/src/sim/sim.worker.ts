/// <reference lib="webworker" />
/**
 * In-browser simulation worker. Runs the ported environment + Transformer policy
 * off the main thread, stepping the fleet and posting telemetry frames in the
 * same shape the SSE stream used — so the 3D/2D renderers work unchanged.
 */
import type { Car, Telemetry } from "@/lib/api";
import { CarEnv } from "./env";
import { Policy, clipAction, type PolicyFile } from "./policy";
import { GpuPolicy } from "./gpuPolicy";
import { Trainer, Rollout, DEFAULT_HPARAMS } from "./trainer";
import type { WorkerIn, WorkerOut, SimTrack, PlaybackMetrics, TrainMetrics } from "./types";

let policy: Policy | null = null;
let gpu: GpuPolicy | null = null;
let useGpu = false;
let env: CarEnv | null = null;
let track: SimTrack | null = null;
let window_: Float32Array | null = null; // [N*K*O]
let actions: Float32Array | null = null; // [N*3]
let K = 8, O = 10, N = 20;
let stepsPerSec = 30;
let running = false;
let timer: ReturnType<typeof setTimeout> | null = null;
let stepCount = 0;

// per-car "driver style" so cars aren't identical copies of the shared policy:
// a persistent throttle scale + a smoothed steering wander give each car its own
// lines and corner speeds (purely a playback flavour, not independent learning).
let accelScale: Float32Array | null = null;
let steerWander: Float32Array | null = null;

// human-driven car (advanced "drive it yourself" mode); -1 = none
let playerIdx = -1;
const playerAction = new Float32Array(3);
// untrained mode: a random policy — what the network looks like before it learns
let untrained = false;

// -------- live "learn from scratch" mode (real in-browser RL, see trainer.ts) --------
let learning = false;
let seed0 = 1;
let trainer: Trainer | null = null;
let rollout: Rollout | null = null;
let brainSeed = 12345; // rotates on each "reset brain" so successive from-scratch runs differ (yet stay reproducible)
const HORIZON = 32;       // PPO rollout length before each update
const LEARN_SUBSTEPS = 3; // env+train steps per rendered frame (gentle time-lapse so learning is brisk)
let lrObs: Float32Array | null = null; // current obs [N*O]
let lrRawAct: Float32Array | null = null;
let lrEnvAct: Float32Array | null = null;
let lrLogp: Float32Array | null = null;
let lrVal: Float32Array | null = null;
let lrRew: Float32Array | null = null;
let lrLastVal: Float32Array | null = null;
let epRet: Float64Array | null = null;
let epLen: Int32Array | null = null;
let recentRet: number[] = [];
let trainHistory: number[] = []; // recent avgReturn samples for the UI sparkline
let updates = 0;
let envSteps = 0;
let bestReturn = -Infinity;
let lastEntropy = 0;
let lastTrainPost = 0;

// fps measurement (rolling)
let fpsWindow: number[] = [];
let lastMetrics = 0;

const post = (m: WorkerOut) => (self as unknown as Worker).postMessage(m);

function buildWindow() {
  window_ = new Float32Array(N * K * O);
  const obs = env!.reset();
  for (let i = 0; i < N; i++) {
    // fresh episode: history zeroed, newest token = spawn obs
    for (let d = 0; d < O; d++) window_[i * K * O + (K - 1) * O + d] = obs[i * O + d];
  }
  actions = new Float32Array(N * 3);
  accelScale = new Float32Array(N);
  steerWander = new Float32Array(N);
  for (let i = 0; i < N; i++) accelScale[i] = 0.88 + Math.random() * 0.24; // persistent per-car throttle bias
  stepCount = 0;
}

function device(): string {
  return useGpu ? "Browser · GPU" : "Browser · CPU";
}

function metrics(): PlaybackMetrics {
  const span = fpsWindow.length > 1 ? fpsWindow[fpsWindow.length - 1] - fpsWindow[0] : 0;
  const sps = span > 0 ? ((fpsWindow.length - 1) / span) * 1000 : 0;
  return {
    status: running ? "running" : "paused",
    numEnvs: N,
    stepsPerSec: sps,
    totalLaps: env!.totalLaps,
    modelStep: policy!.globalStep,
    bestReturn: 0, // filled from policy meta at init via 'ready'
    device: device(),
    track: track!.track,
  };
}

function frame(): Telemetry {
  const e = env!;
  const cars: Car[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const sensors: number[] = new Array(e.numRays);
    for (let r = 0; r < e.numRays; r++) sensors[r] = e.sensors[i * e.numRays + r];
    cars[i] = {
      id: i,
      x: e.x[i],
      y: e.y[i],
      z: e.zOf(i),
      grade: e.gradeOf(i),
      theta: e.theta[i],
      v: e.v[i],
      offtrack: e.offtrack[i] === 1,
      steer: e.lastAction[i * 3],
      accel: e.lastAction[i * 3 + 1],
      brake: e.lastAction[i * 3 + 2],
      reward: e.lastReward[i],
      sensors,
    };
  }
  return { cars, step: stepCount, status: running ? "running" : "paused" };
}

// ------------------------------------------------- live "learn from scratch"
const RING = 200;
const meanRing = (arr: number[]) => (arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : NaN);

/** (re)build the training env + buffers. resetWeights=true wipes the brain
 *  (start learning from zero); false keeps it (e.g. when only the fleet resizes). */
function buildLearn(resetWeights: boolean) {
  const A = 3;
  env = new CarEnv(track!, policy!.physics as never, N, seed0 + 1, false); // effects=false → real RL episodes (crash/timeout → reset)
  if (!trainer) trainer = new Trainer(O, A, brainSeed);
  else if (resetWeights) { brainSeed = (Math.imul(brainSeed, 1664525) + 1013904223) >>> 0; trainer.reset(brainSeed); }
  rollout = new Rollout(HORIZON, N, O, A);
  lrObs = new Float32Array(N * O); lrObs.set(env.reset());
  lrRawAct = new Float32Array(N * A);
  lrEnvAct = new Float32Array(N * A);
  lrLogp = new Float32Array(N);
  lrVal = new Float32Array(N);
  lrRew = new Float32Array(N);
  lrLastVal = new Float32Array(N);
  epRet = new Float64Array(N);
  epLen = new Int32Array(N);
  recentRet = [];
  trainHistory = [];
  updates = 0;
  envSteps = 0;
  bestReturn = -Infinity;
  lastEntropy = 0;
  stepCount = 0;
  playerIdx = -1; // learning is its own mode: no human car / scripted assists
}

/** leave learning and restore the trained-model playground (effects env) */
function stopLearning() {
  learning = false;
  env = new CarEnv(track!, policy!.physics as never, N, seed0, true);
  buildWindow();
}

function trainMetrics(): TrainMetrics {
  const avg = meanRing(recentRet);
  return {
    learning,
    updates,
    envSteps,
    avgReturn: Number.isFinite(avg) ? avg : 0,
    bestReturn: Number.isFinite(bestReturn) ? bestReturn : 0,
    entropy: lastEntropy,
    laps: env ? env.totalLaps : 0,
    history: trainHistory.slice(-60),
  };
}

function doLearnStep() {
  const e = env!, tr = trainer!, ro = rollout!;
  const A = 3;
  const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
  for (let sub = 0; sub < LEARN_SUBSTEPS; sub++) {
    tr.act(lrObs!, N, lrRawAct!, lrLogp!, lrVal!);
    for (let i = 0; i < N; i++) {
      lrEnvAct![i * A] = clamp(lrRawAct![i * A], -1, 1);
      lrEnvAct![i * A + 1] = clamp(lrRawAct![i * A + 1], 0, 1);
      lrEnvAct![i * A + 2] = clamp(lrRawAct![i * A + 2], 0, 1);
    }
    const next = e.step(lrEnvAct!);
    for (let i = 0; i < N; i++) lrRew![i] = e.lastReward[i];
    ro.add(lrObs!, lrRawAct!, lrLogp!, lrVal!, lrRew!, e.done);
    for (let i = 0; i < N; i++) {
      epRet![i] += lrRew![i];
      epLen![i]++;
      if (e.done[i]) {
        recentRet.push(epRet![i]);
        if (recentRet.length > RING) recentRet.shift();
        epRet![i] = 0; epLen![i] = 0;
      }
    }
    lrObs!.set(next);
    envSteps++;
    if (ro.full) {
      for (let i = 0; i < N; i++) lrLastVal![i] = tr.value(lrObs!, i);
      const b = ro.finish(lrLastVal!, DEFAULT_HPARAMS.gamma, DEFAULT_HPARAMS.lambda);
      const stats = tr.update(b.obs, b.act, b.logp, b.adv, b.ret, b.S);
      updates++;
      lastEntropy = stats.entropy;
      const avg = meanRing(recentRet);
      if (Number.isFinite(avg)) {
        if (avg > bestReturn) bestReturn = avg;
        trainHistory.push(avg);
        if (trainHistory.length > 120) trainHistory.shift();
      }
    }
  }
  stepCount++;
  post({ type: "frame", telemetry: frame() });
  const now = performance.now();
  fpsWindow.push(now);
  if (fpsWindow.length > 30) fpsWindow.shift();
  if (now - lastMetrics > 250) { lastMetrics = now; post({ type: "metrics", metrics: metrics() }); }
  if (now - lastTrainPost > 200) { lastTrainPost = now; post({ type: "train", metrics: trainMetrics() }); }
}

async function doStep() {
  if (learning) { doLearnStep(); return; }
  const e = env!;
  const w = window_!;
  const a = actions!;
  const KO = K * O;

  // 1) actions for every car. The window buffer is already in the [N*K*O] batch
  //    layout the GPU shader wants, so pass it straight through.
  if (untrained) {
    // an unlearned (random) policy — exploratory flailing, the "before training" look
    for (let i = 0; i < N; i++) {
      if (steerWander) {
        steerWander[i] = steerWander[i] * 0.93 + (Math.random() - 0.5) * 0.28;
        a[i * 3] = Math.max(-1, Math.min(1, steerWander[i]));
      } else a[i * 3] = Math.random() * 2 - 1;
      a[i * 3 + 1] = 0.35 + Math.random() * 0.5;
      a[i * 3 + 2] = Math.random() < 0.06 ? Math.random() * 0.5 : 0;
    }
  } else if (useGpu && gpu) {
    const means = await gpu.forwardBatch(w, N);
    // a config/reset message can rebuild env/window/actions during the await;
    // if it did, this result is stale — drop the frame rather than index OOB.
    if (env !== e || actions !== a || window_ !== w) return;
    for (let i = 0; i < N; i++) {
      a[i * 3] = Math.max(-1, Math.min(1, means[i * 3]));
      a[i * 3 + 1] = Math.max(0, Math.min(1, means[i * 3 + 1]));
      a[i * 3 + 2] = Math.max(0, Math.min(1, means[i * 3 + 2]));
    }
  } else {
    const p = policy!;
    for (let i = 0; i < N; i++) {
      const mean = p.forward(w.subarray(i * KO, (i + 1) * KO));
      const [s, ac, br] = clipAction(mean);
      a[i * 3] = s;
      a[i * 3 + 1] = ac;
      a[i * 3 + 2] = br;
    }
  }

  // 1b) apply per-car driver style (varied lines + corner speeds)
  if (!untrained && accelScale && steerWander) {
    for (let i = 0; i < N; i++) {
      steerWander[i] = steerWander[i] * 0.9 + (Math.random() - 0.5) * 0.05;
      a[i * 3] = Math.max(-1, Math.min(1, a[i * 3] + steerWander[i]));
      a[i * 3 + 1] = Math.max(0, Math.min(1, a[i * 3 + 1] * accelScale[i]));
    }
  }

  // 1c) traffic assist: ease off when closing on a car ahead. The policy can't
  //     see other cars, so without this they drive straight through each other;
  //     this makes a trailing car follow instead of ramming.
  if (!untrained) {
    const LOOK = 26;
    const LAT = 5.5;
    for (let i = 0; i < N; i++) {
      const hx = Math.cos(e.theta[i]);
      const hy = Math.sin(e.theta[i]);
      let nearest = Infinity;
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        const dx = e.x[j] - e.x[i];
        const dy = e.y[j] - e.y[i];
        const fwd = dx * hx + dy * hy; // distance ahead
        if (fwd <= 0 || fwd > LOOK) continue;
        if (Math.abs(-dx * hy + dy * hx) > LAT) continue; // lateral gate
        if (fwd < nearest) nearest = fwd;
      }
      if (nearest < LOOK) {
        const t = 1 - nearest / LOOK; // 0 far → 1 nose-to-tail
        a[i * 3 + 1] *= 1 - 0.9 * t;
        a[i * 3 + 2] = Math.max(a[i * 3 + 2], t * 0.8);
      }
    }
  }

  // 1d) human-driven car: the player's input replaces the policy for this car
  if (playerIdx >= 0 && playerIdx < N) {
    a[playerIdx * 3] = Math.max(-1, Math.min(1, playerAction[0]));
    a[playerIdx * 3 + 1] = Math.max(0, Math.min(1, playerAction[1]));
    a[playerIdx * 3 + 2] = Math.max(0, Math.min(1, playerAction[2]));
  }

  // 2) advance the world
  const obs = e.step(a);
  stepCount++;

  // 3) slide each car's observation window (zero history for cars that reset)
  for (let i = 0; i < N; i++) {
    const base = i * KO;
    w.copyWithin(base, base + O, base + KO); // shift tokens left
    for (let d = 0; d < O; d++) w[base + (K - 1) * O + d] = obs[i * O + d];
    if (e.done[i]) {
      for (let k = 0; k < K - 1; k++) for (let d = 0; d < O; d++) w[base + k * O + d] = 0;
    }
  }

  // 4) emit
  post({ type: "frame", telemetry: frame() });
  const now = performance.now();
  fpsWindow.push(now);
  if (fpsWindow.length > 30) fpsWindow.shift();
  if (now - lastMetrics > 250) {
    lastMetrics = now;
    post({ type: "metrics", metrics: metrics() });
  }
}

async function loop() {
  if (!running) return;
  const t0 = performance.now();
  try {
    await doStep();
  } catch (err) {
    // a step threw (e.g. WebGPU device loss mid-forward). Fall back to the CPU
    // policy and keep going rather than silently freezing the playground.
    if (useGpu) {
      useGpu = false; gpu = null;
      console.warn("[sim] step failed — falling back to CPU:", err);
    } else {
      post({ type: "error", message: String(err instanceof Error ? err.message : err) });
      stopLoop();
      return;
    }
  }
  const dur = performance.now() - t0;
  // pace to the target rate, but never wait longer than needed — heavy fleets
  // that already exceed the interval just run flat-out (as fast as the hardware allows)
  if (running) timer = setTimeout(loop, Math.max(0, 1000 / stepsPerSec - dur));
}

function stopLoop() {
  running = false;
  if (timer) clearTimeout(timer);
  timer = null;
}

self.onmessage = async (ev: MessageEvent<WorkerIn>) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      track = msg.trackData;
      N = msg.numEnvs;
      seed0 = msg.seed;
      stepsPerSec = msg.stepsPerSec;
      const pf = (await fetch(msg.policyUrl).then((r) => r.json())) as PolicyFile;
      policy = new Policy(pf);
      K = policy.arch.window;
      O = policy.arch.obsDim;
      env = new CarEnv(track, policy.physics as never, N, msg.seed, true); // effects: collisions + wall stop-and-recover
      buildWindow();
      // prefer the GPU (batched policy) when WebGPU is available in the worker
      try {
        gpu = await GpuPolicy.create(pf);
        useGpu = true;
      } catch (err) {
        useGpu = false;
        gpu = null;
        console.warn("[sim] WebGPU unavailable in worker, using CPU:", err);
      }
      post({ type: "ready", device: device(), modelStep: policy.globalStep, bestReturn: (pf as { bestReturn?: number }).bestReturn ?? 0 });
      running = true;
      loop();
    } else if (msg.type === "config") {
      if (msg.stepsPerSec != null) stepsPerSec = msg.stepsPerSec;
      if (msg.numEnvs != null && msg.numEnvs !== N && env && policy && track) {
        N = msg.numEnvs;
        if (learning) buildLearn(false); // resize the fleet, keep the brain it's learned so far
        else { env = new CarEnv(track, policy.physics as never, N, 1, true); buildWindow(); }
      }
    } else if (msg.type === "reset") {
      if (learning && env) {
        // during learning, "reset" restarts the episodes but keeps the brain
        lrObs!.set(env.reset());
        epRet?.fill(0); epLen?.fill(0);
      } else if (env) {
        env.totalLaps = 0;
        buildWindow(); // re-spawns all cars, clears the observation window + step count
      }
    } else if (msg.type === "setLearning") {
      if (msg.value) {
        if (env && policy && track) { untrained = false; learning = true; buildLearn(true); if (!running) { running = true; loop(); } }
      } else if (learning) {
        stopLearning();
      }
    } else if (msg.type === "resetBrain") {
      if (env && policy && track) { untrained = false; learning = true; buildLearn(true); if (!running) { running = true; loop(); } }
    } else if (msg.type === "setPlayer") {
      playerIdx = msg.index;
      playerAction[0] = 0; playerAction[1] = 0; playerAction[2] = 0;
    } else if (msg.type === "playerInput") {
      playerAction[0] = msg.steer;
      playerAction[1] = msg.accel;
      playerAction[2] = msg.brake;
    } else if (msg.type === "setUntrained") {
      untrained = msg.value;
      if (learning) stopLearning(); // untrained + learning are mutually exclusive
      else if (env) buildWindow(); // fresh start so the before/after contrast is clear
    } else if (msg.type === "pause") {
      stopLoop();
    } else if (msg.type === "resume") {
      if (!running && env) {
        running = true;
        loop();
      }
    } else if (msg.type === "stop") {
      stopLoop();
      gpu?.dispose();
      gpu = null;
      policy = null;
      env = null;
      trainer = null;
      rollout = null;
      learning = false;
    }
  } catch (err) {
    post({ type: "error", message: String(err instanceof Error ? err.message : err) });
  }
};
