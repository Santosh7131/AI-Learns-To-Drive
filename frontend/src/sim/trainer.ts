/**
 * In-browser reinforcement learning — a compact actor-critic trained live by
 * PPO, so a visitor can wipe the brain and WATCH a car learn to drive from
 * scratch (crashes -> clean laps) in the page.
 *
 * This is deliberately NOT the deployed Transformer (see policy.ts): training
 * needs a full backward pass, and hand-writing backprop through multi-head
 * attention on the CPU would be far too slow to learn on-screen. Instead this
 * is a small MLP over the current 10-dim observation — Markov enough for
 * reactive lane-following, and light enough to train hundreds of updates per
 * minute in a Web Worker. The shipped Transformer + JS<->Python parity are
 * untouched by anything in this file.
 *
 * Algorithm: synchronous PPO across the parallel fleet. Diagonal-Gaussian
 * policy with a state-independent, learnable log-std; GAE(lambda) advantages;
 * clipped surrogate objective; value + entropy terms; Adam.
 */

const LOG_2PI = Math.log(2 * Math.PI);

/** seeded PRNG (mulberry32) so headless smoke-tests are reproducible */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** one fully-connected layer + its Adam moments and grad accumulator */
interface Layer {
  W: Float32Array; // [nout * nin], row-major
  b: Float32Array; // [nout]
  gW: Float32Array;
  gb: Float32Array;
  mW: Float32Array;
  vW: Float32Array;
  mb: Float32Array;
  vb: Float32Array;
  nin: number;
  nout: number;
}

/** MLP with tanh hidden activations and a linear output layer. */
class MLP {
  layers: Layer[] = [];
  private sizes: number[];
  private rng: () => number;

  constructor(sizes: number[], rng: () => number, outScale = 0.01) {
    this.sizes = sizes;
    this.rng = rng;
    this.init(outScale);
  }

  init(outScale: number) {
    this.layers = [];
    for (let l = 0; l < this.sizes.length - 1; l++) {
      const nin = this.sizes[l];
      const nout = this.sizes[l + 1];
      const isOut = l === this.sizes.length - 2;
      // Xavier-ish for tanh hidden layers; tiny output layer so the initial
      // policy is gentle (near-zero action mean) and the value starts flat.
      const std = isOut ? outScale : Math.sqrt(1 / nin);
      const W = new Float32Array(nout * nin);
      for (let i = 0; i < W.length; i++) W[i] = this.randn() * std;
      this.layers.push({
        W,
        b: new Float32Array(nout),
        gW: new Float32Array(nout * nin),
        gb: new Float32Array(nout),
        mW: new Float32Array(nout * nin),
        vW: new Float32Array(nout * nin),
        mb: new Float32Array(nout),
        vb: new Float32Array(nout),
        nin,
        nout,
      });
    }
  }

  private randn(): number {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = this.rng();
    while (v === 0) v = this.rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** forward one sample; writes each layer's post-activation into `acts` (caller
   *  pre-allocates one Float32Array per layer). Returns the output array. */
  forward(x: Float32Array, acts: Float32Array[]): Float32Array {
    let inp = x;
    for (let l = 0; l < this.layers.length; l++) {
      const { W, b, nin, nout } = this.layers[l];
      const out = acts[l];
      const isOut = l === this.layers.length - 1;
      for (let o = 0; o < nout; o++) {
        let acc = b[o];
        const base = o * nin;
        for (let i = 0; i < nin; i++) acc += inp[i] * W[base + i];
        out[o] = isOut ? acc : Math.tanh(acc);
      }
      inp = out;
    }
    return acts[this.layers.length - 1];
  }

  /** backprop one sample, accumulating grads. `dOut` is d(loss)/d(output).
   *  `x` and `acts` must be the exact forward inputs/activations for this sample.
   *  `scratch` holds two [maxWidth] buffers reused for dA/dZ. */
  backward(x: Float32Array, acts: Float32Array[], dOut: Float32Array, scratch: [Float32Array, Float32Array]) {
    let dA = scratch[0];
    let dNext = scratch[1];
    const L = this.layers.length;
    for (let o = 0; o < this.layers[L - 1].nout; o++) dA[o] = dOut[o];
    for (let l = L - 1; l >= 0; l--) {
      const { W, gW, gb, nin, nout } = this.layers[l];
      const a = acts[l];
      const inp = l === 0 ? x : acts[l - 1];
      const isOut = l === L - 1;
      // dZ = dA * activation'(z);  hidden: tanh' = 1 - a^2 ; output: linear
      for (let o = 0; o < nout; o++) {
        const dz = isOut ? dA[o] : dA[o] * (1 - a[o] * a[o]);
        gb[o] += dz;
        const base = o * nin;
        for (let i = 0; i < nin; i++) gW[base + i] += dz * inp[i];
        dA[o] = dz; // store dZ in place for the input-grad pass below
      }
      if (l > 0) {
        // dInput = W^T dZ
        for (let i = 0; i < nin; i++) {
          let acc = 0;
          for (let o = 0; o < nout; o++) acc += this.layers[l].W[o * nin + i] * dA[o];
          dNext[i] = acc;
        }
        const tmp = dA; dA = dNext; dNext = tmp; // swap for next (shallower) layer
      }
      void W;
    }
  }

  zeroGrad() {
    for (const L of this.layers) { L.gW.fill(0); L.gb.fill(0); }
  }

  /** Adam update. `scale` divides the accumulated grads (= 1/minibatchSize). */
  adam(lr: number, t: number, scale: number, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    for (const L of this.layers) {
      adamArr(L.W, L.gW, L.mW, L.vW, lr, scale, b1, b2, eps, bc1, bc2);
      adamArr(L.b, L.gb, L.mb, L.vb, lr, scale, b1, b2, eps, bc1, bc2);
    }
  }
}

function adamArr(
  p: Float32Array, g: Float32Array, m: Float32Array, v: Float32Array,
  lr: number, scale: number, b1: number, b2: number, eps: number, bc1: number, bc2: number,
) {
  for (let i = 0; i < p.length; i++) {
    const grad = g[i] * scale;
    m[i] = b1 * m[i] + (1 - b1) * grad;
    v[i] = b2 * v[i] + (1 - b2) * grad * grad;
    const mhat = m[i] / bc1;
    const vhat = v[i] / bc2;
    p[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
  }
}

export interface TrainHParams {
  gamma: number;
  lambda: number;
  clip: number;
  lr: number;
  valueCoef: number;
  entCoef: number;
  epochs: number;
  numMinibatch: number;
  hidden: number;
}

export const DEFAULT_HPARAMS: TrainHParams = {
  gamma: 0.99,
  lambda: 0.95,
  clip: 0.2,
  lr: 3e-3,
  valueCoef: 0.5,
  entCoef: 0.004, // modest exploration bonus — high enough to explore early, low enough not to make the converged driver wobble
  epochs: 4,
  numMinibatch: 4,
  hidden: 64,
};

const LOGSTD_MIN = Math.log(0.05);
const LOGSTD_MAX = Math.log(0.8); // cap exploration noise so a learned car drives cleanly, not jittery

export interface UpdateStats {
  entropy: number;
  approxKL: number;
  valueLoss: number;
}

/**
 * PPO trainer over a diagonal-Gaussian policy. `act()` samples actions for the
 * fleet during rollout; `update()` runs the clipped PPO step on a collected
 * batch. Steer is [-1,1]; accel/brake are [0,1] (clipped by the caller when
 * applied to the env — log-probs are computed on the raw Gaussian sample).
 */
export class Trainer {
  readonly obsDim: number;
  readonly actDim: number;
  private actor: MLP;
  private critic: MLP;
  logStd: Float32Array;
  private gLogStd: Float32Array;
  private mLogStd: Float32Array;
  private vLogStd: Float32Array;
  private hp: TrainHParams;
  private rng: () => number;
  private adamT = 0;

  // reusable per-sample activation buffers
  private aActs: Float32Array[];
  private cActs: Float32Array[];
  private scratch: [Float32Array, Float32Array];
  private dMean: Float32Array;
  private dVal: Float32Array;

  constructor(obsDim: number, actDim: number, seed = 12345, hp: Partial<TrainHParams> = {}) {
    this.obsDim = obsDim;
    this.actDim = actDim;
    this.hp = { ...DEFAULT_HPARAMS, ...hp };
    this.rng = mulberry32(seed);
    const H = this.hp.hidden;
    this.actor = new MLP([obsDim, H, H, actDim], this.rng, 0.01);
    this.critic = new MLP([obsDim, H, H, 1], this.rng, 1.0);
    this.logStd = new Float32Array(actDim).fill(Math.log(0.6));
    this.gLogStd = new Float32Array(actDim);
    this.mLogStd = new Float32Array(actDim);
    this.vLogStd = new Float32Array(actDim);

    const mk = (net: MLP) => net.layers.map((l) => new Float32Array(l.nout));
    this.aActs = mk(this.actor);
    this.cActs = mk(this.critic);
    const maxW = Math.max(H, obsDim, actDim);
    this.scratch = [new Float32Array(maxW), new Float32Array(maxW)];
    this.dMean = new Float32Array(actDim);
    this.dVal = new Float32Array(1);
  }

  /** wipe the brain: re-initialize all weights + exploration. This is the
   *  "reset the brain to zero" the user asked for — learning restarts. */
  reset(seed?: number) {
    if (seed != null) this.rng = mulberry32(seed);
    this.actor.init(0.01);
    this.critic.init(1.0);
    this.logStd.fill(Math.log(0.6));
    this.mLogStd.fill(0); this.vLogStd.fill(0);
    this.adamT = 0;
  }

  private randn(): number {
    let u = 0, v = 0;
    while (u === 0) u = this.rng();
    while (v === 0) v = this.rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /**
   * Sample actions for the whole fleet.
   * @param obs   flat [n*obsDim]
   * @param n     number of cars
   * @param outAct   flat [n*actDim] raw sampled actions (unclipped) — filled
   * @param outLogp  [n] log-prob of each sampled action — filled
   * @param outVal   [n] critic value of each obs — filled
   */
  act(obs: Float32Array, n: number, outAct: Float32Array, outLogp: Float32Array, outVal: Float32Array) {
    const A = this.actDim;
    for (let i = 0; i < n; i++) {
      const x = obs.subarray(i * this.obsDim, (i + 1) * this.obsDim);
      const mean = this.actor.forward(x, this.aActs);
      const val = this.critic.forward(x, this.cActs);
      let logp = 0;
      for (let d = 0; d < A; d++) {
        const std = Math.exp(this.logStd[d]);
        const noise = this.randn();
        const a = mean[d] + std * noise;
        outAct[i * A + d] = a;
        // logN(a; mean, std) = -0.5*z^2 - logStd - 0.5*log(2pi), z = noise
        logp += -0.5 * noise * noise - this.logStd[d] - 0.5 * LOG_2PI;
      }
      outLogp[i] = logp;
      outVal[i] = val[0];
    }
  }

  /** greedy (deterministic) action means for the fleet — used for a clean
   *  "show me the learned driver" view without exploration noise. */
  actMean(obs: Float32Array, n: number, outAct: Float32Array) {
    const A = this.actDim;
    for (let i = 0; i < n; i++) {
      const x = obs.subarray(i * this.obsDim, (i + 1) * this.obsDim);
      const mean = this.actor.forward(x, this.aActs);
      for (let d = 0; d < A; d++) outAct[i * A + d] = mean[d];
    }
  }

  value(obs: Float32Array, i: number): number {
    const x = obs.subarray(i * this.obsDim, (i + 1) * this.obsDim);
    return this.critic.forward(x, this.cActs)[0];
  }

  /**
   * One PPO update on a collected batch.
   * @param obs   [S*obsDim]
   * @param act   [S*actDim] raw actions taken
   * @param logpOld [S]
   * @param adv   [S] advantages (will be normalized here)
   * @param ret   [S] value targets (GAE returns)
   * @param S     number of samples
   */
  update(obs: Float32Array, act: Float32Array, logpOld: Float32Array, adv: Float32Array, ret: Float32Array, S: number): UpdateStats {
    const A = this.actDim;
    const { clip, lr, valueCoef, entCoef, epochs, numMinibatch } = this.hp;

    // normalize advantages
    let mean = 0;
    for (let s = 0; s < S; s++) mean += adv[s];
    mean /= S;
    let varr = 0;
    for (let s = 0; s < S; s++) { const d = adv[s] - mean; varr += d * d; }
    const std = Math.sqrt(varr / S) + 1e-8;
    const advN = new Float32Array(S);
    for (let s = 0; s < S; s++) advN[s] = (adv[s] - mean) / std;

    const idx = new Int32Array(S);
    for (let s = 0; s < S; s++) idx[s] = s;

    const mbSize = Math.max(1, Math.floor(S / numMinibatch));
    let entAccum = 0, klAccum = 0, vlAccum = 0, updates = 0;

    for (let ep = 0; ep < epochs; ep++) {
      // Fisher-Yates shuffle
      for (let s = S - 1; s > 0; s--) {
        const j = Math.floor(this.rng() * (s + 1));
        const tmp = idx[s]; idx[s] = idx[j]; idx[j] = tmp;
      }
      for (let mb = 0; mb < S; mb += mbSize) {
        const end = Math.min(S, mb + mbSize);
        const count = end - mb;
        this.actor.zeroGrad();
        this.critic.zeroGrad();
        this.gLogStd.fill(0);
        let entMB = 0, klMB = 0, vlMB = 0;

        for (let m = mb; m < end; m++) {
          const s = idx[m];
          const x = obs.subarray(s * this.obsDim, (s + 1) * this.obsDim);
          const meanOut = this.actor.forward(x, this.aActs);
          const valOut = this.critic.forward(x, this.cActs);

          // new log-prob + entropy for this sample
          let newLogp = 0, ent = 0;
          for (let d = 0; d < A; d++) {
            const ls = this.logStd[d];
            const sd = Math.exp(ls);
            const diff = act[s * A + d] - meanOut[d];
            const z = diff / sd;
            newLogp += -0.5 * z * z - ls - 0.5 * LOG_2PI;
            ent += ls + 0.5 * (LOG_2PI + 1); // 0.5*log(2*pi*e) + logStd
          }
          const ratio = Math.exp(newLogp - logpOld[s]);
          const Aadv = advN[s];
          const surr1 = ratio * Aadv;
          const clipped = Math.max(1 - clip, Math.min(1 + clip, ratio));
          const surr2 = clipped * Aadv;
          // d(policyLoss)/d(newLogp): only the unclipped branch passes gradient
          const gp = surr1 <= surr2 ? -Aadv * ratio : 0;

          // grads into the actor's mean output + the log-std parameter
          for (let d = 0; d < A; d++) {
            const ls = this.logStd[d];
            const sd = Math.exp(ls);
            const diff = act[s * A + d] - meanOut[d];
            const invVar = 1 / (sd * sd);
            // d logp / d mean = diff/var ; d logp / d logStd = (diff^2/var) - 1
            this.dMean[d] = gp * (diff * invVar);
            this.gLogStd[d] += gp * (diff * diff * invVar - 1) - entCoef; // + entropy grad
          }
          this.actor.backward(x, this.aActs, this.dMean, this.scratch);

          // value loss = valueCoef * 0.5 * (V - ret)^2
          const vErr = valOut[0] - ret[s];
          this.dVal[0] = valueCoef * vErr;
          this.critic.backward(x, this.cActs, this.dVal, this.scratch);

          entMB += ent / A;
          klMB += logpOld[s] - newLogp; // approx KL
          vlMB += vErr * vErr;
        }

        this.adamT++;
        const scale = 1 / count;
        this.actor.adam(lr, this.adamT, scale);
        this.critic.adam(lr, this.adamT, scale);
        // Adam on log-std (its own moments)
        adamArr(this.logStd, this.gLogStd, this.mLogStd, this.vLogStd, lr, scale, 0.9, 0.999, 1e-8, 1 - Math.pow(0.9, this.adamT), 1 - Math.pow(0.999, this.adamT));
        for (let d = 0; d < A; d++) this.logStd[d] = Math.max(LOGSTD_MIN, Math.min(LOGSTD_MAX, this.logStd[d]));

        entAccum += entMB / count;
        klAccum += klMB / count;
        vlAccum += vlMB / count;
        updates++;
      }
    }
    return {
      entropy: entAccum / Math.max(1, updates),
      approxKL: klAccum / Math.max(1, updates),
      valueLoss: vlAccum / Math.max(1, updates),
    };
  }
}

/**
 * Fixed-horizon rollout buffer for synchronous PPO across `n` parallel cars.
 * Stores per-step (obs, action, logp, value, reward, done); computes GAE and
 * flattens to a training batch.
 */
export class Rollout {
  readonly horizon: number;
  readonly n: number;
  readonly obsDim: number;
  readonly actDim: number;
  private obs: Float32Array;
  private act: Float32Array;
  private logp: Float32Array;
  private val: Float32Array;
  private rew: Float32Array;
  private done: Float32Array;
  ptr = 0;

  constructor(horizon: number, n: number, obsDim: number, actDim: number) {
    this.horizon = horizon;
    this.n = n;
    this.obsDim = obsDim;
    this.actDim = actDim;
    this.obs = new Float32Array(horizon * n * obsDim);
    this.act = new Float32Array(horizon * n * actDim);
    this.logp = new Float32Array(horizon * n);
    this.val = new Float32Array(horizon * n);
    this.rew = new Float32Array(horizon * n);
    this.done = new Float32Array(horizon * n);
  }

  get full(): boolean {
    return this.ptr >= this.horizon;
  }

  /** record one timestep across the fleet */
  add(obs: Float32Array, act: Float32Array, logp: Float32Array, val: Float32Array, rew: Float32Array, done: Uint8Array) {
    const t = this.ptr;
    this.obs.set(obs.subarray(0, this.n * this.obsDim), t * this.n * this.obsDim);
    this.act.set(act.subarray(0, this.n * this.actDim), t * this.n * this.actDim);
    for (let i = 0; i < this.n; i++) {
      this.logp[t * this.n + i] = logp[i];
      this.val[t * this.n + i] = val[i];
      this.rew[t * this.n + i] = rew[i];
      this.done[t * this.n + i] = done[i];
    }
    this.ptr++;
  }

  /** compute GAE + returns and return the flat training batch. `lastVal[i]` is
   *  V(state after the final stored step) for bootstrapping. */
  finish(lastVal: Float32Array, gamma: number, lambda: number) {
    const T = this.horizon, n = this.n;
    const S = T * n;
    const adv = new Float32Array(S);
    const ret = new Float32Array(S);
    for (let i = 0; i < n; i++) {
      let gae = 0;
      for (let t = T - 1; t >= 0; t--) {
        const k = t * n + i;
        const nonterm = 1 - this.done[k];
        const nextVal = t === T - 1 ? lastVal[i] : this.val[(t + 1) * n + i];
        const delta = this.rew[k] + gamma * nextVal * nonterm - this.val[k];
        gae = delta + gamma * lambda * nonterm * gae;
        adv[k] = gae;
        ret[k] = gae + this.val[k];
      }
    }
    this.ptr = 0;
    return { obs: this.obs, act: this.act, logp: this.logp, adv, ret, S };
  }
}
