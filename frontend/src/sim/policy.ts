/**
 * Client-side Transformer actor — a faithful TS port of the PyTorch
 * `TransformerActorCritic` actor path (see backend/rl/model.py), so the browser
 * drives cars with the exact policy that was trained on the GPU.
 *
 * For playback we only need the deterministic action **mean** (greedy), so the
 * critic / log-std / sampling are omitted. Post-norm encoder layers, 4-head
 * self-attention, exact-erf GELU, and eps=1e-5 LayerNorm all mirror PyTorch.
 *
 * Hot-path buffers are pre-allocated on the instance and reused across calls
 * (forward() runs once per car per step — no per-call allocation / GC churn).
 */

export interface WTensor {
  shape: number[];
  data: number[];
}

export interface PolicyArch {
  obsDim: number;
  actDim: number;
  window: number;
  d_model: number;
  nhead: number;
  num_layers: number;
  dim_ff: number;
}

export interface PolicyFile {
  arch: PolicyArch;
  physics: Record<string, number>;
  weights: Record<string, WTensor>;
  trainedTrack?: string;
  globalStep?: number;
  checkpoint?: string;
}

// exact-erf GELU (Abramowitz & Stegun 7.1.26, max error ~1.5e-7) — matches
// PyTorch's default F.gelu(approximate="none") to float tolerance.
function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return s * y;
}
function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

/** y[o] = sum_i x[i]*W[o*inDim+i] + b[o]  (W is row-major [outDim, inDim]) */
function linear(
  x: Float32Array,
  W: Float32Array,
  b: Float32Array | null,
  outDim: number,
  inDim: number,
  out: Float32Array
): void {
  for (let o = 0; o < outDim; o++) {
    let acc = b ? b[o] : 0;
    const base = o * inDim;
    for (let i = 0; i < inDim; i++) acc += x[i] * W[base + i];
    out[o] = acc;
  }
}

/** in-place LayerNorm over a length-`dim` vector (biased variance, eps=1e-5) */
function layerNorm(x: Float32Array, g: Float32Array, b: Float32Array, dim: number): void {
  let mean = 0;
  for (let i = 0; i < dim; i++) mean += x[i];
  mean /= dim;
  let varr = 0;
  for (let i = 0; i < dim; i++) {
    const d = x[i] - mean;
    varr += d * d;
  }
  varr /= dim;
  const inv = 1 / Math.sqrt(varr + 1e-5);
  for (let i = 0; i < dim; i++) x[i] = (x[i] - mean) * inv * g[i] + b[i];
}

export class Policy {
  readonly arch: PolicyArch;
  readonly physics: Record<string, number>;
  readonly trainedTrack: string;
  readonly globalStep: number;
  private W: Record<string, Float32Array> = {};
  private pe: Float32Array; // [window * d_model]

  // pre-allocated hot-path scratch (reused every forward call)
  private H: Float32Array[];
  private Q: Float32Array[];
  private Kk: Float32Array[];
  private V: Float32Array[];
  private attn: Float32Array[];
  private scores: Float32Array;
  private tmpO: Float32Array;
  private tmpF: Float32Array;
  private tmpD: Float32Array;
  private hid: Float32Array;
  private outMean: Float32Array;

  constructor(f: PolicyFile) {
    this.arch = f.arch;
    this.physics = f.physics;
    this.trainedTrack = f.trainedTrack ?? "default";
    this.globalStep = f.globalStep ?? 0;
    for (const k in f.weights) this.W[k] = Float32Array.from(f.weights[k].data);

    const { window: K, d_model: D, dim_ff, actDim } = this.arch;

    // fixed sinusoidal positional encoding (mirrors PositionalEncoding)
    this.pe = new Float32Array(K * D);
    for (let pos = 0; pos < K; pos++) {
      for (let j = 0; 2 * j < D; j++) {
        const div = Math.exp((-Math.log(10000.0) / D) * (2 * j));
        this.pe[pos * D + 2 * j] = Math.sin(pos * div);
        if (2 * j + 1 < D) this.pe[pos * D + 2 * j + 1] = Math.cos(pos * div);
      }
    }

    const mk = () => Array.from({ length: K }, () => new Float32Array(D));
    this.H = mk();
    this.Q = mk();
    this.Kk = mk();
    this.V = mk();
    this.attn = mk();
    this.scores = new Float32Array(K);
    this.tmpO = new Float32Array(D);
    this.tmpF = new Float32Array(dim_ff);
    this.tmpD = new Float32Array(D);
    this.hid = new Float32Array(64);
    this.outMean = new Float32Array(actDim);
  }

  /**
   * Forward pass for one car.
   * @param window flat Float32Array of shape [K * obsDim] (oldest→newest)
   * @returns the 3 action means (steer, accel, brake), UNclipped. NOTE: the
   *   returned array is reused across calls — read/clip it before the next call.
   */
  forward(window: Float32Array): Float32Array {
    const { obsDim, d_model: D, nhead, num_layers, dim_ff, window: K } = this.arch;
    const headDim = D / nhead;
    const scale = 1 / Math.sqrt(headDim);
    const { H, Q, Kk, V, attn, scores, tmpO, tmpF, tmpD } = this;

    // ---- embed + positional encoding → K hidden tokens ----
    const embedW = this.W["embed.weight"];
    const embedB = this.W["embed.bias"];
    for (let k = 0; k < K; k++) {
      const h = H[k];
      linear(window.subarray(k * obsDim, (k + 1) * obsDim), embedW, embedB, D, obsDim, h);
      const pek = k * D;
      for (let i = 0; i < D; i++) h[i] += this.pe[pek + i];
    }

    for (let L = 0; L < num_layers; L++) {
      const p = `encoder.layers.${L}.`;
      const inW = this.W[p + "self_attn.in_proj_weight"]; // [3D, D]
      const inB = this.W[p + "self_attn.in_proj_bias"]; // [3D]
      const outW = this.W[p + "self_attn.out_proj.weight"]; // [D, D]
      const outB = this.W[p + "self_attn.out_proj.bias"];
      const n1W = this.W[p + "norm1.weight"];
      const n1B = this.W[p + "norm1.bias"];
      const n2W = this.W[p + "norm2.weight"];
      const n2B = this.W[p + "norm2.bias"];
      const l1W = this.W[p + "linear1.weight"]; // [dim_ff, D]
      const l1B = this.W[p + "linear1.bias"];
      const l2W = this.W[p + "linear2.weight"]; // [D, dim_ff]
      const l2B = this.W[p + "linear2.bias"];

      // q,k,v projections (in_proj packs [Wq;Wk;Wv] as rows)
      for (let k = 0; k < K; k++) {
        const h = H[k];
        const qk = Q[k], kk = Kk[k], vk = V[k];
        for (let o = 0; o < D; o++) {
          let q = inB[o], kv = inB[D + o], v = inB[2 * D + o];
          const rq = o * D, rk = (D + o) * D, rv = (2 * D + o) * D;
          for (let i = 0; i < D; i++) {
            const hi = h[i];
            q += hi * inW[rq + i];
            kv += hi * inW[rk + i];
            v += hi * inW[rv + i];
          }
          qk[o] = q;
          kk[o] = kv;
          vk[o] = v;
        }
      }

      // multi-head self-attention (no mask; full window)
      for (let i = 0; i < K; i++) {
        const ai = attn[i];
        ai.fill(0);
        const qi = Q[i];
        for (let hd = 0; hd < nhead; hd++) {
          const off = hd * headDim;
          let maxS = -Infinity;
          for (let j = 0; j < K; j++) {
            let s = 0;
            const kj = Kk[j];
            for (let d = 0; d < headDim; d++) s += qi[off + d] * kj[off + d];
            s *= scale;
            scores[j] = s;
            if (s > maxS) maxS = s;
          }
          let sum = 0;
          for (let j = 0; j < K; j++) {
            const e = Math.exp(scores[j] - maxS);
            scores[j] = e;
            sum += e;
          }
          const invSum = 1 / sum;
          for (let j = 0; j < K; j++) {
            const w = scores[j] * invSum;
            const vj = V[j];
            for (let d = 0; d < headDim; d++) ai[off + d] += w * vj[off + d];
          }
        }
      }

      // out projection + residual + norm1; then FFN + residual + norm2
      for (let i = 0; i < K; i++) {
        linear(attn[i], outW, outB, D, D, tmpO);
        const h = H[i];
        for (let d = 0; d < D; d++) h[d] += tmpO[d];
        layerNorm(h, n1W, n1B, D);

        linear(h, l1W, l1B, dim_ff, D, tmpF);
        for (let d = 0; d < dim_ff; d++) tmpF[d] = gelu(tmpF[d]);
        linear(tmpF, l2W, l2B, D, dim_ff, tmpD);
        for (let d = 0; d < D; d++) h[d] += tmpD[d];
        layerNorm(h, n2W, n2B, D);
      }
    }

    // ---- final norm on the last token, then actor head ----
    const feat = H[K - 1];
    layerNorm(feat, this.W["norm.weight"], this.W["norm.bias"], D);

    const hid = this.hid;
    linear(feat, this.W["actor_mean.0.weight"], this.W["actor_mean.0.bias"], 64, D, hid);
    for (let i = 0; i < 64; i++) hid[i] = Math.tanh(hid[i]);
    linear(hid, this.W["actor_mean.2.weight"], this.W["actor_mean.2.bias"], this.arch.actDim, 64, this.outMean);
    return this.outMean;
  }
}

/** clip a raw action mean exactly as the environment applies it */
export function clipAction(a: Float32Array): [number, number, number] {
  return [
    Math.max(-1, Math.min(1, a[0])),
    Math.max(0, Math.min(1, a[1])),
    Math.max(0, Math.min(1, a[2])),
  ];
}
