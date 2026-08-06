/// <reference types="@webgpu/types" />
/**
 * WebGPU batched Transformer actor. Computes the greedy action for ALL cars in
 * one compute dispatch (one workgroup per car, 64 threads = d_model), so the
 * visitor's GPU drives hundreds–thousands of cars. Numerically matches the CPU
 * `Policy` (which is parity-validated against Python).
 *
 * Fixed architecture: obsDim=10, d_model=64, window(K)=8, nhead=4, headDim=16,
 * dim_ff=128, num_layers=2, actDim=3.
 */
import type { PolicyFile } from "./policy";

const O = 10, D = 64, K = 8, F = 128, A = 3;

interface Off {
  [k: string]: number;
}

/** pack all weights (+ positional encoding) into one flat buffer, recording offsets */
function packWeights(f: PolicyFile): { data: Float32Array; off: Off } {
  const w = f.weights;
  const off: Off = {};
  const parts: number[][] = [];
  let cursor = 0;
  const add = (name: string, arr: number[] | Float32Array) => {
    off[name] = cursor;
    parts.push(arr as number[]);
    cursor += arr.length;
  };

  // positional encoding (same formula as policy.ts)
  const pe = new Float32Array(K * D);
  for (let pos = 0; pos < K; pos++)
    for (let j = 0; 2 * j < D; j++) {
      const div = Math.exp((-Math.log(10000.0) / D) * (2 * j));
      pe[pos * D + 2 * j] = Math.sin(pos * div);
      if (2 * j + 1 < D) pe[pos * D + 2 * j + 1] = Math.cos(pos * div);
    }

  add("embW", w["embed.weight"].data);
  add("embB", w["embed.bias"].data);
  add("pe", Array.from(pe));
  for (let L = 0; L < 2; L++) {
    const p = `encoder.layers.${L}.`;
    add(`l${L}_inW`, w[p + "self_attn.in_proj_weight"].data);
    add(`l${L}_inB`, w[p + "self_attn.in_proj_bias"].data);
    add(`l${L}_outW`, w[p + "self_attn.out_proj.weight"].data);
    add(`l${L}_outB`, w[p + "self_attn.out_proj.bias"].data);
    add(`l${L}_n1W`, w[p + "norm1.weight"].data);
    add(`l${L}_n1B`, w[p + "norm1.bias"].data);
    add(`l${L}_n2W`, w[p + "norm2.weight"].data);
    add(`l${L}_n2B`, w[p + "norm2.bias"].data);
    add(`l${L}_l1W`, w[p + "linear1.weight"].data);
    add(`l${L}_l1B`, w[p + "linear1.bias"].data);
    add(`l${L}_l2W`, w[p + "linear2.weight"].data);
    add(`l${L}_l2B`, w[p + "linear2.bias"].data);
  }
  add("normW", w["norm.weight"].data);
  add("normB", w["norm.bias"].data);
  add("am0W", w["actor_mean.0.weight"].data);
  add("am0B", w["actor_mean.0.bias"].data);
  add("am2W", w["actor_mean.2.weight"].data);
  add("am2B", w["actor_mean.2.bias"].data);

  const data = new Float32Array(cursor);
  let c = 0;
  for (const part of parts) {
    data.set(part, c);
    c += part.length;
  }
  return { data, off };
}

/** WGSL for one post-norm encoder layer, offsets baked in. Wrapped in a block. */
function layerWGSL(L: number, off: Off): string {
  const inW = off[`l${L}_inW`], inB = off[`l${L}_inB`], outW = off[`l${L}_outW`], outB = off[`l${L}_outB`];
  const n1W = off[`l${L}_n1W`], n1B = off[`l${L}_n1B`], n2W = off[`l${L}_n2W`], n2B = off[`l${L}_n2B`];
  const l1W = off[`l${L}_l1W`], l1B = off[`l${L}_l1B`], l2W = off[`l${L}_l2W`], l2B = off[`l${L}_l2B`];
  return `
  { // ---- encoder layer ${L} ----
    // q,k,v projections: thread computes column tid for every token
    for (var k=0u;k<8u;k++){
      var q=W[${inB}u+tid]; var kk=W[${inB}u+64u+tid]; var vv=W[${inB}u+128u+tid];
      for (var i=0u;i<64u;i++){
        let h=shH[k*64u+i];
        q  += h*W[${inW}u + tid*64u + i];
        kk += h*W[${inW}u + (64u+tid)*64u + i];
        vv += h*W[${inW}u + (128u+tid)*64u + i];
      }
      shA[k*64u+tid]=q; shB[k*64u+tid]=kk; shC[k*64u+tid]=vv;
    }
    workgroupBarrier();
    // multi-head self-attention: threads 0..31 own one (token,head) pair
    if (tid < 32u){
      let qi = tid / 4u; let hh = tid % 4u; let base = hh*16u;
      var sc: array<f32,8>;
      var mx = -1e30;
      for (var j=0u;j<8u;j++){
        var s=0.0;
        for (var d=0u;d<16u;d++){ s += shA[qi*64u+base+d]*shB[j*64u+base+d]; }
        s = s*0.25; sc[j]=s; mx=max(mx,s);
      }
      var sum=0.0;
      for (var j=0u;j<8u;j++){ let e=exp(sc[j]-mx); sc[j]=e; sum+=e; }
      for (var d=0u;d<16u;d++){
        var o=0.0;
        for (var j=0u;j<8u;j++){ o += sc[j]/sum * shC[j*64u+base+d]; }
        shF[qi*64u+base+d]=o; // attnOut lives in shF during attention
      }
    }
    workgroupBarrier();
    // out projection + residual
    for (var k=0u;k<8u;k++){
      var acc=W[${outB}u+tid];
      for (var i=0u;i<64u;i++){ acc += shF[k*64u+i]*W[${outW}u + tid*64u + i]; }
      shH[k*64u+tid] += acc;
    }
    workgroupBarrier();
    // norm1
    if (tid<8u){
      var m=0.0; for (var d=0u;d<64u;d++){ m+=shH[tid*64u+d]; } m=m/64.0;
      var v=0.0; for (var d=0u;d<64u;d++){ let x=shH[tid*64u+d]-m; v+=x*x; } v=v/64.0;
      shMean[tid]=m; shInv[tid]=inverseSqrt(v+1e-5);
    }
    workgroupBarrier();
    for (var k=0u;k<8u;k++){ shH[k*64u+tid]=(shH[k*64u+tid]-shMean[k])*shInv[k]*W[${n1W}u+tid]+W[${n1B}u+tid]; }
    workgroupBarrier();
    // FFN linear1 + gelu -> shF[k*128 + f]; each thread does f=tid and f=tid+64
    for (var k=0u;k<8u;k++){
      var a1=W[${l1B}u+tid];
      for (var i=0u;i<64u;i++){ a1 += shH[k*64u+i]*W[${l1W}u + tid*64u + i]; }
      shF[k*128u+tid]=gelu(a1);
      var a2=W[${l1B}u+tid+64u];
      for (var i=0u;i<64u;i++){ a2 += shH[k*64u+i]*W[${l1W}u + (tid+64u)*64u + i]; }
      shF[k*128u+tid+64u]=gelu(a2);
    }
    workgroupBarrier();
    // FFN linear2 + residual
    for (var k=0u;k<8u;k++){
      var acc=W[${l2B}u+tid];
      for (var ff=0u;ff<128u;ff++){ acc += shF[k*128u+ff]*W[${l2W}u + tid*128u + ff]; }
      shH[k*64u+tid] += acc;
    }
    workgroupBarrier();
    // norm2
    if (tid<8u){
      var m=0.0; for (var d=0u;d<64u;d++){ m+=shH[tid*64u+d]; } m=m/64.0;
      var v=0.0; for (var d=0u;d<64u;d++){ let x=shH[tid*64u+d]-m; v+=x*x; } v=v/64.0;
      shMean[tid]=m; shInv[tid]=inverseSqrt(v+1e-5);
    }
    workgroupBarrier();
    for (var k=0u;k<8u;k++){ shH[k*64u+tid]=(shH[k*64u+tid]-shMean[k])*shInv[k]*W[${n2W}u+tid]+W[${n2B}u+tid]; }
    workgroupBarrier();
  }`;
}

function buildShader(off: Off): string {
  return `
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> obs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outA: array<f32>;

var<workgroup> shH: array<f32, 512>;
var<workgroup> shA: array<f32, 512>;
var<workgroup> shB: array<f32, 512>;
var<workgroup> shC: array<f32, 512>;
var<workgroup> shF: array<f32, 1024>;
var<workgroup> shMean: array<f32, 8>;
var<workgroup> shInv: array<f32, 8>;

fn erf(x:f32)->f32 {
  let s = select(1.0, -1.0, x < 0.0);
  let ax = abs(x);
  let t = 1.0/(1.0 + 0.3275911*ax);
  let y = 1.0 - ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-ax*ax);
  return s*y;
}
fn gelu(x:f32)->f32 { return 0.5*x*(1.0 + erf(x*0.70710678118)); }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let car = wid.x;
  let tid = lid.x;
  let obsBase = car*${K * O}u;

  // embed + positional encoding
  for (var k=0u;k<8u;k++){
    var acc=W[${off.embB}u+tid];
    let ob = obsBase + k*10u;
    for (var o=0u;o<10u;o++){ acc += obs[ob+o]*W[${off.embW}u + tid*10u + o]; }
    acc += W[${off.pe}u + k*64u + tid];
    shH[k*64u+tid]=acc;
  }
  workgroupBarrier();

  ${layerWGSL(0, off)}
  ${layerWGSL(1, off)}

  // final norm on last token (k=7) -> feat in shA[0..63]
  if (tid==0u){
    var m=0.0; for (var d=0u;d<64u;d++){ m+=shH[7u*64u+d]; } m=m/64.0;
    var v=0.0; for (var d=0u;d<64u;d++){ let x=shH[7u*64u+d]-m; v+=x*x; } v=v/64.0;
    shMean[0]=m; shInv[0]=inverseSqrt(v+1e-5);
  }
  workgroupBarrier();
  shA[tid] = (shH[7u*64u+tid]-shMean[0])*shInv[0]*W[${off.normW}u+tid]+W[${off.normB}u+tid];
  workgroupBarrier();

  // actor head: hid[j]=tanh(feat·am0W[j]+am0B[j]); thread j=tid
  var hj=W[${off.am0B}u+tid];
  for (var d=0u;d<64u;d++){ hj += shA[d]*W[${off.am0W}u + tid*64u + d]; }
  shB[tid]=tanh(hj);
  workgroupBarrier();

  // mean[a]=hid·am2W[a]+am2B[a]; threads 0..2 write output
  if (tid<3u){
    var a=W[${off.am2B}u+tid];
    for (var j=0u;j<64u;j++){ a += shB[j]*W[${off.am2W}u + tid*64u + j]; }
    outA[car*3u+tid]=a;
  }
}`;
}

export class GpuPolicy {
  readonly arch: PolicyFile["arch"];
  readonly physics: PolicyFile["physics"];
  readonly trainedTrack: string;
  readonly globalStep: number;
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private wBuf: GPUBuffer;
  private obsBuf: GPUBuffer | null = null;
  private outBuf: GPUBuffer | null = null;
  private stagingBuf: GPUBuffer | null = null;
  private bind: GPUBindGroup | null = null;
  private capacity = 0;

  private constructor(device: GPUDevice, f: PolicyFile, packed: { data: Float32Array; off: Off }) {
    this.device = device;
    this.arch = f.arch;
    this.physics = f.physics;
    this.trainedTrack = f.trainedTrack ?? "default";
    this.globalStep = f.globalStep ?? 0;

    this.wBuf = device.createBuffer({
      size: packed.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.wBuf, 0, packed.data as unknown as GPUAllowSharedBufferSource);

    const module = device.createShaderModule({ code: buildShader(packed.off) });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  static async create(f: PolicyFile): Promise<GpuPolicy> {
    if (!("gpu" in navigator)) throw new Error("WebGPU unavailable");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    return new GpuPolicy(device, f, packWeights(f));
  }

  private ensureCapacity(n: number) {
    if (n <= this.capacity && this.obsBuf) return;
    this.obsBuf?.destroy();
    this.outBuf?.destroy();
    this.stagingBuf?.destroy();
    const KO = this.arch.window * this.arch.obsDim;
    this.obsBuf = this.device.createBuffer({ size: n * KO * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.outBuf = this.device.createBuffer({ size: n * A * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.stagingBuf = this.device.createBuffer({ size: n * A * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.bind = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.wBuf } },
        { binding: 1, resource: { buffer: this.obsBuf } },
        { binding: 2, resource: { buffer: this.outBuf } },
      ],
    });
    this.capacity = n;
  }

  /** @param obs flat [n*K*O]. @returns flat [n*3] action means (unclipped). */
  async forwardBatch(obs: Float32Array, n: number): Promise<Float32Array> {
    this.ensureCapacity(n);
    this.device.queue.writeBuffer(
      this.obsBuf!,
      0,
      obs as unknown as GPUAllowSharedBufferSource,
      0,
      n * this.arch.window * this.arch.obsDim
    );
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind!);
    pass.dispatchWorkgroups(n);
    pass.end();
    enc.copyBufferToBuffer(this.outBuf!, 0, this.stagingBuf!, 0, n * A * 4);
    this.device.queue.submit([enc.finish()]);
    await this.stagingBuf!.mapAsync(GPUMapMode.READ, 0, n * A * 4);
    const out = new Float32Array(this.stagingBuf!.getMappedRange(0, n * A * 4)).slice();
    this.stagingBuf!.unmap();
    return out;
  }
}
