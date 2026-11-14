import { describe, expect, it } from "vitest";

import { fft, spectrogram } from "./fft";
import { parseNpy } from "./npy";
import { gaussians, parsePly } from "./ply";

function npy(descr: string, shape: number[], data: ArrayBufferView, version = 1): ArrayBuffer {
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  const pre = version === 1 ? 10 : 12;
  const pad = 64 - ((pre + header.length + 1) % 64);
  header += " ".repeat(pad % 64) + "\n";
  const out = new Uint8Array(pre + header.length + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, version, 0]);
  const view = new DataView(out.buffer);
  if (version === 1) view.setUint16(8, header.length, true);
  else view.setUint32(8, header.length, true);
  out.set(new TextEncoder().encode(header), pre);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), pre + header.length);
  return out.buffer;
}

describe("parseNpy", () => {
  it("reads float32 (N, 3) points", () => {
    const a = parseNpy(npy("<f4", [2, 3], new Float32Array([1, 2, 3, 4, 5, 6])));
    expect(a.shape).toEqual([2, 3]);
    expect(Array.from(a.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });
  it("reads uint8 colours, float64 and version 2 headers", () => {
    expect(Array.from(parseNpy(npy("|u1", [3], new Uint8Array([7, 8, 9]))).data)).toEqual([7, 8, 9]);
    expect(Array.from(parseNpy(npy("<f8", [2], new Float64Array([0.5, -1]), 2)).data)).toEqual([0.5, -1]);
  });
  it("rejects files that aren't .npy", () => {
    expect(() => parseNpy(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).buffer)).toThrow(/not a .npy/);
  });
});

function ply(props: string[], rows: number[][]): ArrayBuffer {
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${rows.length}\n${props.map((p) => `property float ${p}\n`).join("")}end_header\n`;
  const h = new TextEncoder().encode(header);
  const body = new Float32Array(rows.flat());
  const out = new Uint8Array(h.length + body.byteLength);
  out.set(h);
  out.set(new Uint8Array(body.buffer), h.length);
  return out.buffer;
}

describe("parsePly / gaussians", () => {
  const props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"];
  it("decodes 3DGS columns into centres, colours, opacity and size", () => {
    const p = parsePly(ply(props, [[1, 2, 3, 0, 0, 0, 0, Math.log(0.1), Math.log(0.1), Math.log(0.1)]]));
    expect(p.count).toBe(1);
    const g = gaussians(p);
    expect(Array.from(g.positions)).toEqual([1, 2, 3]);
    expect(g.colors[0]).toBeCloseTo(0.5); // f_dc = 0 is mid grey
    expect(g.opacity[0]).toBeCloseTo(0.5); // logit 0
    expect(g.scale[0]).toBeCloseTo(0.1);
  });
  it("rejects ASCII and truncated files", () => {
    const ascii = new TextEncoder().encode("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n").buffer;
    expect(() => parsePly(ascii)).toThrow(/unsupported PLY format/);
    const full = new Uint8Array(ply(["x", "y", "z"], [[1, 2, 3]]));
    expect(() => parsePly(full.slice(0, full.length - 4).buffer)).toThrow(/truncated/);
  });
});

describe("fft / spectrogram", () => {
  it("puts a pure tone in the right bin", () => {
    const n = 256;
    const re = Float32Array.from({ length: n }, (_, i) => Math.cos((2 * Math.PI * 16 * i) / n));
    const im = new Float32Array(n);
    fft(re, im);
    const mags = Array.from(re, (r, i) => Math.hypot(r, im[i]));
    expect(mags.indexOf(Math.max(...mags.slice(0, n / 2)))).toBe(16);
  });
  it("frames a signal into a spectrogram", () => {
    const sr = 16000;
    const x = Float32Array.from({ length: sr }, (_, i) => Math.sin((2 * Math.PI * 1000 * i) / sr));
    const s = spectrogram(x, sr, 512, 256);
    expect(s.bins).toBe(256);
    expect(s.frames).toBe(Math.floor((sr - 512) / 256) + 1);
    const row = Array.from(s.db.subarray(10 * s.bins, 11 * s.bins));
    expect(row.indexOf(Math.max(...row))).toBe(32); // 1000 Hz / (16000 / 512)
  });
});
