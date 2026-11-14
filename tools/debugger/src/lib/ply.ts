/** Reader for binary little-endian PLY vertex data, incl. 3D Gaussian Splatting files. */

export interface PlyVertices {
  count: number;
  properties: string[];
  columns: Record<string, Float32Array>;
}

const SIZES: Record<string, [number, (v: DataView, o: number) => number]> = {
  char: [1, (v, o) => v.getInt8(o)],
  int8: [1, (v, o) => v.getInt8(o)],
  uchar: [1, (v, o) => v.getUint8(o)],
  uint8: [1, (v, o) => v.getUint8(o)],
  short: [2, (v, o) => v.getInt16(o, true)],
  int16: [2, (v, o) => v.getInt16(o, true)],
  ushort: [2, (v, o) => v.getUint16(o, true)],
  uint16: [2, (v, o) => v.getUint16(o, true)],
  int: [4, (v, o) => v.getInt32(o, true)],
  int32: [4, (v, o) => v.getInt32(o, true)],
  uint: [4, (v, o) => v.getUint32(o, true)],
  uint32: [4, (v, o) => v.getUint32(o, true)],
  float: [4, (v, o) => v.getFloat32(o, true)],
  float32: [4, (v, o) => v.getFloat32(o, true)],
  double: [8, (v, o) => v.getFloat64(o, true)],
  float64: [8, (v, o) => v.getFloat64(o, true)],
};

export function parsePly(buffer: ArrayBuffer): PlyVertices {
  const bytes = new Uint8Array(buffer);
  const marker = new TextEncoder().encode("end_header\n");
  let end = -1;
  for (let i = 0; i < Math.min(bytes.length, 65536) - marker.length; i++) {
    if (marker.every((b, j) => bytes[i + j] === b)) {
      end = i + marker.length;
      break;
    }
  }
  if (end < 0) throw new Error("PLY header not found");
  const lines = new TextDecoder().decode(bytes.subarray(0, end)).split("\n");
  if (lines[0].trim() !== "ply") throw new Error("not a PLY file");
  const format = lines.find((l) => l.startsWith("format"))?.split(/\s+/)[1];
  if (format !== "binary_little_endian") throw new Error(`unsupported PLY format: ${format}`);

  let count = 0;
  let inVertex = false;
  let offsetBeforeVertex = 0; // bytes of elements declared before "vertex" (unsupported if > 0)
  const props: { name: string; size: number; read: (v: DataView, o: number) => number }[] = [];
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "element") {
      inVertex = parts[1] === "vertex";
      if (inVertex) count = Number(parts[2]);
      else if (count === 0) offsetBeforeVertex += 1;
    } else if (parts[0] === "property" && inVertex) {
      if (parts[1] === "list") throw new Error("list properties on vertices are not supported");
      const t = SIZES[parts[1]];
      if (!t) throw new Error(`unknown PLY type ${parts[1]}`);
      props.push({ name: parts[2], size: t[0], read: t[1] });
    }
  }
  if (offsetBeforeVertex) throw new Error("PLY elements before 'vertex' are not supported");
  const stride = props.reduce((a, p) => a + p.size, 0);
  if (end + stride * count > buffer.byteLength) throw new Error("PLY file is truncated");

  const view = new DataView(buffer, end);
  const columns: Record<string, Float32Array> = {};
  for (const p of props) columns[p.name] = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let o = i * stride;
    for (const p of props) {
      columns[p.name][i] = p.read(view, o);
      o += p.size;
    }
  }
  return { count, properties: props.map((p) => p.name), columns };
}

const SH_C0 = 0.28209479177387814;

export interface GaussianView {
  count: number;
  positions: Float32Array; // xyz interleaved
  colors: Float32Array; // rgb 0..1 interleaved
  opacity: Float32Array; // 0..1
  scale: Float32Array; // mean linear scale per gaussian
}

/** Centres, base colours, opacities and sizes of 3DGS gaussians, or plain coloured points. */
export function gaussians(ply: PlyVertices): GaussianView {
  const { count, columns: c } = ply;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const opacity = new Float32Array(count);
  const scale = new Float32Array(count);
  const hasDc = "f_dc_0" in c;
  const hasRgb = "red" in c;
  for (let i = 0; i < count; i++) {
    positions[i * 3] = c.x[i];
    positions[i * 3 + 1] = c.y[i];
    positions[i * 3 + 2] = c.z[i];
    for (let k = 0; k < 3; k++) {
      let v = 0.7;
      if (hasDc) v = 0.5 + SH_C0 * c[`f_dc_${k}`][i];
      else if (hasRgb) v = c[["red", "green", "blue"][k]][i] / 255;
      colors[i * 3 + k] = Math.min(1, Math.max(0, v));
    }
    opacity[i] = "opacity" in c ? 1 / (1 + Math.exp(-c.opacity[i])) : 1;
    scale[i] = "scale_0" in c ? (Math.exp(c.scale_0[i]) + Math.exp(c.scale_1[i]) + Math.exp(c.scale_2[i])) / 3 : 0.05;
  }
  return { count, positions, colors, opacity, scale };
}
