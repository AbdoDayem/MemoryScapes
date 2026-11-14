/** Reader for NumPy .npy files (format versions 1-3, C order). */

export type NpyData = Float32Array | Float64Array | Uint8Array | Int32Array | Int16Array | Uint16Array;

export interface NpyArray {
  dtype: string;
  shape: number[];
  data: NpyData;
}

export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buffer);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
  if (!magic.every((b, i) => bytes[i] === b)) throw new Error("not a .npy file");
  const major = bytes[6];
  const view = new DataView(buffer);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const offset = (major === 1 ? 10 : 12) + headerLen;
  const header = new TextDecoder("latin1").decode(bytes.subarray(major === 1 ? 10 : 12, offset));

  const descr = header.match(/'descr':\s*'([^']+)'/)?.[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shapeStr = header.match(/'shape':\s*\(([^)]*)\)/)?.[1];
  if (!descr || shapeStr === undefined) throw new Error(`unreadable .npy header: ${header}`);
  if (fortran) throw new Error("Fortran-order .npy is not supported");
  const shape = shapeStr
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  if (descr.startsWith(">")) throw new Error("big-endian .npy is not supported");

  const body = buffer.slice(offset); // copy so typed arrays are aligned
  const kind = descr.slice(1);
  let data: NpyData;
  switch (kind) {
    case "f4":
      data = new Float32Array(body, 0, count);
      break;
    case "f8":
      data = new Float64Array(body, 0, count);
      break;
    case "u1":
      data = new Uint8Array(body, 0, count);
      break;
    case "i4":
      data = new Int32Array(body, 0, count);
      break;
    case "i2":
      data = new Int16Array(body, 0, count);
      break;
    case "u2":
      data = new Uint16Array(body, 0, count);
      break;
    case "i8": {
      const big = new BigInt64Array(body, 0, count);
      data = Float64Array.from(big, (v) => Number(v));
      break;
    }
    default:
      throw new Error(`unsupported .npy dtype ${descr}`);
  }
  return { dtype: descr, shape, data };
}
