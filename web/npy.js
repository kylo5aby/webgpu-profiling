// Tensor data helpers shared by the browser page (and usable from Node).
//
// - parseNpy(): minimal NumPy .npy reader (v1/v2/v3 headers, C-order, little-endian)
// - castTensorData(): the equivalent of numpy `astype` between ORT tensor types
// - randomTensorData(): scaled Gaussian noise, mirrors `np.random.randn(*shape) * 0.5`
// - resolveShape(): replace symbolic / non-positive dims with 1, like profile_model.py

// numpy descr -> onnxruntime-web tensor type
const NPY_DESCR_TO_ORT = {
  '<f2': 'float16',
  '<f4': 'float32',
  '<f8': 'float64',
  '<i8': 'int64',
  '<i4': 'int32',
  '<i2': 'int16',
  '|i1': 'int8',
  '|u1': 'uint8',
  '<u2': 'uint16',
  '<u4': 'uint32',
  '<u8': 'uint64',
  '|b1': 'bool',
};

// onnxruntime-web tensor type -> TypedArray constructor
export const ORT_TYPE_TO_ARRAY = {
  float16: Uint16Array, // raw IEEE-754 half bits; ort-web accepts Uint16Array for 'float16'
  float32: Float32Array,
  float64: Float64Array,
  int64: BigInt64Array,
  int32: Int32Array,
  int16: Int16Array,
  int8: Int8Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  uint64: BigUint64Array,
  bool: Uint8Array,
};

export function resolveShape(shape) {
  return shape.map((d) => (typeof d === 'number' && Number.isInteger(d) && d > 0 ? d : 1));
}

export function numElements(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

// ---------------------------------------------------------------------------------------------
// float16 <-> float32
// ---------------------------------------------------------------------------------------------

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

export function f16ToF32(h) {
  const s = (h & 0x8000) << 16;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) {
    if (m === 0) {
      _u32[0] = s;
    } else {
      // subnormal
      let mant = m;
      let exp = 113; // 127 - 14
      while ((mant & 0x400) === 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ff;
      _u32[0] = s | (exp << 23) | (mant << 13);
    }
  } else if (e === 0x1f) {
    _u32[0] = s | 0x7f800000 | (m << 13); // inf / nan
  } else {
    _u32[0] = s | ((e + 112) << 23) | (m << 13);
  }
  return _f32[0];
}

export function f32ToF16(f) {
  _f32[0] = f;
  const x = _u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) {
    return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / nan
  }
  let e = exp - 127 + 15;
  if (e >= 0x1f) {
    return sign | 0x7c00; // overflow -> inf
  }
  if (e <= 0) {
    if (e < -10) return sign; // underflow -> 0
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1))) half++;
    return sign | half;
  }
  let half = sign | (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++; // round-to-nearest-even
  return half;
}

// ---------------------------------------------------------------------------------------------
// .npy
// ---------------------------------------------------------------------------------------------

/**
 * Parse a .npy file.
 * @param {ArrayBuffer} buffer
 * @returns {{ descr: string, type: string, shape: number[], data: ArrayBufferView }}
 */
export function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 6));
  if (magic !== '\x93NUMPY') {
    throw new Error('Not a .npy file (bad magic).');
  }
  const major = bytes[6];
  const view = new DataView(buffer);
  let headerLen;
  let headerStart;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    headerStart = 10;
  } else {
    headerLen = view.getUint32(8, true);
    headerStart = 12;
  }
  const header = new TextDecoder('latin1').decode(bytes.subarray(headerStart, headerStart + headerLen));
  const dataStart = headerStart + headerLen;

  const descrMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
  const orderMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);
  const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!descrMatch || !shapeMatch) {
    throw new Error(`Cannot parse .npy header: ${header}`);
  }
  if (orderMatch && orderMatch[1] === 'True') {
    throw new Error('Fortran-ordered .npy is not supported.');
  }
  const descr = descrMatch[1];
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));

  const type = NPY_DESCR_TO_ORT[descr];
  if (!type) {
    throw new Error(`Unsupported .npy dtype '${descr}'.`);
  }
  const Ctor = ORT_TYPE_TO_ARRAY[type];
  const count = numElements(shape);
  const byteLen = count * Ctor.BYTES_PER_ELEMENT;
  if (dataStart + byteLen > buffer.byteLength) {
    throw new Error(`.npy payload truncated: need ${byteLen} bytes, have ${buffer.byteLength - dataStart}.`);
  }
  // Copy into a fresh buffer so the TypedArray is always correctly aligned.
  const data = new Ctor(bytes.slice(dataStart, dataStart + byteLen).buffer);
  return { descr, type, shape, data };
}

// ---------------------------------------------------------------------------------------------
// casting / random data
// ---------------------------------------------------------------------------------------------

function toNumber(data, type, i) {
  if (type === 'float16') return f16ToF32(data[i]);
  if (type === 'int64' || type === 'uint64') return Number(data[i]);
  return data[i];
}

function fromNumber(v, type) {
  if (type === 'float16') return f32ToF16(v);
  if (type === 'int64') return BigInt(Math.trunc(v));
  if (type === 'uint64') return BigInt(Math.max(0, Math.trunc(v)));
  if (type === 'bool') return v ? 1 : 0;
  if (type === 'float32' || type === 'float64') return v;
  return Math.trunc(v);
}

/** numpy-`astype`-like conversion between ORT tensor types. */
export function castTensorData(data, fromType, toType) {
  if (fromType === toType) return data;
  const Ctor = ORT_TYPE_TO_ARRAY[toType];
  if (!Ctor) throw new Error(`Unsupported target type '${toType}'.`);
  const out = new Ctor(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = fromNumber(toNumber(data, fromType, i), toType);
  }
  return out;
}

/** Standard normal via Box-Muller. */
function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** `(np.random.randn(n) * scale).astype(type)` */
export function randomTensorData(type, count, scale = 0.5) {
  const Ctor = ORT_TYPE_TO_ARRAY[type];
  if (!Ctor) throw new Error(`Unsupported tensor type '${type}'.`);
  const out = new Ctor(count);
  for (let i = 0; i < count; i++) {
    out[i] = fromNumber(randn() * scale, type);
  }
  return out;
}
