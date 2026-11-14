/** In-place radix-2 FFT and a log-magnitude spectrogram for the audio views. */

export function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  if (n & (n - 1)) throw new Error("FFT size must be a power of two");
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

export interface Spectrogram {
  frames: number;
  bins: number;
  hop: number;
  sampleRate: number;
  db: Float32Array; // frames x bins, row-major
}

export function spectrogram(samples: Float32Array, sampleRate: number, size = 512, hop = 256): Spectrogram {
  const bins = size / 2;
  const frames = Math.max(1, Math.floor((samples.length - size) / hop) + 1);
  const db = new Float32Array(frames * bins);
  const win = new Float32Array(size).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1)));
  const re = new Float32Array(size);
  const im = new Float32Array(size);
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < size; i++) {
      re[i] = (samples[f * hop + i] ?? 0) * win[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let b = 0; b < bins; b++) db[f * bins + b] = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-12);
  }
  return { frames, bins, hop, sampleRate, db };
}

export function rmsDb(samples: Float32Array): number {
  let s = 0;
  for (const v of samples) s += v * v;
  return 10 * Math.log10(s / Math.max(1, samples.length) + 1e-20);
}
