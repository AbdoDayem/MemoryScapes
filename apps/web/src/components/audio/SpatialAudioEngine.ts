import type { MemoryManifest, Vec3 } from "@memoryscapes/manifest";

/**
 * Runtime spatializer: one HRTF PannerNode per placed source, plus an ambient bed.
 *
 * Every buffer is decoded before anything starts, and all of them start on one
 * AudioContext time: the separation stage splits the recording so that stems + ambient
 * add back up to it, which only holds if they stay sample-aligned. Unplaced (diffuse)
 * sources and stereo/mono beds play without spatialization. Ambisonic beds still need a
 * decoder (Omnitone or Resonance Audio); until then they play as plain channels.
 */
export class SpatialAudioEngine {
  private ctx = new AudioContext();
  private nodes: AudioBufferSourceNode[] = [];
  private startedAt: number | null = null;
  private duration = 0;

  /** resolve: manifest URL (relative to the memory) -> fetchable URL. */
  constructor(private resolve: (url: string) => string = (u) => u) {}

  async load(manifest: MemoryManifest, onProgress?: (done: number, total: number) => void): Promise<void> {
    const { sources, ambient_bed: bed } = manifest.audio;
    const jobs: { url: string; build: (buffer: AudioBuffer) => AudioNode }[] = sources.map((src) => ({
      url: src.url,
      build: () => {
        const gain = new GainNode(this.ctx, { gain: src.gain ?? 1 });
        if (!src.position) {
          gain.connect(this.ctx.destination); // diffuse: not spatialized
          return gain;
        }
        const panner = new PannerNode(this.ctx, {
          panningModel: "HRTF",
          distanceModel: "inverse",
          refDistance: src.ref_distance ?? 1,
          rolloffFactor: src.rolloff ?? 1,
          positionX: src.position[0],
          positionY: src.position[1],
          positionZ: src.position[2],
        });
        gain.connect(panner).connect(this.ctx.destination);
        return gain;
      },
    }));
    if (bed) {
      if (bed.format.startsWith("ambisonic")) console.warn("Ambisonic bed played without decoding");
      jobs.push({ url: bed.url, build: () => this.ctx.destination });
    }

    let done = 0;
    const buffers = await Promise.all(
      jobs.map(async (job) => {
        const buffer = await this.fetchBuffer(job.url);
        onProgress?.(++done, jobs.length);
        return buffer;
      }),
    );
    this.duration = Math.max(0, ...buffers.map((b) => b.duration));
    this.nodes = buffers.map((buffer, i) => {
      const node = new AudioBufferSourceNode(this.ctx, { buffer, loop: true });
      // Loop everything over the longest buffer so the layers stay aligned.
      node.loopEnd = this.duration;
      node.connect(jobs[i].build(buffer));
      return node;
    });
  }

  /** Start every layer together. Must follow a user gesture (autoplay rules). */
  async start(): Promise<void> {
    await this.ctx.resume();
    if (this.startedAt !== null) return;
    const at = this.ctx.currentTime + 0.1;
    for (const node of this.nodes) node.start(at);
    this.startedAt = at;
  }

  get hasAudio(): boolean {
    return this.nodes.length > 0;
  }

  /** Seconds into the recording, or null before start(). Wraps at the loop. */
  get playbackTime(): number | null {
    if (this.startedAt === null || this.duration === 0) return null;
    return Math.max(0, this.ctx.currentTime - this.startedAt) % this.duration;
  }

  /** Call every frame with the camera pose. */
  updateListener(position: Vec3, forward: Vec3, up: Vec3): void {
    const l = this.ctx.listener;
    if (!l.positionX) {
      // Firefox: no AudioParams on the listener, only the older setters.
      l.setPosition(...position);
      l.setOrientation(...forward, ...up);
      return;
    }
    const t = this.ctx.currentTime;
    l.positionX.setValueAtTime(position[0], t);
    l.positionY.setValueAtTime(position[1], t);
    l.positionZ.setValueAtTime(position[2], t);
    l.forwardX.setValueAtTime(forward[0], t);
    l.forwardY.setValueAtTime(forward[1], t);
    l.forwardZ.setValueAtTime(forward[2], t);
    l.upX.setValueAtTime(up[0], t);
    l.upY.setValueAtTime(up[1], t);
    l.upZ.setValueAtTime(up[2], t);
  }

  async dispose(): Promise<void> {
    for (const node of this.nodes) {
      try {
        node.stop();
      } catch {
        // never started
      }
    }
    this.nodes = [];
    await this.ctx.close();
  }

  private async fetchBuffer(url: string): Promise<AudioBuffer> {
    const res = await fetch(this.resolve(url));
    if (!res.ok) throw new Error(`Audio ${url}: HTTP ${res.status}`);
    return this.ctx.decodeAudioData(await res.arrayBuffer());
  }
}
