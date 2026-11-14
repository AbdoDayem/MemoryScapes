import type { MemoryManifest, Vec3 } from "@memoryscapes/manifest";

/**
 * Runtime spatializer: one HRTF PannerNode per placed source, plus an ambient bed.
 * Ambisonic beds will need a decoder such as Omnitone or Resonance Audio.
 */
export class SpatialAudioEngine {
  private ctx = new AudioContext();
  private panners = new Map<string, PannerNode>();

  async load(manifest: MemoryManifest): Promise<void> {
    for (const src of manifest.audio.sources) {
      const buffer = await this.fetchBuffer(src.url);
      const node = this.ctx.createBufferSource();
      node.buffer = buffer;
      node.loop = src.loop ?? true;

      const gain = this.ctx.createGain();
      gain.gain.value = src.gain ?? 1;

      if (src.position) {
        const panner = new PannerNode(this.ctx, {
          panningModel: "HRTF",
          distanceModel: "inverse",
          refDistance: src.ref_distance ?? 1,
          rolloffFactor: src.rolloff ?? 1,
          positionX: src.position[0],
          positionY: src.position[1],
          positionZ: src.position[2],
        });
        node.connect(gain).connect(panner).connect(this.ctx.destination);
        this.panners.set(src.id, panner);
      } else {
        node.connect(gain).connect(this.ctx.destination); // diffuse
      }
      node.start();
    }
    // TODO: ambient bed (stereo or ambisonic decode).
  }

  /** Call every frame with the camera pose. */
  updateListener(position: Vec3, forward: Vec3, up: Vec3): void {
    const l = this.ctx.listener;
    l.positionX.value = position[0];
    l.positionY.value = position[1];
    l.positionZ.value = position[2];
    l.forwardX.value = forward[0];
    l.forwardY.value = forward[1];
    l.forwardZ.value = forward[2];
    l.upX.value = up[0];
    l.upY.value = up[1];
    l.upZ.value = up[2];
  }

  resume(): Promise<void> {
    return this.ctx.resume();
  }

  private async fetchBuffer(url: string): Promise<AudioBuffer> {
    const res = await fetch(url);
    return this.ctx.decodeAudioData(await res.arrayBuffer());
  }
}
