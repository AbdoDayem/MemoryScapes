import type { MemoryManifest } from "@memoryscapes/manifest";
import type { Camera } from "three";
import type { SatieScene } from "satie-runtime/three";

/**
 * The viewer's only spatializer: SATIE's own runtime (satie-runtime/three) playing the
 * memory's SATIE soundscape.
 *
 * The worker's soundscape stage had SATIE arrange the separated, localized stems; the
 * manifest carries the script, samples.json (our stems, as `provided` clips) and the host
 * map. Each host is bound to an anchor at its source's position, and the camera is the
 * listener, so SATIE does the spatialization. Positions are metres in the memory's
 * frame (+y up), which is also SATIE's, so nothing is converted.
 */
export class SatieAudioEngine {
  private scene: SatieScene | null = null;
  private started = false;
  /** Hosts bound at load; the script's other hosts (if any) are in `warnings`. */
  hosts: string[] = [];
  warnings: string[] = [];

  /** resolve: manifest URL (relative to the memory) -> fetchable URL. */
  constructor(private resolve: (url: string) => string = (u) => u) {}

  async load(manifest: MemoryManifest): Promise<void> {
    const satie = manifest.audio.satie;
    if (!satie) throw new Error("This memory has no SATIE soundscape");
    const [{ SatieScene }, THREE] = await Promise.all([import("satie-runtime/three"), import("three")]);
    // A fixed seed: the same memory sounds the same on every visit.
    const scene = await SatieScene.load(this.resolve(satie.script_url), this.resolve(satie.samples_url), { seed: 7 });

    const positions = new Map(manifest.audio.sources.map((s) => [s.id, s.position]));
    for (const [host, sourceId] of Object.entries(satie.host_map)) {
      const position = positions.get(sourceId);
      if (!position) {
        this.warnings.push(`host ${host}: source ${sourceId} has no position`);
        continue;
      }
      const anchor = new THREE.Object3D(); // a point: the source as it was localized
      anchor.position.set(...position);
      anchor.updateMatrixWorld(true);
      try {
        scene.bind(host, anchor, { anchor: "origin" });
        this.hosts.push(host);
      } catch (err) {
        // A placed source SATIE left out of the script: nothing to bind.
        this.warnings.push(`host ${host}: ${err instanceof Error ? err.message : err}`);
      }
    }
    this.warnings.push(...scene.bindWarnings);
    this.scene = scene;
  }

  /** Must follow a user gesture (autoplay rules). */
  async start(): Promise<void> {
    if (!this.scene || this.started) return;
    await this.scene.start();
    this.started = true;
  }

  /** Every frame, after the camera moves: SATIE reads the listener (position and facing) from it. */
  update(camera: Camera): void {
    this.scene?.update(camera);
  }

  /** For the dev mixer. */
  get runtime(): SatieScene | null {
    return this.scene;
  }

  async dispose(): Promise<void> {
    this.scene?.dispose();
    this.scene = null;
  }
}
