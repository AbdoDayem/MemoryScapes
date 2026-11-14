"use client";

/**
 * Loads a memory manifest, renders the splat and attaches the spatial audio engine.
 *
 * Planned:
 * - Three.js + Spark or GaussianSplats3D; load low LOD first, then refine.
 * - First-person walk/fly controls clamped to manifest.navigation.walkable_bounds.
 * - Optional replay of manifest.navigation.original_path.
 * - Audio starts from a user gesture (browser autoplay rules).
 */
export default function MemoryViewer({ memoryId }: { memoryId: string }) {
  return <div data-memory-id={memoryId}>Memory viewer (not implemented)</div>;
}
