// TypeScript mirror of memory.schema.json. Keep the two in sync.

export type Vec3 = [number, number, number];

export interface AudioSource {
  id: string;
  label: string;
  url: string;
  /** null means a diffuse, unplaced source */
  position: Vec3 | null;
  /** URL to a time-indexed path for moving sources */
  trajectory?: string | null;
  ref_distance?: number;
  rolloff?: number;
  gain?: number;
  loop?: boolean;
  confidence?: number;
  /** how the position was found: audio-visual grounding, stereo DOA, both, or not at all
   * (diffuse; narration is the filmer's own voice, heard everywhere at its recorded level) */
  method?: "av" | "doa" | "av+doa" | "diffuse" | "narration";
}

/** SATIE output, played in the viewer by satie-runtime. */
export interface SatieSoundscape {
  script_url: string;
  samples_url: string;
  /** SATIE host id -> AudioSource id; bind each host at that source's position */
  host_map: Record<string, string>;
}

/**
 * Relative URLs (e.g. "splat/lods/lod0.spz") are relative to the memory's asset base,
 * which the API returns next to the manifest.
 */
export interface MemoryManifest {
  id: string;
  version: number;
  coord_frame: {
    units: "meters";
    up: "+y" | "+z";
    /** [lat, lon, alt] */
    origin_geo?: [number, number, number];
    heading_deg?: number;
  };
  splat: {
    /** .spz levels, most detailed first */
    lods: { url: string; count?: number }[];
    /** The same splat as a Spark LoD tree for streaming (a chunked .rad from Spark's build-lod).
     * `chunks` are the files the header names, listed so the API stores them with the memory. */
    rad?: { url: string; chunks: string[]; count?: number };
  };
  navigation?: {
    spawn?: Record<string, unknown>;
    walkable_bounds?: Record<string, unknown>;
    original_path?: string;
    /** ground height in the world frame */
    floor_y?: number;
    /** camera height above the ground during capture */
    eye_height_m?: number;
  };
  audio: {
    ambient_bed?: { url: string; format: "mono" | "stereo" | "ambisonic_foa" | "ambisonic_hoa" };
    sources: AudioSource[];
    soundscape_provider?: { name: string; job_id?: string; version?: string; fallback_reason?: string };
    satie?: SatieSoundscape | null;
  };
  meta: {
    title?: string;
    captured_at?: string;
    visibility?: "private" | "unlisted" | "public";
    owner?: string;
  };
}
