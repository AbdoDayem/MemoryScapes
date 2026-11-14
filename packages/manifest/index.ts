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
  /** how the position was found: audio-visual grounding, stereo DOA, both, or not at all */
  method?: "av" | "doa" | "av+doa" | "diffuse";
}

/** SATIE output, played in the viewer by satie-runtime. */
export interface SatieSoundscape {
  script_url: string;
  samples_url: string;
  /** SATIE host id -> AudioSource id; bind each host at that source's position */
  host_map: Record<string, string>;
}

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
  splat: { lods: { url: string; count?: number }[] };
  navigation?: {
    spawn?: Record<string, unknown>;
    walkable_bounds?: Record<string, unknown>;
    original_path?: string;
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
