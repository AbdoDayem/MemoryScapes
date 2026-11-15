/** Shapes of the pipeline's stage outputs (see services/workers/README.md and docs/SPLAT_OUTPUTS.md). */

export type Vec3 = [number, number, number];

export interface MemorySummary {
  id: string;
  stages: Record<string, number>;
}

export interface FileEntry {
  key: string;
  size: number;
}

export interface IngestMeta {
  mic: { channels: number; layout: "mono" | "stereo" | "foa"; spacing_m?: number };
  scenario?: string;
  /** Spatial (stereo) video: both eyes were kept (docs/SPLAT_OUTPUTS.md, ingest/). */
  stereo?: { baseline_m: number | null; hfov_deg: number | null; left_view: number; right_view: number;
             right_frames: string } | null;
}

export interface Camera {
  frame_id: string;
  t: number;
  width: number;
  height: number;
  K: number[][];
  c2w: number[][];
  image_key: string | null;
}

export interface StemInfo {
  id: string;
  label: string;
  kind: string;
  key: string;
  level_dbfs: number;
  activity: number[];
  tag_score: number;
}

export interface StemsIndex {
  sample_rate: number;
  channels: number;
  layout: string;
  duration_s: number;
  window_times: number[];
  stems: StemInfo[];
  ambient_key: string;
  tags: Record<string, number>;
  dropped?: { label: string; reason: string }[];
  /** What cleaning did before separation (services/workers/memoryscapes_workers/audio/denoise.py).
   *  Absent for memories separated before cleaning existed. */
  cleaning?: {
    highpass_hz?: number | null;
    /** "suppressed", "off", or why wind suppression was skipped. */
    wind?: string;
    /** Share of frames that looked like mic wind (two-channel recordings only). */
    wind_like_frames?: number;
    /** dB removed per band: "20-100 Hz", "100-500 Hz", "500-2000 Hz", "all". */
    removed_db?: Record<string, number>;
  };
}

export interface LocalizedSource {
  id: string;
  label: string;
  kind: string;
  stem_key: string;
  position: Vec3 | null;
  spread_m: number | null;
  confidence: number;
  method: "av" | "doa" | "av+doa" | "diffuse" | "narration";
  level_dbfs: number;
  level_db_at_1m: number | null;
  trajectory?: string | null;
  /** Id of the source this one shares a position with (both found on one object). */
  shares_with?: string | null;
}

export interface Localized {
  bounds: { min: Vec3; max: Vec3 };
  sources: LocalizedSource[];
}

export interface SoundscapeScene {
  provider: { name: string; notes: string[]; fallback_reason?: string; realize_id?: string | null };
  ambient_bed: { key: string; format: string };
  sources: { id: string; label: string; key: string; position: Vec3 | null; ref_distance: number; rolloff: number; gain: number; loop: boolean; confidence: number; method: string }[];
  satie: { script_key: string; samples_key: string; host_map: Record<string, string> } | null;
}

export interface SplatMeta {
  trainer?: string;
  trainer_version?: string;
  iterations?: number;
  sh_degree?: number;
  units?: string;
  up?: string;
  eval_frames?: string[];
}

export type SplatMetrics = Record<string, number[]> & { iteration?: number[] };

/** splat/train_report.json: the run's scores and settings (docs/SPLAT_OUTPUTS.md). */
export interface TrainReport {
  psnr?: number;
  psnr_split?: "holdout" | "train";
  /** PSNR after fitting each scored frame's colour: structure, not exposure. */
  psnr_cc?: number;
  depth_error_vs_sfm?: number | null;
  train_images?: number;
  right_eye_images?: number;
  depth_supervised_images?: number;
  exposure_comp?: boolean;
  /** Learned brightness per training frame (1 = the splat's own); null without exposure_comp. */
  exposure_gains?: Record<string, number> | null;
  [key: string]: unknown;
}

export interface Truth {
  scenario?: string;
  description?: string;
  sources: { label: string; position: Vec3 }[];
  diffuse?: string[];
  splat_transform?: { scale: number; offset: Vec3 };
}

/** Everything the debugger knows about one memory. Missing files stay undefined. */
export interface MemoryBundle {
  id: string;
  files: FileEntry[];
  ingestMeta?: IngestMeta;
  cameras?: Camera[];
  points?: Float32Array; // xyz interleaved
  pointColors?: Uint8Array; // rgb interleaved
  stems?: StemsIndex;
  localized?: Localized;
  scene?: SoundscapeScene;
  splatMeta?: SplatMeta;
  splatMetrics?: SplatMetrics;
  trainReport?: TrainReport;
  truth?: Truth;
  errors: { key: string; message: string }[];
}
