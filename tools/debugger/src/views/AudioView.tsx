import { useEffect, useMemo, useRef, useState } from "react";

import { audioContext, loadAudio, relKey } from "../lib/api";
import { spectrogram } from "../lib/fft";
import type { MemoryBundle } from "../lib/types";
import { cssVar, DIFFUSE_CLASSES, fmt, methodColor, TAG_STRONG, TAG_THRESHOLD, useCanvas } from "../lib/ui";

interface Track {
  id: string;
  label: string;
  key: string; // relative to the memory
  meta: string;
  activity?: number[];
  method?: string;
}

function usePlayer() {
  const src = useRef<AudioBufferSourceNode | null>(null);
  const [playing, setPlaying] = useState<{ id: string; start: number; duration: number } | null>(null);
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setNow(audioContext().currentTime), 50);
    return () => clearInterval(t);
  }, [playing]);
  const stop = () => {
    src.current?.stop();
    src.current = null;
    setPlaying(null);
  };
  const play = async (id: string, buffer: AudioBuffer) => {
    stop();
    const ctx = audioContext();
    await ctx.resume();
    const node = ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(ctx.destination);
    node.onended = () => setPlaying((p) => (p?.id === id ? null : p));
    node.start();
    src.current = node;
    setPlaying({ id, start: ctx.currentTime, duration: buffer.duration });
  };
  useEffect(() => stop, []);
  const progress = (id: string) => (playing?.id === id ? Math.min(1, (now - playing.start) / playing.duration) : null);
  return { play, stop, playing, progress };
}

export function AudioView({ bundle }: { bundle: MemoryBundle }) {
  const stems = bundle.stems;
  const localized = new Map(bundle.localized?.sources.map((s) => [s.id, s]) ?? []);
  const tracks: Track[] = useMemo(() => {
    const out: Track[] = [];
    if (bundle.files.some((f) => f.key === "ingest/audio.wav")) {
      const mic = bundle.ingestMeta?.mic;
      out.push({ id: "recording", label: "Recording", key: "ingest/audio.wav", meta: mic ? `${mic.layout}, ${mic.channels} ch${mic.spacing_m ? `, ${mic.spacing_m} m` : ""}` : "" });
    }
    for (const s of stems?.stems ?? []) {
      const loc = localized.get(s.id);
      out.push({
        id: s.id,
        label: s.label,
        key: relKey(bundle.id, s.key),
        meta: `${fmt(s.level_dbfs, 1)} dBFS · tag ${fmt(s.tag_score)}${loc ? ` · ${loc.method} ${fmt(loc.confidence)}` : ""}`,
        activity: s.activity,
        method: loc?.method,
      });
    }
    if (stems) out.push({ id: "ambient", label: "Ambient bed", key: relKey(bundle.id, stems.ambient_key), meta: "everything not in a stem" });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bundle]);

  const [selected, setSelected] = useState<string>(tracks[0]?.id ?? "");
  const player = usePlayer();

  if (!tracks.length) return <p className="empty">No audio yet: ingest/audio.wav and separate/ are missing.</p>;
  const sel = tracks.find((t) => t.id === selected) ?? tracks[0];

  return (
    <>
      {stems && <TagsPanel bundle={bundle} />}
      <section className="panel">
        <div className="row">
          <h2>Tracks</h2>
          <span className="muted">Click a track to see its spectrogram. Stems + ambient add up to the recording.</span>
        </div>
        <div>
          {tracks.map((t) => (
            <TrackRow
              key={t.id}
              memoryId={bundle.id}
              track={t}
              windowTimes={stems?.window_times}
              selected={t.id === sel.id}
              progress={player.progress(t.id)}
              onSelect={() => setSelected(t.id)}
              onPlay={(buf) => (player.playing?.id === t.id ? player.stop() : player.play(t.id, buf))}
              isPlaying={player.playing?.id === t.id}
            />
          ))}
        </div>
      </section>
      <SpectrogramPanel memoryId={bundle.id} track={sel} />
      {!!stems?.dropped?.length && (
        <section className="panel">
          <h2>Dropped stems</h2>
          <table>
            <tbody>
              {stems.dropped.map((d) => (
                <tr key={d.label}>
                  <td>{d.label}</td>
                  <td className="muted">{d.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}

function TagsPanel({ bundle }: { bundle: MemoryBundle }) {
  const stems = bundle.stems!;
  const separated = new Set(stems.stems.map((s) => s.label));
  const dropped = new Set(stems.dropped?.map((d) => d.label));
  const tags = Object.entries(stems.tags).sort((a, b) => b[1] - a[1]);
  const status = (label: string, score: number) =>
    separated.has(label) ? "separated" : dropped.has(label) ? "dropped" : DIFFUSE_CLASSES.has(label) && score >= TAG_THRESHOLD ? "ambient (diffuse class)" : score >= TAG_THRESHOLD ? "not selected" : "below threshold";
  const color = (s: string) => (s === "separated" ? cssVar("--accent") : s === "dropped" ? cssVar("--fail") : s.startsWith("ambient") ? cssVar("--diffuse") : cssVar("--line"));
  const rowH = 18;
  const ref = useCanvas(
    (ctx, w) => {
      const labelW = 90, statusW = 150, barW = w - labelW - statusW - 10;
      ctx.font = "12px Inter, system-ui, sans-serif";
      tags.forEach(([label, score], i) => {
        const y = i * rowH;
        const st = status(label, score);
        ctx.fillStyle = cssVar("--ink");
        ctx.fillText(label, 0, y + 13);
        ctx.fillStyle = color(st);
        ctx.fillRect(labelW, y + 4, Math.max(1, score * barW), rowH - 8);
        ctx.fillStyle = cssVar("--muted");
        ctx.fillText(`${score.toFixed(2)}  ${st}`, labelW + barW + 10, y + 13);
      });
      for (const [v, dash] of [[TAG_THRESHOLD, [3, 3]], [TAG_STRONG, [6, 3]]] as const) {
        ctx.strokeStyle = cssVar("--muted");
        ctx.setLineDash(dash as unknown as number[]);
        ctx.beginPath();
        ctx.moveTo(labelW + v * barW, 0);
        ctx.lineTo(labelW + v * barW, tags.length * rowH);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    },
    Math.max(rowH, tags.length * rowH),
    [bundle],
  );
  return (
    <section className="panel">
      <div className="row">
        <h2>Tags</h2>
        <span className="muted">
          Peak window score per class. Dashed lines: threshold {TAG_THRESHOLD} (needs two windows) and strong {TAG_STRONG} (one window is enough).
        </span>
      </div>
      <canvas ref={ref} className="fill" />
    </section>
  );
}

function TrackRow(props: {
  memoryId: string;
  track: Track;
  windowTimes?: number[];
  selected: boolean;
  progress: number | null;
  isPlaying: boolean;
  onSelect: () => void;
  onPlay: (b: AudioBuffer) => void;
}) {
  const { memoryId, track, windowTimes, selected, progress, isPlaying, onSelect, onPlay } = props;
  const [buffer, setBuffer] = useState<AudioBuffer>();
  const [error, setError] = useState<string>();
  useEffect(() => {
    loadAudio(memoryId, track.key).then(setBuffer).catch((e) => setError(String(e)));
  }, [memoryId, track.key]);

  const ref = useCanvas(
    (ctx, w, h) => {
      if (!buffer) return;
      const chans = Math.min(2, buffer.numberOfChannels);
      const accent = track.method ? methodColor(track.method) : cssVar("--accent");
      for (let c = 0; c < chans; c++) {
        const data = buffer.getChannelData(c);
        const step = Math.max(1, Math.floor(data.length / w));
        ctx.fillStyle = c === 0 ? accent : cssVar("--muted");
        ctx.globalAlpha = c === 0 ? 0.85 : 0.45;
        const mid = h / 2;
        for (let x = 0; x < w; x++) {
          let lo = 0, hi = 0;
          for (let i = x * step; i < Math.min(data.length, (x + 1) * step); i++) {
            const v = data[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
          ctx.fillRect(x, mid - hi * mid * 1.6, 1, Math.max(1, (hi - lo) * mid * 1.6));
        }
      }
      ctx.globalAlpha = 1;
      if (track.activity && windowTimes?.length) {
        ctx.strokeStyle = cssVar("--ink");
        ctx.lineWidth = 1;
        ctx.beginPath();
        track.activity.forEach((a, i) => {
          const x = (windowTimes[i] / buffer.duration) * w;
          const y = h - 2 - a * (h - 4);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
      if (progress !== null) {
        ctx.fillStyle = cssVar("--fail");
        ctx.fillRect(progress * w, 0, 2, h);
      }
    },
    44,
    [buffer, progress, track],
  );

  return (
    <div className={`track ${selected ? "sel" : ""}`} onClick={onSelect}>
      <button
        className={`btn ${isPlaying ? "primary" : ""}`}
        aria-label={isPlaying ? `Stop ${track.label}` : `Play ${track.label}`}
        disabled={!buffer}
        onClick={(e) => {
          e.stopPropagation();
          if (buffer) onPlay(buffer);
        }}
      >
        {isPlaying ? "■" : "▶"}
      </button>
      <div>
        <div className="label">{track.label}</div>
        <div className="meta">{error ?? track.meta}</div>
      </div>
      <canvas ref={ref} className="fill" />
    </div>
  );
}

const STOPS = [
  [13, 8, 135],
  [126, 3, 168],
  [204, 71, 120],
  [248, 149, 64],
  [240, 249, 33],
];
function colormap(t: number): [number, number, number] {
  const x = Math.min(0.9999, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.floor(x), f = x - i;
  return [0, 1, 2].map((k) => STOPS[i][k] + (STOPS[i + 1][k] - STOPS[i][k]) * f) as [number, number, number];
}

function SpectrogramPanel({ memoryId, track }: { memoryId: string; track: Track }) {
  const [buffer, setBuffer] = useState<AudioBuffer>();
  useEffect(() => {
    setBuffer(undefined);
    loadAudio(memoryId, track.key).then(setBuffer).catch(() => undefined);
  }, [memoryId, track.key]);
  const spec = useMemo(() => {
    if (!buffer) return null;
    const ch0 = buffer.getChannelData(0);
    let mono = ch0;
    if (buffer.numberOfChannels > 1) {
      const ch1 = buffer.getChannelData(1);
      mono = ch0.map((v, i) => (v + ch1[i]) / 2);
    }
    return spectrogram(mono, buffer.sampleRate, 512, 256);
  }, [buffer]);
  const height = 200;
  const ref = useCanvas(
    (ctx, w, h) => {
      if (!spec) return;
      let max = -Infinity;
      for (const v of spec.db) if (v > max) max = v;
      const floor = max - 80;
      const img = ctx.createImageData(spec.frames, spec.bins);
      for (let f = 0; f < spec.frames; f++) {
        for (let b = 0; b < spec.bins; b++) {
          const [r, g, bl] = colormap((spec.db[f * spec.bins + b] - floor) / 80);
          const o = ((spec.bins - 1 - b) * spec.frames + f) * 4;
          img.data[o] = r;
          img.data[o + 1] = g;
          img.data[o + 2] = bl;
          img.data[o + 3] = 255;
        }
      }
      const off = document.createElement("canvas");
      off.width = spec.frames;
      off.height = spec.bins;
      off.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, w, h);
    },
    height,
    [spec],
  );
  return (
    <section className="panel">
      <div className="row">
        <h2>Spectrogram: {track.label}</h2>
        {buffer && (
          <span className="muted">
            0–{(buffer.sampleRate / 2000).toFixed(0)} kHz, {buffer.duration.toFixed(1)} s, 80 dB range
          </span>
        )}
      </div>
      <canvas ref={ref} className="fill" />
    </section>
  );
}
