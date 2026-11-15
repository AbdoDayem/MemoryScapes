import { useEffect, useMemo, useState } from "react";

import { dataUrl, relKey } from "../lib/api";
import { compareClouds, percentile } from "../lib/checks";
import { exposureSeries, reportRows } from "../lib/report";
import type { GaussianView } from "../lib/ply";
import type { MemoryBundle } from "../lib/types";
import { bytes, cssVar, fmt, useCanvas } from "../lib/ui";

export function SplatView({ bundle, splat, splatError }: { bundle: MemoryBundle; splat?: GaussianView; splatError?: string }) {
  const meta = bundle.splatMeta;
  const lods = bundle.files.filter((f) => f.key.startsWith("splat/lods/"));
  const hasAny = bundle.files.some((f) => f.key.startsWith("splat/"));
  if (!hasAny) return <p className="empty">The splat stage hasn't written anything yet (see docs/SPLAT_OUTPUTS.md).</p>;
  return (
    <>
      <div className="grid2">
        <section className="panel">
          <h2>Trainer</h2>
          {meta ? (
            <table>
              <tbody>
                {Object.entries(meta).map(([k, v]) => (
                  <tr key={k}>
                    <td className="muted">{k}</td>
                    <td className="mono">{Array.isArray(v) ? `${v.length} frames: ${v.join(", ")}` : String(v)}</td>
                  </tr>
                ))}
                {lods.map((f) => (
                  <tr key={f.key}>
                    <td className="muted">{f.key.replace("splat/", "")}</td>
                    <td className="mono">{bytes(f.size)} (not parsed)</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">No splat/meta.json.</p>
          )}
        </section>
        <GaussianStats bundle={bundle} splat={splat} error={splatError} />
      </div>
      {bundle.trainReport && <Report bundle={bundle} />}
      {bundle.splatMetrics && <Metrics bundle={bundle} />}
      {bundle.cameras && <FrameCompare bundle={bundle} />}
    </>
  );
}

function GaussianStats({ bundle, splat, error }: { bundle: MemoryBundle; splat?: GaussianView; error?: string }) {
  const stats = useMemo(() => {
    if (!splat) return null;
    const p = splat.positions;
    const ax = (k: number) => Float32Array.from({ length: splat.count }, (_, i) => p[i * 3 + k]);
    const align = bundle.points?.length ? compareClouds(bundle.points, p) : null;
    return {
      lo: [0, 1, 2].map((k) => percentile(ax(k), 1)),
      hi: [0, 1, 2].map((k) => percentile(ax(k), 99)),
      align,
      transparent: splat.opacity.filter((o) => o < 0.05).length / splat.count,
    };
  }, [bundle, splat]);
  return (
    <section className="panel">
      <h2>Gaussians</h2>
      {error && <p className="muted">{error}</p>}
      {!splat && !error && <p className="muted">Loading point_cloud.ply…</p>}
      {splat && stats && (
        <>
          <table>
            <tbody>
              <tr>
                <td className="muted">count</td>
                <td className="mono">{splat.count.toLocaleString()}</td>
              </tr>
              <tr>
                <td className="muted">bounds (1–99%)</td>
                <td className="mono">
                  ({stats.lo.map((v) => v.toFixed(1)).join(", ")}) → ({stats.hi.map((v) => v.toFixed(1)).join(", ")})
                </td>
              </tr>
              <tr>
                <td className="muted">near-transparent</td>
                <td className="mono">{(stats.transparent * 100).toFixed(1)}% with opacity &lt; 0.05</td>
              </tr>
              {stats.align && (
                <tr>
                  <td className="muted">vs. poses</td>
                  <td className="mono">
                    gap {fmt(stats.align.medianGapM)} m · scale {fmt(stats.align.scaleRatio)}× · offset ({stats.align.offset.map((v) => v.toFixed(1)).join(", ")})
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <div className="grid2">
            <Histogram title="opacity" values={splat.opacity} />
            <Histogram title="log10 scale (m)" values={splat.scale.map((s) => Math.log10(Math.max(s, 1e-6)))} />
          </div>
        </>
      )}
    </section>
  );
}

function Histogram({ title, values, bins = 30 }: { title: string; values: Float32Array; bins?: number }) {
  const ref = useCanvas(
    (ctx, w, h) => {
      const lo = percentile(values, 0.5), hi = percentile(values, 99.5);
      const counts = new Array(bins).fill(0);
      for (const v of values) counts[Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo || 1)) * bins)))]++;
      const max = Math.max(...counts);
      ctx.fillStyle = cssVar("--accent");
      counts.forEach((c, i) => ctx.fillRect((i * w) / bins + 1, h - 14 - (c / max) * (h - 18), w / bins - 2, (c / max) * (h - 18)));
      ctx.fillStyle = cssVar("--muted");
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.fillText(lo.toFixed(2), 0, h - 2);
      const hiText = hi.toFixed(2);
      ctx.fillText(hiText, w - ctx.measureText(hiText).width, h - 2);
    },
    90,
    [values],
  );
  return (
    <div>
      <h3>{title}</h3>
      <canvas ref={ref} className="fill" />
    </div>
  );
}

/** splat/train_report.json: what to compare between runs (docs/SPATIAL_VIDEO.md A/B). */
function Report({ bundle }: { bundle: MemoryBundle }) {
  const r = bundle.trainReport!;
  const exposure = exposureSeries(r);
  return (
    <section className="panel">
      <div className="row">
        <h2>Training report</h2>
        <span className="muted">
          {r.psnr_split === "holdout" ? "scored on held-out frames" : "scored on training frames (set splat.train.holdout_every for an honest score)"}
        </span>
      </div>
      <div className="grid2">
        <table>
          <tbody>
            {reportRows(r).map(([k, v]) => (
              <tr key={k}>
                <td className="muted">{k}</td>
                <td className="mono">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {exposure && (
          <LineChart
            title={`exposure gain per training frame (${fmt(Math.min(...exposure.gains))}–${fmt(Math.max(...exposure.gains))})`}
            xs={exposure.gains.map((_, i) => i)}
            ys={exposure.gains}
          />
        )}
      </div>
    </section>
  );
}

function Metrics({ bundle }: { bundle: MemoryBundle }) {
  const m = bundle.splatMetrics!;
  const x = m.iteration ?? [];
  const series = Object.entries(m).filter(([k, v]) => k !== "iteration" && k !== "elapsed_s" && v.length === x.length);
  return (
    <section className="panel">
      <div className="row">
        <h2>Training</h2>
        {m.elapsed_s && <span className="muted">{fmt(m.elapsed_s[m.elapsed_s.length - 1], 0)} s for {x[x.length - 1]?.toLocaleString()} iterations</span>}
      </div>
      <div className="grid2">
        {series.map(([name, ys]) => (
          <LineChart key={name} title={name} xs={x} ys={ys} />
        ))}
      </div>
    </section>
  );
}

function LineChart({ title, xs, ys }: { title: string; xs: number[]; ys: number[] }) {
  const ref = useCanvas(
    (ctx, w, h) => {
      const pad = { l: 46, r: 8, t: 6, b: 18 };
      const finite = ys.filter(Number.isFinite);
      const lo = Math.min(...finite), hi = Math.max(...finite);
      const xMax = xs[xs.length - 1] || 1;
      const X = (v: number) => pad.l + (v / xMax) * (w - pad.l - pad.r);
      const Y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (h - pad.t - pad.b);
      ctx.strokeStyle = cssVar("--soft");
      ctx.fillStyle = cssVar("--muted");
      ctx.font = "11px Inter, system-ui, sans-serif";
      for (const v of [lo, (lo + hi) / 2, hi]) {
        ctx.beginPath();
        ctx.moveTo(pad.l, Y(v));
        ctx.lineTo(w - pad.r, Y(v));
        ctx.stroke();
        const label = Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(Math.abs(v) < 1 ? 3 : 1);
        ctx.fillText(label, 2, Y(v) + 4);
      }
      ctx.fillText("0", pad.l, h - 4);
      const end = xMax.toLocaleString();
      ctx.fillText(end, w - pad.r - ctx.measureText(end).width, h - 4);
      ctx.strokeStyle = cssVar("--accent");
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ys.forEach((v, i) => (i ? ctx.lineTo(X(xs[i]), Y(v)) : ctx.moveTo(X(xs[i]), Y(v))));
      ctx.stroke();
      const last = ys[ys.length - 1];
      ctx.fillStyle = cssVar("--accent");
      ctx.beginPath();
      ctx.arc(X(xs[xs.length - 1]), Y(last), 3, 0, Math.PI * 2);
      ctx.fill();
    },
    150,
    [xs, ys],
  );
  return (
    <div>
      <h3>
        {title} <span className="mono">{Number.isInteger(ys[ys.length - 1]) ? ys[ys.length - 1].toLocaleString() : fmt(ys[ys.length - 1], 3)}</span>
      </h3>
      <canvas ref={ref} className="fill" />
    </div>
  );
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`couldn't load ${src}`));
    img.src = src;
  });
}

function FrameCompare({ bundle }: { bundle: MemoryBundle }) {
  const cams = bundle.cameras!;
  const [i, setI] = useState(0);
  const cam = cams[Math.min(i, cams.length - 1)];
  const keys = new Set(bundle.files.map((f) => f.key));
  const frameKey = cam.image_key ? relKey(bundle.id, cam.image_key) : null;
  const renderKey = `splat/renders/${cam.frame_id}.png`;
  const isImage = frameKey && /\.(png|jpe?g|webp)$/i.test(frameKey) && keys.has(frameKey);
  const hasRender = keys.has(renderKey);
  const evalFrames = new Set(bundle.splatMeta?.eval_frames ?? []);
  const [diff, setDiff] = useState<{ url: string; psnr: number } | null>(null);

  useEffect(() => {
    setDiff(null);
    if (!isImage || !hasRender) return;
    let live = true;
    Promise.all([loadImage(dataUrl(bundle.id, frameKey!)), loadImage(dataUrl(bundle.id, renderKey))]).then(([a, b]) => {
      if (!live) return;
      const w = b.naturalWidth, h = b.naturalHeight;
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(a, 0, 0, w, h);
      const da = ctx.getImageData(0, 0, w, h);
      ctx.drawImage(b, 0, 0, w, h);
      const db = ctx.getImageData(0, 0, w, h);
      const out = ctx.createImageData(w, h);
      let se = 0;
      for (let p = 0; p < da.data.length; p += 4) {
        let e = 0;
        for (let k = 0; k < 3; k++) {
          const d = da.data[p + k] - db.data[p + k];
          se += d * d;
          e += Math.abs(d);
        }
        const v = Math.min(255, e * 1.5);
        out.data[p] = v;
        out.data[p + 1] = v * 0.4;
        out.data[p + 2] = 255 - v;
        out.data[p + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
      const mse = se / ((da.data.length / 4) * 3);
      setDiff({ url: c.toDataURL(), psnr: 10 * Math.log10((255 * 255) / Math.max(mse, 1e-6)) });
    });
    return () => {
      live = false;
    };
  }, [bundle.id, frameKey, renderKey, isImage, hasRender]);

  return (
    <section className="panel">
      <div className="row">
        <h2>Keyframe vs. splat render</h2>
        <label className="row">
          <input type="range" min={0} max={cams.length - 1} value={i} onChange={(e) => setI(Number(e.target.value))} aria-label="Keyframe" />
          <span className="mono">
            {cam.frame_id} · t={cam.t.toFixed(2)} s{evalFrames.has(cam.frame_id) ? " · held-out eval frame" : ""}
          </span>
        </label>
        {diff && <span className="mono">PSNR {diff.psnr.toFixed(1)} dB</span>}
      </div>
      <div className="compare">
        <figure>
          {isImage ? <img src={dataUrl(bundle.id, frameKey!)} alt={`Keyframe ${cam.frame_id}`} /> : <div className="empty">no image for this frame</div>}
          <figcaption>keyframe ({frameKey ?? "no image_key"})</figcaption>
        </figure>
        <figure>
          {hasRender ? <img src={dataUrl(bundle.id, renderKey)} alt={`Splat render ${cam.frame_id}`} /> : <div className="empty">no render</div>}
          <figcaption>splat render</figcaption>
        </figure>
        <figure>
          {diff ? <img src={diff.url} alt="Difference" /> : <div className="empty">{isImage && hasRender ? "computing…" : "needs both images"}</div>}
          <figcaption>difference (bright = mismatch)</figcaption>
        </figure>
      </div>
    </section>
  );
}
