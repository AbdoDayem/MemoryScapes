import { useState } from "react";

import type { MemoryBundle } from "../lib/types";
import { cssVar, fmt, fmtVec, METHOD_VAR, methodColor, useCanvas } from "../lib/ui";

const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

export function LocalizationView({ bundle }: { bundle: MemoryBundle }) {
  const [sel, setSel] = useState<string | null>(null);
  const loc = bundle.localized;
  if (!loc) return <p className="empty">No localize/sources.json yet.</p>;
  const truth = new Map(bundle.truth?.sources.map((t) => [t.label, t.position]) ?? []);

  return (
    <>
      <section className="panel">
        <div className="row">
          <h2>Top-down map</h2>
          <span className="muted">x to the right, z (forward) up. Circles show each estimate's spread.</span>
        </div>
        <TopDown bundle={bundle} selected={sel} />
        <div className="legend">
          {Object.entries(METHOD_VAR).map(([m, v]) => (
            <span key={m}>
              <i style={{ background: `var(${v})` }} />
              {m}
            </span>
          ))}
          {truth.size > 0 && (
            <span>
              <i style={{ background: "var(--truth)" }} />
              ground truth
            </span>
          )}
          <span>– – camera path</span>
        </div>
      </section>
      <section className="panel">
        <h2>Sources</h2>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>Sound</th>
                <th>Method</th>
                <th className="num">Confidence</th>
                <th>Position (m)</th>
                <th className="num">Spread (m)</th>
                <th className="num">dB @ 1 m</th>
                {truth.size > 0 && <th className="num">Error (m)</th>}
              </tr>
            </thead>
            <tbody>
              {loc.sources.map((s) => {
                const t = truth.get(s.label);
                return (
                  <tr key={s.id} className={`click ${sel === s.id ? "sel" : ""}`} onClick={() => setSel(sel === s.id ? null : s.id)}>
                    <td>{s.label}</td>
                    <td>
                      <span className="method">
                        <i style={{ background: `var(${METHOD_VAR[s.method]})` }} />
                        {s.method}
                      </span>
                    </td>
                    <td className="num">{fmt(s.confidence)}</td>
                    <td className="mono">{fmtVec(s.position)}</td>
                    <td className="num">{fmt(s.spread_m)}</td>
                    <td className="num">{fmt(s.level_db_at_1m, 1)}</td>
                    {truth.size > 0 && <td className="num">{t && s.position ? fmt(dist(s.position, t)) : "–"}</td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function TopDown({ bundle, selected }: { bundle: MemoryBundle; selected: string | null }) {
  const loc = bundle.localized!;
  const cams = bundle.cameras ?? [];
  const truth = bundle.truth?.sources ?? [];
  const height = 460;
  const ref = useCanvas(
    (ctx, w, h) => {
      const pts: number[][] = [...cams.map((c) => [c.c2w[0][3], c.c2w[2][3]])];
      for (const s of loc.sources) if (s.position) pts.push([s.position[0], s.position[2]]);
      for (const t of truth) pts.push([t.position[0], t.position[2]]);
      // frame what matters (cameras, sounds, truth), not the whole point cloud
      if (pts.length < 2) pts.push([loc.bounds.min[0], loc.bounds.min[2]], [loc.bounds.max[0], loc.bounds.max[2]]);
      const xs = pts.map((p) => p[0]), zs = pts.map((p) => p[1]);
      const x0 = Math.min(...xs) - 2.5, x1 = Math.max(...xs) + 2.5, z0 = Math.min(...zs) - 2.5, z1 = Math.max(...zs) + 2.5;
      const s = Math.min(w / (x1 - x0), h / (z1 - z0));
      const ox = (w - (x1 - x0) * s) / 2, oz = (h - (z1 - z0) * s) / 2;
      const X = (x: number) => ox + (x - x0) * s;
      const Z = (z: number) => h - oz - (z - z0) * s;

      // grid every metre (every 5 m bolder)
      ctx.lineWidth = 1;
      for (let gx = Math.ceil(x0); gx <= x1; gx++) {
        ctx.strokeStyle = cssVar(gx % 5 === 0 ? "--line" : "--soft");
        ctx.beginPath();
        ctx.moveTo(X(gx), 0);
        ctx.lineTo(X(gx), h);
        ctx.stroke();
      }
      for (let gz = Math.ceil(z0); gz <= z1; gz++) {
        ctx.strokeStyle = cssVar(gz % 5 === 0 ? "--line" : "--soft");
        ctx.beginPath();
        ctx.moveTo(0, Z(gz));
        ctx.lineTo(w, Z(gz));
        ctx.stroke();
      }

      // scene points
      if (bundle.points) {
        ctx.fillStyle = cssVar("--muted");
        ctx.globalAlpha = 0.25;
        const p = bundle.points;
        const step = Math.max(1, Math.floor(p.length / 3 / 6000)) * 3;
        for (let i = 0; i < p.length; i += step) ctx.fillRect(X(p[i]) - 0.75, Z(p[i + 2]) - 0.75, 1.5, 1.5);
        ctx.globalAlpha = 1;
      }

      // camera path and headings
      ctx.strokeStyle = cssVar("--ink");
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      cams.forEach((c, i) => (i ? ctx.lineTo(X(c.c2w[0][3]), Z(c.c2w[2][3])) : ctx.moveTo(X(c.c2w[0][3]), Z(c.c2w[2][3]))));
      ctx.stroke();
      ctx.setLineDash([]);
      cams.forEach((c, i) => {
        if (i % 3) return;
        const px = c.c2w[0][3], pz = c.c2w[2][3];
        ctx.beginPath();
        ctx.moveTo(X(px), Z(pz));
        ctx.lineTo(X(px + c.c2w[0][2] * 0.8), Z(pz + c.c2w[2][2] * 0.8));
        ctx.stroke();
      });

      ctx.font = "12px Inter, system-ui, sans-serif";

      // estimates
      for (const src of loc.sources) {
        if (!src.position) continue;
        const color = methodColor(src.method);
        const ex = X(src.position[0]), ez = Z(src.position[2]);
        const t = truth.find((tt) => tt.label === src.label);
        if (t) {
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.6;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(ex, ez);
          ctx.lineTo(X(t.position[0]), Z(t.position[2]));
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
        if (src.spread_m) {
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.12;
          ctx.beginPath();
          ctx.arc(ex, ez, Math.max(3, src.spread_m * s), 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
        const r = selected === src.id ? 9 : 6;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(ex, ez, r, 0, Math.PI * 2);
        ctx.fill();
        if (selected === src.id) {
          ctx.strokeStyle = cssVar("--ink");
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        ctx.fillStyle = cssVar("--ink");
        ctx.fillText(`${src.label} ${src.position[1].toFixed(1)} m up`, ex + 10, ez - 8);
      }

      // ground truth, on top so it stays visible under a close estimate
      ctx.strokeStyle = cssVar("--truth");
      ctx.lineWidth = 2;
      for (const t of truth) {
        const tx = X(t.position[0]), tz = Z(t.position[2]);
        ctx.beginPath();
        ctx.moveTo(tx - 6, tz - 6);
        ctx.lineTo(tx + 6, tz + 6);
        ctx.moveTo(tx - 6, tz + 6);
        ctx.lineTo(tx + 6, tz - 6);
        ctx.stroke();
      }

      // scale bar
      ctx.fillStyle = cssVar("--muted");
      ctx.fillRect(12, h - 16, s * 5, 2);
      ctx.fillText("5 m", 12, h - 22);
    },
    height,
    [bundle, selected],
  );
  return <canvas ref={ref} className="fill" />;
}

