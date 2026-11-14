import type { Check } from "../lib/checks";

export function ChecksView({ checks, splatLoading, splatError }: { checks: Check[]; splatLoading: boolean; splatError?: string }) {
  const by = (l: Check["level"]) => checks.filter((c) => c.level === l).length;
  return (
    <section className="panel">
      <div className="row">
        <h2>Integration checks</h2>
        <span className="badge fail">{by("fail")}</span>
        <span className="badge warn">{by("warn")}</span>
        <span className="badge info">{by("info")}</span>
        <span className="badge ok">{by("ok")}</span>
        {splatLoading && <span className="muted">Loading the splat for the alignment check…</span>}
        {splatError && <span className="muted">Splat didn't load: {splatError}</span>}
      </div>
      <div className="checks">
        {checks.map((c) => (
          <div key={c.id} className="check">
            <span className={`lvl ${c.level}`}>{c.level}</span>
            <span className="area">{c.area}</span>
            <div>
              <div>{c.title}</div>
              {c.detail && <div className="detail">{c.detail}</div>}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
