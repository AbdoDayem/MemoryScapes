import { dataUrl } from "../lib/api";
import type { MemoryBundle } from "../lib/types";
import { bytes } from "../lib/ui";

export function FilesView({ bundle }: { bundle: MemoryBundle }) {
  const total = bundle.files.reduce((a, f) => a + f.size, 0);
  return (
    <section className="panel">
      <div className="row">
        <h2>Files</h2>
        <span className="muted">
          {bundle.files.length} files, {bytes(total)}
        </span>
      </div>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Key</th>
              <th className="num">Size</th>
            </tr>
          </thead>
          <tbody>
            {bundle.files.map((f) => (
              <tr key={f.key}>
                <td className="mono">
                  <a href={dataUrl(bundle.id, f.key)} target="_blank" rel="noreferrer">
                    {f.key}
                  </a>
                </td>
                <td className="num">{bytes(f.size)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
