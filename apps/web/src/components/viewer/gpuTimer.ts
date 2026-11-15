/**
 * GPU time per frame (EXT_disjoint_timer_query_webgl2), for the viewer's `?perf` readout.
 *
 * The frame rate stops at the display's refresh (or 30 fps in a browser's battery saver),
 * so it can't show how much headroom a device has; GPU milliseconds can. Results arrive a
 * few frames late, so queries are polled. Returns null where the extension is missing.
 */

interface TimerExt {
  TIME_ELAPSED_EXT: number;
  GPU_DISJOINT_EXT: number;
}

export interface GpuTimer {
  begin(): void;
  end(): void;
  /** Median GPU ms of the frames measured since the last call (NaN if none). */
  take(): number;
}

export function createGpuTimer(gl: WebGL2RenderingContext): GpuTimer | null {
  const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2") as TimerExt | null;
  if (!ext) return null;
  const pending: WebGLQuery[] = [];
  let open: WebGLQuery | null = null;
  let samples: number[] = [];

  const poll = () => {
    while (pending.length && gl.getQueryParameter(pending[0], gl.QUERY_RESULT_AVAILABLE)) {
      const q = pending.shift()!;
      // A disjoint event (power state change, context switch) makes results meaningless.
      if (!gl.getParameter(ext.GPU_DISJOINT_EXT)) samples.push(gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6);
      gl.deleteQuery(q);
    }
  };

  return {
    begin() {
      poll();
      if (open || pending.length >= 8) return; // results lagging: skip this frame
      open = gl.createQuery();
      if (open) gl.beginQuery(ext.TIME_ELAPSED_EXT, open);
    },
    end() {
      if (!open) return;
      gl.endQuery(ext.TIME_ELAPSED_EXT);
      pending.push(open);
      open = null;
    },
    take() {
      poll();
      const s = samples.sort((a, b) => a - b);
      samples = [];
      return s.length ? s[s.length >> 1] : NaN;
    },
  };
}
