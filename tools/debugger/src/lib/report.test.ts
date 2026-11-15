import { describe, expect, it } from "vitest";

import { exposureSeries, reportRows } from "./report";

describe("train report", () => {
  it("lists scores first and leaves lists and objects out", () => {
    const rows = reportRows({
      unbounded: false, seconds: 12.34, psnr_cc: 24.5, psnr: 23.123, psnr_split: "holdout",
      depth_error_vs_sfm: null, image_size: [640, 360], exposure_gains: { a: 1 }, gpu: "cuda",
    });
    expect(rows.map(([k]) => k)).toEqual(["psnr", "psnr_split", "psnr_cc", "depth_error_vs_sfm", "seconds", "gpu", "unbounded"]);
    expect(Object.fromEntries(rows)).toMatchObject({ psnr: "23.12", depth_error_vs_sfm: "–", unbounded: "false" });
  });

  it("orders exposure gains by frame number", () => {
    const s = exposureSeries({ exposure_gains: { frame_10: 1.1, frame_2: 0.9, frame_1_right: 1.03, frame_1: 1.0 } });
    expect(s).toEqual({ frames: ["frame_1", "frame_1_right", "frame_2", "frame_10"], gains: [1.0, 1.03, 0.9, 1.1] });
    expect(exposureSeries({ exposure_gains: null })).toBeNull();
    expect(exposureSeries({})).toBeNull();
  });
});
