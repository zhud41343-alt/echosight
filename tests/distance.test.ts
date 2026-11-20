import { describe, it, expect } from 'vitest';
import {
  CameraCalibration,
  computeDistanceErrorStats,
  estimateMonocularDistance,
  REAL_HEIGHT_M,
} from '../src/engine/aiObjectDetector';

describe('Monocular distance estimation & MAE/P90 benchmark harness', () => {
  const W = 640;
  const H = 480;
  const cal: CameraCalibration = {
    calibrated: true,
    focalLengthPx: 500,
    referenceHeightPx: 480,
    deviceLabel: 'Synthetic Reference Camera',
  };

  it('does not output decimal meter claims when camera is uncalibrated', () => {
    const est = estimateMonocularDistance('chair', [200, 140, 120, 170], W, H, null);
    expect(est.valid).toBe(true);
    expect(est.calibrated).toBe(false);
    expect(est.displayText).toContain('未标定粗估');
    expect(est.displayText).not.toMatch(/约\d+\.\d+米/);
  });

  it('rejects clipped, tiny, and unknown-height bounding boxes as unknown', () => {
    const tiny = estimateMonocularDistance('person', [100, 100, 6, 10], W, H, cal);
    expect(tiny.valid).toBe(false);
    expect(tiny.invalidReason).toBe('bbox_too_small');

    const clipped = estimateMonocularDistance('person', [100, 0, 180, 480], W, H, cal);
    expect(clipped.valid).toBe(false);
    expect(clipped.invalidReason).toBe('clipped_vertical');

    const unknownCls = estimateMonocularDistance('zebra', [100, 100, 120, 120], W, H, cal);
    expect(unknownCls.valid).toBe(false);
    expect(unknownCls.invalidReason).toBe('unknown_class_height');
  });

  it('evaluates MAE & P90 across 0.5, 1, 2, 3, 5, 8, 10m synthetic reference distances', () => {
    const refDistances = [0.5, 1, 2, 3, 5, 8, 10];
    const samples = refDistances.map((d) => {
      // Simulate a person (1.68m) with +8% bbox height perturbation at distance d
      const idealH = (REAL_HEIGHT_M.person * cal.focalLengthPx) / d;
      const clampedH = Math.min(H - 12, idealH * 1.08);
      const y = Math.max(6, (H - clampedH) / 2);
      const est = estimateMonocularDistance('person', [220, y, 100, clampedH], W, H, cal);
      return {
        referenceMeters: d,
        predictedMeters: est.estimatedMeters,
      };
    });

    const report = computeDistanceErrorStats(samples);
    expect(report.total).toBe(7);
    expect(report.validCount).toBeGreaterThan(0);
    expect(report.maeMeters).not.toBeNull();
    expect(report.p90ErrorMeters).not.toBeNull();
    expect(report.byBand.length).toBe(3);
  });
});
