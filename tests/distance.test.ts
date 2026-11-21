import { describe, it, expect } from 'vitest';
import {
  CameraCalibration,
  computeBboxIoU,
  computeCameraCalibrationFromDetection,
  computeDistanceErrorStats,
  estimateMonocularDistance,
  REAL_HEIGHT_M,
  SpatialDetection,
  SpatialObjectTracker,
} from '../src/engine/aiObjectDetector';

function makeDet(
  labelEn: string,
  bbox: [number, number, number, number],
  score: number
): SpatialDetection {
  return {
    labelEn,
    labelZh: labelEn,
    score,
    bbox,
    direction: '正前方',
    clockFace: '12点方向',
    distance: '近距离',
    distanceMeters: 1.5,
    distanceText: '较近（未标定粗估）',
    distanceCalibrated: false,
    priority: 'normal',
    source: 'tfjs',
    detectedAt: 1000,
  };
}

describe('Monocular distance estimation & P0-1 SpatialObjectTracker', () => {
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
    expect(estimateMonocularDistance('person', [100, 100, 6, 10], W, H, cal).invalidReason).toBe(
      'bbox_too_small'
    );
    expect(estimateMonocularDistance('person', [100, 0, 180, 480], W, H, cal).invalidReason).toBe(
      'clipped_vertical'
    );
    expect(estimateMonocularDistance('zebra', [100, 100, 120, 120], W, H, cal).invalidReason).toBe(
      'unknown_class_height'
    );
  });

  it('evaluates MAE & P90 across 0.5, 1, 2, 3, 5, 8, 10m synthetic reference distances', () => {
    const refDistances = [0.5, 1, 2, 3, 5, 8, 10];
    const samples = refDistances.map((d) => {
      const idealH = (REAL_HEIGHT_M.person * cal.focalLengthPx) / d;
      const clampedH = Math.min(H - 12, idealH * 1.08);
      const y = Math.max(6, (H - clampedH) / 2);
      const est = estimateMonocularDistance('person', [220, y, 100, clampedH], W, H, cal);
      return { referenceMeters: d, predictedMeters: est.estimatedMeters };
    });

    const report = computeDistanceErrorStats(samples);
    expect(report.total).toBe(7);
    expect(report.validCount).toBeGreaterThan(0);
    expect(report.maeMeters).not.toBeNull();
    expect(report.p90ErrorMeters).not.toBeNull();
    expect(report.byBand.length).toBe(3);
  });

  it('P0-1: NEVER confirms two same-class objects at different positions (left chair vs right chair)', () => {
    const tracker = new SpatialObjectTracker();
    const leftChair = makeDet('chair', [20, 150, 100, 140], 0.55);
    const rightChair = makeDet('chair', [500, 150, 100, 140], 0.56);

    expect(computeBboxIoU(leftChair.bbox, rightChair.bbox)).toBe(0);
    // Frame 1: left chair (medium confidence) held for spatial confirmation
    expect(tracker.update([leftChair], W, H)).toEqual([]);
    // Frame 2: completely different right chair MUST NOT be confirmed by left chair's class!
    expect(tracker.update([rightChair], W, H)).toEqual([]);
    expect(tracker.getActiveTrackCount()).toBe(2);
  });

  it('P0-1: confirms same target with slight jitter, tracks multiple same-class objects independently, and expires disappeared targets', () => {
    const tracker = new SpatialObjectTracker();
    const chairA1 = makeDet('chair', [40, 150, 100, 140], 0.55);
    const chairB1 = makeDet('chair', [460, 150, 100, 140], 0.54);
    const chairA2 = makeDet('chair', [48, 154, 100, 140], 0.57);
    const chairB2 = makeDet('chair', [452, 148, 100, 140], 0.56);

    expect(tracker.update([chairA1, chairB1], W, H)).toEqual([]);
    const f2 = tracker.update([chairA2, chairB2], W, H);
    expect(f2.length).toBe(2);

    // High-confidence fast mover confirms immediately without lag
    const fastCar = makeDet('car', [240, 180, 140, 90], 0.82);
    const f3 = tracker.update([fastCar], W, H);
    expect(f3.map((d) => d.labelEn)).toEqual(['car']);

    // Two empty frames prune disappeared tracks completely
    tracker.update([], W, H);
    tracker.update([], W, H);
    expect(tracker.getActiveTrackCount()).toBe(0);

    // Reset clears all tracks immediately
    tracker.update([chairA1], W, H);
    tracker.reset();
    expect(tracker.getActiveTrackCount()).toBe(0);
  });

  it('P0-1 (V5): SpatialObjectTracker NEVER confirms a medium-confidence target from duplicate frameId — requires 2 distinct frames', () => {
    const tracker = new SpatialObjectTracker();
    const chairF1 = makeDet('chair', [120, 160, 100, 140], 0.55);

    // Frame 1 inferred once -> unconfirmed (hits = 1)
    expect(tracker.update([chairF1], W, H, 1)).toEqual([]);
    // Same Frame 1 accidentally re-inferred -> MUST NOT increment hits or confirm!
    expect(tracker.update([chairF1], W, H, 1)).toEqual([]);
    // Even out-of-order older frameId -> MUST NOT increment hits
    expect(tracker.update([chairF1], W, H, 0)).toEqual([]);

    // Brand-new Frame 2 with spatially matched chair -> NOW confirmed (2 distinct frames)!
    const chairF2 = makeDet('chair', [124, 162, 100, 140], 0.56);
    const confirmed = tracker.update([chairF2], W, H, 2);
    expect(confirmed.length).toBe(1);
    expect(confirmed[0].labelEn).toBe('chair');
  });

  it('computes pinhole camera calibration from a 2.0m reference object and rejects clipped/invalid boxes', () => {
    // Chair (0.85m) at 2.0m with focalLengthPx=500 -> h = 0.85 * 500 / 2.0 = 212.5px
    const computed = computeCameraCalibrationFromDetection('chair', [200, 120, 120, 212.5], 2.0, 480);
    expect(computed).not.toBeNull();
    expect(computed!.calibrated).toBe(true);
    expect(computed!.focalLengthPx).toBe(500);

    // Using computed calibration yields calibrated conservative interval around 2.0m
    const est = estimateMonocularDistance('chair', [200, 120, 120, 212.5], 640, 480, computed);
    expect(est.calibrated).toBe(true);
    expect(est.estimatedMeters).toBeCloseTo(2.0, 1);
    expect(est.displayText).toMatch(/约\d+(\.\d+)?–\d+(\.\d+)?米/);

    // Clipped box touching top/bottom edge is rejected for calibration
    expect(computeCameraCalibrationFromDetection('chair', [200, 0, 120, 212], 2.0, 480)).toBeNull();
    // Out-of-range distance or unknown class rejected
    expect(computeCameraCalibrationFromDetection('zebra', [200, 120, 120, 212], 2.0, 480)).toBeNull();
    expect(computeCameraCalibrationFromDetection('chair', [200, 120, 120, 212], 0.1, 480)).toBeNull();
  });
});
