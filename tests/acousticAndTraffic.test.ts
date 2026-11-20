import { describe, it, expect } from 'vitest';
import { evaluateAcousticDataset } from '../src/engine/acousticThreatDetector';
import {
  classifyTrafficLightPixels,
  TrafficLightStabilizer,
} from '../src/engine/blindAssistTools';

function makePatch(r: number, g: number, b: number, pixels = 400): Uint8ClampedArray {
  const arr = new Uint8ClampedArray(pixels * 4);
  for (let i = 0; i < arr.length; i += 4) {
    arr[i] = r;
    arr[i + 1] = g;
    arr[i + 2] = b;
    arr[i + 3] = 255;
  }
  return arr;
}

describe('Experimental acoustic anomaly detector & traffic light pixel color classifier', () => {
  it('computes FPR/FNR offline on horn, speech, wind, and mechanical noise samples', () => {
    const summary = evaluateAcousticDataset([
      {
        id: 'horn_blast',
        expectedAnomaly: true,
        frame: { avgTotal: 55, avgLow: 40, midHighPeak: 225, baseline: 28 },
      },
      {
        id: 'normal_speech',
        expectedAnomaly: false,
        frame: { avgTotal: 38, avgLow: 30, midHighPeak: 110, baseline: 32 },
      },
      {
        id: 'gentle_wind',
        expectedAnomaly: false,
        frame: { avgTotal: 45, avgLow: 85, midHighPeak: 60, baseline: 35 },
      },
      {
        id: 'sudden_engine_surge',
        expectedAnomaly: true,
        frame: { avgTotal: 90, avgLow: 165, midHighPeak: 120, baseline: 30 },
      },
    ]);
    expect(summary.total).toBe(4);
    expect(summary.falsePositiveRate).toBe(0);
    expect(summary.falseNegativeRate).toBe(0);
  });

  it('rejects overexposed glare & mixed multi-lamp interference and never grants crossing clearance', () => {
    const glare = classifyTrafficLightPixels(makePatch(250, 250, 248));
    expect(glare.color).toBe('未知');
    expect(glare.isCrossingClearance).toBe(false);

    const dark = classifyTrafficLightPixels(makePatch(20, 22, 20));
    expect(dark.color).toBe('未知');

    // Mixed red + green interference (50/50 split) should return 未知
    const mixed = new Uint8ClampedArray(400 * 4);
    for (let i = 0; i < 200; i++) {
      mixed[i * 4] = 230;
      mixed[i * 4 + 1] = 20;
      mixed[i * 4 + 2] = 20;
      mixed[i * 4 + 3] = 255;
    }
    for (let i = 200; i < 400; i++) {
      mixed[i * 4] = 20;
      mixed[i * 4 + 1] = 220;
      mixed[i * 4 + 2] = 40;
      mixed[i * 4 + 3] = 255;
    }
    expect(classifyTrafficLightPixels(mixed).color).toBe('未知');
  });

  it('TrafficLightStabilizer requires 3 consecutive agreeing frames', () => {
    const stab = new TrafficLightStabilizer(3);
    expect(stab.push('绿色像素')).toBe('未知');
    expect(stab.push('绿色像素')).toBe('未知');
    expect(stab.push('绿色像素')).toBe('绿色像素');
    expect(stab.push('红色像素')).toBe('未知');
  });
});
