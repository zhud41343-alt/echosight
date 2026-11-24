import { describe, it, expect } from 'vitest';
import { AcousticMatrixRouter, validateAcousticMatrix } from '../src/engine/acousticRouter';

function fillMatrix(val: number, cols = 48, rows = 16): number[][] {
  return Array.from({ length: cols }, () => new Array(rows).fill(val));
}

describe('P0-1 (V8.4): AcousticMatrixRouter — Strict Sobel vs EchoDepthNet Neural Isolation', () => {
  it('HRTF reads Neural Matrix B in depth mode and Sobel Matrix A updates NEVER overwrite it', () => {
    const router = new AcousticMatrixRouter(48, 16);
    const sobelA = fillMatrix(0.15);
    const neuralB = fillMatrix(0.88);

    router.updateSobel(sobelA);
    expect(router.resolveActive('sobel', 1000).source).toBe('sobel');
    expect(router.resolveActive('sobel', 1000).matrix![0][0]).toBeCloseTo(0.15);

    // Switch to 'depth' mode -> begin new generation & commit Neural Matrix B
    const gen = router.beginGeneration();
    const res = router.commitNeuralPacket({
      matrix: neuralB,
      inputSource: 'camera',
      frameId: 1,
      generation: gen,
      capturedAt: 980,
      startedAt: 1000,
      completedAt: 1045,
      expiresAt: 3980,
      modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
    });
    expect(res.accepted).toBe(true);

    // P0-1 (V8.5): Equal frameId resubmission MUST be rejected (cannot refresh TTL!)
    expect(
      router.commitNeuralPacket({
        matrix: neuralB,
        inputSource: 'camera',
        frameId: 1,
        generation: gen,
        capturedAt: 980,
        startedAt: 1450,
        completedAt: 1500,
        expiresAt: 4500,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
    ).toBe(false);

    // Live camera rAF loop keeps updating Sobel Matrix A -> MUST NOT overwrite Neural Matrix B!
    router.updateSobel(fillMatrix(0.22));
    router.updateSobel(fillMatrix(0.31));

    const feed = router.resolveActive('depth', 1500);
    expect(feed.source).toBe('neural_echodepth');
    expect(feed.matrix![0][0]).toBeCloseTo(0.88);
    expect(feed.matrix![47][15]).toBeCloseTo(0.88);

    // Switching back to 'sobel' mode immediately routes to latest Sobel Matrix A (0.31)
    const sobelFeed = router.resolveActive('sobel', 1500);
    expect(sobelFeed.source).toBe('sobel');
    expect(sobelFeed.matrix![0][0]).toBeCloseTo(0.31);
  });

  it('rejects late neural inference after view/input switch, out-of-order frameId, and expired packets', () => {
    const router = new AcousticMatrixRouter(48, 16);
    const g1 = router.beginGeneration();

    // User switches view/preset -> generation advances to g2
    const g2 = router.beginGeneration();

    // Late inference from g1 arrives -> REJECTED!
    expect(
      router.commitNeuralPacket({
        matrix: fillMatrix(0.9),
        inputSource: 'camera',
        frameId: 1,
        generation: g1,
        capturedAt: 990,
        startedAt: 1000,
        completedAt: 1050,
        expiresAt: 3990,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
    ).toBe(false);

    // Valid frame 2 arrives on g2
    expect(
      router.commitNeuralPacket({
        matrix: fillMatrix(0.7),
        inputSource: 'camera',
        frameId: 2,
        generation: g2,
        capturedAt: 1990,
        startedAt: 2000,
        completedAt: 2050,
        expiresAt: 4990,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
    ).toBe(true);

    // Out-of-order older frame 1 on g2 -> REJECTED!
    expect(
      router.commitNeuralPacket({
        matrix: fillMatrix(0.2),
        inputSource: 'camera',
        frameId: 1,
        generation: g2,
        capturedAt: 1880,
        startedAt: 1900,
        completedAt: 2060,
        expiresAt: 4880,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
    ).toBe(false);

    // ONNX completing AFTER raw frame T0 already expired (> 3000ms) -> REJECTED!
    expect(
      router.commitNeuralPacket(
        {
          matrix: fillMatrix(0.6),
          inputSource: 'camera',
          frameId: 3,
          generation: g2,
          capturedAt: 2000,
          startedAt: 5100,
          completedAt: 5200,
          expiresAt: 8200,
          modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
        },
        5200
      ).accepted
    ).toBe(false);

    // After TTL expiration (camera freeze > 3000ms) -> stops audio (source: 'none', matrix: null)
    const expired = router.resolveActive('depth', 5500);
    expect(expired.source).toBe('none');
    expect(expired.matrix).toBeNull();
    expect(expired.reason).toBe('neural_expired');

    // Page hidden -> immediately blocks audio even when fresh
    const hidden = router.resolveActive('depth', 2500, true);
    expect(hidden.source).toBe('none');
    expect(hidden.matrix).toBeNull();
    expect(hidden.reason).toBe('page_hidden');
  });

  it('rejects NaN, Infinity, and malformed matrix dimensions', () => {
    const router = new AcousticMatrixRouter(48, 16);
    const g = router.beginGeneration();

    const nanMat = fillMatrix(0.5);
    nanMat[10][4] = Number.NaN;
    expect(validateAcousticMatrix(nanMat)).toBe(false);

    const infMat = fillMatrix(0.5);
    infMat[0][0] = Number.POSITIVE_INFINITY;
    expect(validateAcousticMatrix(infMat)).toBe(false);

    // Wrong shape (16x48 instead of 48x16)
    expect(validateAcousticMatrix(fillMatrix(0.5, 16, 48))).toBe(false);

    expect(
      router.commitNeuralPacket({
        matrix: nanMat,
        inputSource: 'preset',
        frameId: 1,
        generation: g,
        capturedAt: 95,
        startedAt: 100,
        completedAt: 150,
        expiresAt: Number.POSITIVE_INFINITY,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
    ).toBe(false);
    expect(router.resolveActive('depth', 200).matrix).toBeNull();
  });

  it('P1-01 (V8.7): rejects equal and older camera frameId in updateSobel so TTL cannot be refreshed', () => {
    const router = new AcousticMatrixRouter(48, 16);
    const m1 = fillMatrix(0.4);
    const m2 = fillMatrix(0.9);

    // First camera frame (frameId=5, T0=1000) accepted
    expect(
      router.updateSobel(m1, { inputSource: 'camera', frameId: 5, capturedAt: 1000 })
    ).toBe(true);
    expect(router.resolveActive('sobel', 1200).matrix![0][0]).toBeCloseTo(0.4);

    // Equal frameId=5 resubmitted later (T0=2500) -> MUST BE REJECTED!
    expect(
      router.updateSobel(m2, { inputSource: 'camera', frameId: 5, capturedAt: 2500 })
    ).toBe(false);
    // Older frameId=4 -> MUST BE REJECTED!
    expect(
      router.updateSobel(m2, { inputSource: 'camera', frameId: 4, capturedAt: 2600 })
    ).toBe(false);

    // Original frameId=5 expires strictly based on T0=1000 (+3000ms = 4000ms)
    expect(router.resolveActive('sobel', 4100).source).toBe('none');
    expect(router.resolveActive('sobel', 4100).reason).toBe('sobel_frame_stale');

    // Genuine new frameId=6 (T0=4200) accepted immediately
    expect(
      router.updateSobel(m2, { inputSource: 'camera', frameId: 6, capturedAt: 4200 })
    ).toBe(true);
    expect(router.resolveActive('sobel', 4300).matrix![0][0]).toBeCloseTo(0.9);
  });
});
