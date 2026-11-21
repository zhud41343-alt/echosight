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
      startedAt: 1000,
      completedAt: 1045,
      expiresAt: 4045,
      modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
    });
    expect(res.accepted).toBe(true);

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
        startedAt: 1000,
        completedAt: 1050,
        expiresAt: 4050,
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
        startedAt: 2000,
        completedAt: 2050,
        expiresAt: 5050,
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
        startedAt: 1900,
        completedAt: 2060,
        expiresAt: 5060,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
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
        startedAt: 100,
        completedAt: 150,
        expiresAt: Number.POSITIVE_INFINITY,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
    ).toBe(false);
    expect(router.resolveActive('depth', 200).matrix).toBeNull();
  });
});
