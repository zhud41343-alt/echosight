import { describe, it, expect } from 'vitest';
import { AcousticMatrixRouter } from '../src/engine/acousticRouter';
import { FrameTracker } from '../src/engine/echoSafe';

function fillMatrix(val: number, cols = 48, rows = 16): number[][] {
  return Array.from({ length: cols }, () => new Array(rows).fill(val));
}

describe('V8.7–V8.8 Safety Hardening: Camera Disconnect, Freeze, Visibility & Recovery', () => {
  it('P0-01: camera disconnect invalidates BOTH Sobel & Neural routers so audio never falls back to stale data', () => {
    const router = new AcousticMatrixRouter(48, 16);
    const gen = router.beginGeneration();

    expect(
      router.updateSobel(fillMatrix(0.45), { inputSource: 'camera', frameId: 1, capturedAt: 1000 })
    ).toBe(true);
    expect(
      router.commitNeuralPacket({
        matrix: fillMatrix(0.82),
        inputSource: 'camera',
        frameId: 1,
        generation: gen,
        capturedAt: 1000,
        startedAt: 1010,
        completedAt: 1055,
        expiresAt: 4000,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
    ).toBe(true);

    // Simulate track.onended disconnect cleanup
    router.invalidateSobel();
    router.invalidateNeural();

    expect(router.resolveActive('sobel', 1100).matrix).toBeNull();
    expect(router.resolveActive('source', 1100).matrix).toBeNull();
    expect(router.resolveActive('matrix', 1100).matrix).toBeNull();
    expect(router.resolveActive('depth', 1100).matrix).toBeNull();
  });

  it('P0-02 + P1-01: FrameTracker + Router freeze blocks both Sobel & Neural strictly from T0 and recovers on real new frame', () => {
    const tracker = new FrameTracker();
    const router = new AcousticMatrixRouter(48, 16);
    const gen = router.beginGeneration();

    // T0 = 1000: real video frame #1 arrives
    const f1 = tracker.recordNewFrame(1000);
    expect(f1).toBe(1);
    expect(
      router.updateSobel(fillMatrix(0.35), { inputSource: 'camera', frameId: f1, capturedAt: 1000 })
    ).toBe(true);
    expect(
      router.commitNeuralPacket({
        matrix: fillMatrix(0.75),
        inputSource: 'camera',
        frameId: f1,
        generation: gen,
        capturedAt: 1000,
        startedAt: 1010,
        completedAt: 1050,
        expiresAt: 4000,
        modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
      }).accepted
    ).toBe(true);

    // Camera freezes: repeated polling reads same frameId=1 -> MUST NOT refresh TTL!
    const snapFrozen = tracker.getSnapshot();
    expect(snapFrozen.latestFrameId).toBe(1);
    expect(
      router.updateSobel(fillMatrix(0.99), {
        inputSource: 'camera',
        frameId: snapFrozen.latestFrameId,
        capturedAt: 2800,
      })
    ).toBe(false);
    expect(
      router.commitNeuralPacket(
        {
          matrix: fillMatrix(0.99),
          inputSource: 'camera',
          frameId: snapFrozen.latestFrameId,
          generation: gen,
          capturedAt: 1000,
          startedAt: 2800,
          completedAt: 2845,
          expiresAt: 5845,
          modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
        },
        2845
      ).accepted
    ).toBe(false);

    // At T = 4100 (> T0 + 3000ms), BOTH Sobel and Neural expire cleanly
    expect(router.resolveActive('sobel', 4100).source).toBe('none');
    expect(router.resolveActive('depth', 4100).source).toBe('none');

    // At T = 4200, camera recovers with genuine new frame #2
    const f2 = tracker.recordNewFrame(4200);
    expect(f2).toBe(2);
    expect(
      router.updateSobel(fillMatrix(0.6), { inputSource: 'camera', frameId: f2, capturedAt: 4200 })
    ).toBe(true);
    expect(router.resolveActive('sobel', 4250).matrix![0][0]).toBeCloseTo(0.6);
  });

  it('P1-03: page hidden immediately mutes ALL view modes (source, sobel, matrix, depth)', () => {
    const router = new AcousticMatrixRouter(48, 16);
    const gen = router.beginGeneration();
    router.updateSobel(fillMatrix(0.5), { inputSource: 'camera', frameId: 1, capturedAt: 1000 });
    router.commitNeuralPacket({
      matrix: fillMatrix(0.8),
      inputSource: 'camera',
      frameId: 1,
      generation: gen,
      capturedAt: 1000,
      startedAt: 1010,
      completedAt: 1050,
      expiresAt: 4000,
      modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
    });

    for (const mode of ['source', 'sobel', 'matrix', 'depth'] as const) {
      const hiddenFeed = router.resolveActive(mode, 1200, true);
      expect(hiddenFeed.source).toBe('none');
      expect(hiddenFeed.matrix).toBeNull();
      expect(hiddenFeed.reason).toBe('page_hidden');
    }
  });
});
