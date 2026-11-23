import { describe, it, expect, vi } from 'vitest';
import { FrameTracker } from '../src/engine/echoSafe';
import { AcousticMatrixRouter, ScannerViewMode } from '../src/engine/acousticRouter';
import { spatialAudio } from '../src/engine/spatialAudio';

function makeMatrix(val: number): number[][] {
  return Array.from({ length: 48 }, () => new Array(16).fill(val));
}

describe('P0-1 & P0-2 (V8.5): End-to-End Video Frame -> ONNX -> AcousticMatrixRouter -> Web Audio HRTF Pipeline', () => {
  it('1fps camera + 2.2fps ONNX timer: runs ONNX once per real frame, freezes cleanly without refreshing TTL, stops HRTF, and recovers on new frame', () => {
    const frameTracker = new FrameTracker();
    const router = new AcousticMatrixRouter(48, 16);
    const playSpy = vi.spyOn(spatialAudio, 'playScanColumn').mockReturnValue({
      azimuthDeg: 0,
      dominantFreqHz: 440,
    });
    const stopSpy = vi.spyOn(spatialAudio, 'stopAllImmediate').mockImplementation(() => {});

    const gen = router.beginGeneration();
    let viewMode: ScannerViewMode = 'depth';
    let onnxRuns = 0;

    // Helper simulating one VisionScanner ONNX tick (450ms cadence)
    const runOnnxTick = (now: number, neuralVal: number) => {
      const claimed = frameTracker.claimNextFrameForInference(now, 2500);
      if (!claimed) {
        const cur = router.resolveActive(viewMode, now, false);
        if (!cur.matrix) spatialAudio.stopAllImmediate();
        return false;
      }
      onnxRuns += 1;
      const completedAt = now + 45;
      router.commitNeuralPacket(
        {
          matrix: makeMatrix(neuralVal),
          inputSource: 'camera',
          frameId: claimed.frameId,
          generation: gen,
          capturedAt: claimed.capturedAt,
          startedAt: now,
          completedAt,
          expiresAt: claimed.capturedAt + 3000,
          modelVersion: 'EchoDepthNet-ResidualEGDFM-1.11M',
        },
        completedAt
      );
      return true;
    };

    // Helper simulating one VisionScanner rAF HRTF scan tick (only updates Sobel on genuinely NEW camera frames)
    let lastSobelFrameId = 0;
    const runHrtfTick = (now: number, sobelVal: number) => {
      const fSnap = frameTracker.getSnapshot();
      if (fSnap.latestFrameId > lastSobelFrameId && fSnap.latestFrameCapturedAt !== null) {
        lastSobelFrameId = fSnap.latestFrameId;
        router.updateSobel(makeMatrix(sobelVal), {
          inputSource: 'camera',
          frameId: fSnap.latestFrameId,
          capturedAt: fSnap.latestFrameCapturedAt,
        });
      }
      const feed = router.resolveActive(viewMode, now, false);
      if (feed.matrix) {
        spatialAudio.playScanColumn(feed.matrix[0], 0.5, 0.055, {
          scaleMode: 'pentatonic',
          useHRTF: true,
          volume: 0.7,
        });
      } else {
        spatialAudio.stopAllImmediate();
      }
    };

    // t=1000ms: Camera delivers Frame 1 (1fps)
    frameTracker.recordNewFrame(1000);
    expect(runOnnxTick(1050, 0.85)).toBe(true);
    // High-frequency Sobel updates (0.12) MUST NOT overwrite Neural Matrix (0.85)!
    runHrtfTick(1200, 0.12);
    expect(playSpy).toHaveBeenLastCalledWith(
      expect.arrayContaining([0.85]),
      0.5,
      0.055,
      expect.any(Object)
    );

    // t=1500ms & t=1950ms: ONNX 2.2fps timer fires WITHOUT a new camera frame -> MUST skip!
    expect(runOnnxTick(1500, 0.85)).toBe(false);
    expect(runOnnxTick(1950, 0.85)).toBe(false);
    expect(onnxRuns).toBe(1);

    // Camera FREEZES after Frame 1 (track stays live, no new frame arrives past t=4100ms)
    expect(runOnnxTick(4100, 0.85)).toBe(false);
    const playCallsBeforeFreeze = playSpy.mock.calls.length;
    runHrtfTick(4200, 0.25);
    // HRTF MUST NOT play expired neural matrix; stopAllImmediate MUST be called!
    expect(playSpy.mock.calls.length).toBe(playCallsBeforeFreeze);
    expect(stopSpy).toHaveBeenCalled();

    // P0 (V8.6): Switching to default 'sobel' mode while camera is STILL FROZEN MUST ALSO STAY SILENT!
    viewMode = 'sobel';
    const stopCallsBeforeSobelFreeze = stopSpy.mock.calls.length;
    runHrtfTick(4300, 0.25);
    expect(playSpy.mock.calls.length).toBe(playCallsBeforeFreeze);
    expect(stopSpy.mock.calls.length).toBeGreaterThan(stopCallsBeforeSobelFreeze);
    expect(router.resolveActive('sobel', 4300, false).reason).toBe('sobel_frame_stale');

    // Camera RECOVERS at t=5000ms with genuine new Frame 2 -> Sobel plays fresh Frame 2, then Depth plays fresh Neural Matrix (0.92)
    frameTracker.recordNewFrame(5000);
    runHrtfTick(5020, 0.25);
    expect(playSpy).toHaveBeenLastCalledWith(
      expect.arrayContaining([0.25]),
      0.5,
      0.055,
      expect.any(Object)
    );

    viewMode = 'depth';
    expect(runOnnxTick(5050, 0.92)).toBe(true);
    expect(onnxRuns).toBe(2);
    runHrtfTick(5150, 0.11);
    expect(playSpy).toHaveBeenLastCalledWith(
      expect.arrayContaining([0.92]),
      0.5,
      0.055,
      expect.any(Object)
    );

    playSpy.mockRestore();
    stopSpy.mockRestore();
  });
});
