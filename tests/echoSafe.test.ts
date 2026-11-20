import { describe, it, expect } from 'vitest';
import {
  createInitialSignals,
  evaluateEchoSafe,
  SingleFlightRunner,
  RuntimeMetricsTracker,
} from '../src/engine/echoSafe';

describe('EchoSafe unified reliability state machine', () => {
  const T = 50_000;

  it('starts in IDLE and cannot trust live perception', () => {
    const snap = evaluateEchoSafe(createInitialSignals(), T);
    expect(snap.state).toBe('IDLE');
    expect(snap.canTrustLivePerception).toBe(false);
    expect(snap.dataSource).toBe('none');
  });

  it('enters DEMO and strictly isolates demo_preset from live perception', () => {
    const snap = evaluateEchoSafe({ ...createInitialSignals(), isDemoMode: true }, T);
    expect(snap.state).toBe('DEMO');
    expect(snap.dataSource).toBe('demo_preset');
    expect(snap.canTrustLivePerception).toBe(false);
  });

  it('enters UNAVAILABLE when camera permission is denied', () => {
    const snap = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        cameraPermission: 'denied',
        modelState: 'ready',
        audioState: 'running',
      },
      T
    );
    expect(snap.state).toBe('UNAVAILABLE');
    expect(snap.canTrustLivePerception).toBe(false);
    expect(snap.dataSource).toBe('none');
  });

  it('enters UNAVAILABLE when camera track stops or video frames go stale', () => {
    const disconnected = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        cameraPermission: 'granted',
        cameraTrackLive: false,
        modelState: 'ready',
        audioState: 'running',
      },
      T
    );
    expect(disconnected.state).toBe('UNAVAILABLE');

    const stale = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: T - 3500,
        modelState: 'ready',
        audioState: 'running',
      },
      T
    );
    expect(stale.state).toBe('UNAVAILABLE');
    expect(stale.frameStale).toBe(true);
  });

  it('enters UNAVAILABLE when page is hidden in background', () => {
    const snap = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: T - 100,
        modelState: 'ready',
        audioState: 'running',
        pageHidden: true,
      },
      T
    );
    expect(snap.state).toBe('UNAVAILABLE');
    expect(snap.canTrustLivePerception).toBe(false);
  });

  it('enters DEGRADED when model fails to load (never pretends healthy)', () => {
    const snap = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: T - 100,
        modelState: 'failed',
        audioState: 'running',
      },
      T
    );
    expect(snap.state).toBe('DEGRADED');
    expect(snap.canTrustLivePerception).toBe(false);
  });

  it('enters ACTIVE only when camera, frames, model, page visibility, and audio are all healthy', () => {
    const snap = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: T - 120,
        modelState: 'ready',
        lastInferenceAt: T - 400,
        lastInferenceDurationMs: 140,
        audioState: 'running',
        pageHidden: false,
      },
      T
    );
    expect(snap.state).toBe('ACTIVE');
    expect(snap.canTrustLivePerception).toBe(true);
    expect(snap.dataSource).toBe('live_camera');
  });

  it('SingleFlightRunner prevents overlapping async inference pile-up', async () => {
    const runner = new SingleFlightRunner();
    let resolveFirst!: (v: string) => void;
    const p1 = runner.run(
      () =>
        new Promise<string>((res) => {
          resolveFirst = res;
        })
    );
    const p2 = await runner.run(async () => 'second');
    expect(p2.skipped).toBe(true);
    resolveFirst('first');
    const r1 = await p1;
    expect(r1.skipped).toBe(false);
    expect(r1.value).toBe('first');
  });

  it('RuntimeMetricsTracker separates vision FPS from AI inference FPS & computes P95', () => {
    const tracker = new RuntimeMetricsTracker();
    for (let i = 0; i < 60; i++) tracker.recordVisionFrame(1000 + i * 33);
    for (let i = 0; i < 10; i++) tracker.recordInference(1000 + i * 200, 80 + i * 20);
    const snap = tracker.getSnapshot(3000);
    expect(snap.visionFps).toBeGreaterThan(snap.aiFps);
    expect(snap.p95InferenceMs).toBeGreaterThanOrEqual(snap.avgInferenceMs);
  });
});
