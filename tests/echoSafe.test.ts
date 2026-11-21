import { describe, it, expect } from 'vitest';
import {
  canEmitContinuousSpatialAudio,
  createInitialSignals,
  evaluateEchoSafe,
  FrameTracker,
  isPerceptionPacketUsable,
  PerceptionPacket,
  RuntimeMetricsTracker,
  SessionGuard,
  SingleFlightRunner,
} from '../src/engine/echoSafe';

describe('EchoSafe V3 reliability state machine, SessionGuard & PerceptionPacket', () => {
  const T = 50_000;

  it('starts in IDLE and cannot trust live perception', () => {
    const snap = evaluateEchoSafe(createInitialSignals(), T);
    expect(snap.state).toBe('IDLE');
    expect(snap.canTrustLivePerception).toBe(false);
    expect(snap.dataSource).toBe('UNAVAILABLE');
  });

  it('enters DEMO and strictly isolates SIMULATED_SCENE from live perception', () => {
    const snap = evaluateEchoSafe({ ...createInitialSignals(), isDemoMode: true }, T);
    expect(snap.state).toBe('DEMO');
    expect(snap.dataSource).toBe('SIMULATED_SCENE');
    expect(snap.canTrustLivePerception).toBe(false);
  });

  it('P0-1: NEVER enters ACTIVE when lastFrameAt is null (stays INITIALIZING within grace, then UNAVAILABLE)', () => {
    const waiting = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        sessionStartedAt: T - 1000,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: null,
        modelState: 'ready',
        audioState: 'running',
      },
      T
    );
    expect(waiting.state).toBe('INITIALIZING');
    expect(waiting.canTrustLivePerception).toBe(false);

    const timedOut = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        sessionStartedAt: T - 6000,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: null,
        modelState: 'ready',
        audioState: 'running',
      },
      T
    );
    expect(timedOut.state).toBe('UNAVAILABLE');
    expect(timedOut.canTrustLivePerception).toBe(false);
  });

  it('P0-1: NEVER enters ACTIVE when model is ready but lastInferenceAt is null (enters DEGRADED)', () => {
    const snap = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        sessionStartedAt: T - 1500,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: T - 100,
        modelState: 'ready',
        lastInferenceAt: null,
        audioState: 'running',
      },
      T
    );
    expect(snap.state).toBe('DEGRADED');
    expect(snap.inferenceFresh).toBe(false);
    expect(snap.canTrustLivePerception).toBe(false);
  });

  it('P0-1: enters UNAVAILABLE when camera permission denied, track ended, video frozen, or page hidden', () => {
    expect(
      evaluateEchoSafe(
        { ...createInitialSignals(), requestedLive: true, cameraPermission: 'denied' },
        T
      ).state
    ).toBe('UNAVAILABLE');

    expect(
      evaluateEchoSafe(
        {
          ...createInitialSignals(),
          requestedLive: true,
          cameraPermission: 'granted',
          cameraTrackLive: false,
        },
        T
      ).state
    ).toBe('UNAVAILABLE');

    // Frozen video (no real frame advance for 3.5s)
    const frozen = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: T - 3500,
        modelState: 'ready',
        lastInferenceAt: T - 3500,
        audioState: 'running',
      },
      T
    );
    expect(frozen.state).toBe('UNAVAILABLE');
    expect(frozen.frameFresh).toBe(false);

    // Page hidden
    expect(
      evaluateEchoSafe(
        {
          ...createInitialSignals(),
          requestedLive: true,
          cameraPermission: 'granted',
          cameraTrackLive: true,
          lastFrameAt: T - 100,
          modelState: 'ready',
          lastInferenceAt: T - 200,
          audioState: 'running',
          pageHidden: true,
        },
        T
      ).state
    ).toBe('UNAVAILABLE');
  });

  it('enters ACTIVE only when camera, real frame, model, inference, visibility, and audio are all fresh', () => {
    const snap = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        sessionStartedAt: T - 5000,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: T - 120,
        modelState: 'ready',
        lastInferenceAt: T - 400,
        lastInferenceDurationMs: 95,
        audioState: 'running',
        pageHidden: false,
      },
      T
    );
    expect(snap.state).toBe('ACTIVE');
    expect(snap.frameFresh).toBe(true);
    expect(snap.inferenceFresh).toBe(true);
    expect(snap.canTrustLivePerception).toBe(true);
    expect(snap.dataSource).toBe('REAL_CAMERA');
  });

  it('P0-2: SessionGuard invalidates stale async results across start -> stop -> restart cycles', () => {
    const guard = new SessionGuard();
    const s1 = guard.begin();
    expect(guard.isValid(s1)).toBe(true);

    // User stops while inference from s1 is still in flight
    guard.invalidate();
    expect(guard.isValid(s1)).toBe(false);
    expect(guard.isActive).toBe(false);

    // Rapid restart creates s2; late results from s1 remain rejected
    const s2 = guard.begin();
    expect(s2).toBeGreaterThan(s1);
    expect(guard.isValid(s1)).toBe(false);
    expect(guard.isValid(s2)).toBe(true);
  });

  it('P0-4 & P0-2: isPerceptionPacketUsable enforces currentSafeState, frameId, raw frame age, temporal order, and rejects future timestamps', () => {
    const base: PerceptionPacket<string> = {
      dataSource: 'REAL_CAMERA',
      sessionId: 3,
      frameId: 5,
      capturedAt: T - 300,
      inferredAt: T - 200,
      expiresAt: T + 3000,
      safeState: 'ACTIVE',
      detections: ['chair'],
    };
    expect(isPerceptionPacketUsable(base, 3, 'ACTIVE', T)).toBe(true);
    // Current system state degraded/unavailable rejects even a freshly-created packet
    expect(isPerceptionPacketUsable(base, 3, 'DEGRADED', T)).toBe(false);
    expect(isPerceptionPacketUsable(base, 3, 'UNAVAILABLE', T)).toBe(false);
    // Stale session or expired TTL
    expect(isPerceptionPacketUsable(base, 4, 'ACTIVE', T)).toBe(false);
    expect(isPerceptionPacketUsable(base, 3, 'ACTIVE', T + 4000)).toBe(false);
    // Stale raw camera frame (> 3000ms) even if expiresAt is still in future
    expect(
      isPerceptionPacketUsable({ ...base, capturedAt: T - 3500 }, 3, 'ACTIVE', T, 3000)
    ).toBe(false);
    // Illegal temporal ordering (capturedAt > inferredAt) or future timestamp
    expect(
      isPerceptionPacketUsable({ ...base, capturedAt: T - 100, inferredAt: T - 200 }, 3, 'ACTIVE', T)
    ).toBe(false);
    expect(
      isPerceptionPacketUsable({ ...base, capturedAt: T + 500, inferredAt: T + 600 }, 3, 'ACTIVE', T)
    ).toBe(false);
    // Simulated data strictly rejected
    expect(
      isPerceptionPacketUsable({ ...base, dataSource: 'SIMULATED_SCENE' }, 3, 'ACTIVE', T)
    ).toBe(false);
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

  it('RuntimeMetricsTracker separates vision FPS from AI FPS and tracks E2E P95 latency', () => {
    const tracker = new RuntimeMetricsTracker();
    for (let i = 0; i < 60; i++) tracker.recordVisionFrame(1000 + i * 33);
    for (let i = 0; i < 10; i++) tracker.recordInference(1000 + i * 200, 60 + i * 10, 120 + i * 15);
    const snap = tracker.getSnapshot(3000);
    expect(snap.visionFps).toBeGreaterThan(snap.aiFps);
    expect(snap.p95InferenceMs).toBeGreaterThanOrEqual(snap.avgInferenceMs);
    expect(snap.p95E2eMs).toBeGreaterThanOrEqual(snap.avgE2eMs);
  });

  it('P0-1 (V5): FrameTracker prevents duplicate inference on same video frame (1fps cam / 3fps sched & 2fps cam / 5fps sched)', () => {
    const ft = new FrameTracker();
    // 1fps camera (frames at t=1000, 2000, 3000) vs 3fps inference scheduler (every 333ms)
    let claimedCount = 0;
    for (let t = 1000; t <= 3000; t += 333) {
      if (t === 1000 || t === 1999 || t === 2998) {
        ft.recordNewFrame(t);
      }
      const token = ft.claimNextFrameForInference(t, 2500);
      if (token) claimedCount += 1;
    }
    expect(claimedCount).toBe(3);

    // 2fps camera (frames every 500ms) vs 5fps scheduler (every 200ms) over 2 seconds = 4 frames
    ft.reset();
    const claimedIds: number[] = [];
    for (let t = 0; t < 2000; t += 200) {
      if (t % 500 === 0 || (t > 0 && Math.floor(t / 500) > Math.floor((t - 200) / 500))) {
        ft.recordNewFrame(t);
      }
      const token = ft.claimNextFrameForInference(t, 2500);
      if (token) claimedIds.push(token.frameId);
    }
    expect(claimedIds).toEqual([1, 2, 3, 4]);
    // Polling again without a new frame returns null
    expect(ft.claimNextFrameForInference(2100, 2500)).toBeNull();
  });

  it('P0-1 (V5): FrameTracker rejects frozen/stale frames and recovers cleanly on first new frame after freeze', () => {
    const ft = new FrameTracker();
    ft.recordNewFrame(1000);
    // Camera freezes for 4 seconds (> 2500ms threshold) before scheduler polls
    expect(ft.claimNextFrameForInference(5000, 2500)).toBeNull();
    // Camera recovers with a brand-new frame at t=5200
    const newId = ft.recordNewFrame(5200);
    expect(newId).toBe(2);
    const token = ft.claimNextFrameForInference(5250, 2500);
    expect(token).toEqual({ frameId: 2, capturedAt: 5200 });
  });

  it('P0-2 (V5): canEmitContinuousSpatialAudio immediately stops continuous audio on video freeze, UNAVAILABLE, camera revoke, background, or suspended audio', () => {
    const liveSignals = {
      ...createInitialSignals(),
      requestedLive: true,
      sessionStartedAt: T - 5000,
      cameraPermission: 'granted' as const,
      cameraTrackLive: true,
      lastFrameAt: T - 100,
      modelState: 'ready' as const,
      lastInferenceAt: T - 200,
      audioState: 'running' as const,
      pageHidden: false,
    };
    expect(canEmitContinuousSpatialAudio(liveSignals, T).allowed).toBe(true);

    // Video frozen (> 2500ms without new frame) -> MUST stop immediately
    expect(
      canEmitContinuousSpatialAudio({ ...liveSignals, lastFrameAt: T - 2600 }, T).allowed
    ).toBe(false);

    // Camera track ended / permission revoked -> MUST stop immediately
    expect(
      canEmitContinuousSpatialAudio({ ...liveSignals, cameraTrackLive: false }, T).allowed
    ).toBe(false);
    expect(
      canEmitContinuousSpatialAudio({ ...liveSignals, cameraPermission: 'denied' }, T).allowed
    ).toBe(false);

    // Page hidden -> MUST stop immediately
    expect(
      canEmitContinuousSpatialAudio({ ...liveSignals, pageHidden: true }, T).allowed
    ).toBe(false);

    // AudioContext suspended -> MUST stop immediately
    expect(
      canEmitContinuousSpatialAudio({ ...liveSignals, audioState: 'suspended' }, T).allowed
    ).toBe(false);

    // Idle / stopped -> MUST stop immediately
    expect(
      canEmitContinuousSpatialAudio({ ...liveSignals, requestedLive: false }, T).allowed
    ).toBe(false);
  });

  it('P1-2 (V6): isPerceptionPacketUsable rejects missing, malformed, or out-of-order frameId at runtime', () => {
    const base: PerceptionPacket<string> = {
      dataSource: 'REAL_CAMERA',
      sessionId: 3,
      frameId: 12,
      capturedAt: T - 300,
      inferredAt: T - 200,
      expiresAt: T + 3000,
      safeState: 'ACTIVE',
      detections: ['chair'],
    };
    expect(isPerceptionPacketUsable(base, 3, 'ACTIVE', T, 3000, 12)).toBe(true);
    // Missing frameId at runtime MUST be rejected
    expect(
      isPerceptionPacketUsable(
        { ...base, frameId: undefined as unknown as number },
        3,
        'ACTIVE',
        T
      )
    ).toBe(false);
    // Out-of-order older frameId (< minFrameId) within same session rejected
    expect(isPerceptionPacketUsable({ ...base, frameId: 11 }, 3, 'ACTIVE', T, 3000, 12)).toBe(false);
    expect(isPerceptionPacketUsable({ ...base, frameId: 0 }, 3, 'ACTIVE', T)).toBe(false);
    expect(isPerceptionPacketUsable({ ...base, frameId: -2 }, 3, 'ACTIVE', T)).toBe(false);
  });
});
