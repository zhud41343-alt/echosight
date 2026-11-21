import { describe, it, expect, vi } from 'vitest';
import {
  AccessibleNotificationDispatcher,
  planEchoSafeTransition,
} from '../src/engine/accessibleNotifier';
import { AudioSpeechScheduler, SpeechBackend } from '../src/engine/audioScheduler';
import { HapticFeedbackController } from '../src/engine/deviceFeedback';
import { createInitialSignals, evaluateEchoSafe } from '../src/engine/echoSafe';
import { buildSpatialDetectionNotification, SpatialDetection } from '../src/engine/aiObjectDetector';

function makeChairDetection(): SpatialDetection {
  return {
    labelEn: 'chair',
    labelZh: '椅子',
    score: 0.78,
    bbox: [220, 140, 120, 160],
    direction: '正前方',
    clockFace: '12点方向',
    distance: '近距离',
    distanceMeters: 1.5,
    distanceText: '较近（未标定粗估）',
    distanceCalibrated: false,
    priority: 'obstacle',
    source: 'tfjs',
    detectedAt: 1000,
  };
}

describe('P0-1 & P0-2 (V6): AccessibleNotificationDispatcher & EchoSafe Transition Coordinator', () => {
  const createDeps = (ttsAvailable = true, vibrateSupported = true) => {
    let now = 10_000;
    const speakMock = vi.fn((_text: string, _rate: number, onDone: () => void) => {
      // Keep active until explicitly finished or cancelled
      void onDone;
    });
    const cancelMock = vi.fn();
    const backend: SpeechBackend = {
      isAvailable: () => ttsAvailable,
      isSpeaking: () => false,
      cancel: cancelMock,
      speak: speakMock,
    };
    const vibrateMock = vi.fn(() => true);
    const scheduler = new AudioSpeechScheduler(backend, () => now);
    const haptics = new HapticFeedbackController(
      vibrateSupported ? { vibrate: vibrateMock } : {},
      () => now
    );
    const dispatcher = new AccessibleNotificationDispatcher(scheduler, haptics, () => now);
    return {
      scheduler,
      haptics,
      dispatcher,
      speakMock,
      cancelMock,
      vibrateMock,
      setNow: (v: number) => {
        now = v;
      },
      getNow: () => now,
    };
  };

  it('P0-1: delivers automatic object detection to ARIA Live EVEN when screenReaderSafeMode suppresses browser TTS', () => {
    const { scheduler, dispatcher, speakMock } = createDeps(true, true);
    scheduler.setScreenReaderSafeMode(true);

    const payload = buildSpatialDetectionNotification([makeChairDetection()], false);
    expect(payload).not.toBeNull();

    const res = dispatcher.dispatch(payload!);
    expect(res.delivered).toBe(true);
    expect(res.text).toContain('正前方');
    expect(res.text).toContain('椅子');
    expect(res.ariaChannel).toBe('polite');
    expect(res.ttsAttempted).toBe(false);
    expect(res.ttsSpoken).toBe(false);
    expect(speakMock).not.toHaveBeenCalled();
  });

  it('P0-1: delivers user query in screenReaderSafeMode & when browser TTS is unavailable, while throttling duplicate ambient spam', () => {
    const { dispatcher, setNow } = createDeps(false, false);

    // Ambient message still delivered to ARIA even when browser TTS is unavailable
    const a1 = dispatcher.dispatch({
      text: '正前方检测到椅子',
      priority: 'AMBIENT',
      dedupeKey: 'chair-front',
      cooldownMs: 10000,
    });
    expect(a1.delivered).toBe(true);
    expect(a1.ariaChannel).toBe('polite');
    expect(a1.ttsSpoken).toBe(false);

    // Same ambient message 1s later -> throttled so VoiceOver/TalkBack is not spammed
    setNow(11_000);
    const a2 = dispatcher.dispatch({
      text: '正前方检测到椅子',
      priority: 'AMBIENT',
      dedupeKey: 'chair-front',
      cooldownMs: 10000,
    });
    expect(a2.delivered).toBe(false);
    expect(a2.dropReason).toBe('cooldown');

    // Active user query -> NEVER dropped even right after ambient
    const q1 = dispatcher.dispatch({
      text: '正前方较近检测到椅子',
      priority: 'USER_QUERY',
    });
    expect(q1.delivered).toBe(true);
    expect(q1.ariaChannel).toBe('polite');
  });

  it('P0-2: video freeze (ACTIVE -> UNAVAILABLE) triggers assertive ARIA + TTS + audio stop even when vibration is unsupported, and does NOT spam every 1s', () => {
    const { dispatcher, speakMock } = createDeps(true, false);
    const T = 50_000;

    // Frozen video snapshot (no frame for 3.5s)
    const frozenSnap = evaluateEchoSafe(
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
    expect(frozenSnap.state).toBe('UNAVAILABLE');

    // Transition ACTIVE -> UNAVAILABLE
    const plan1 = planEchoSafeTransition('ACTIVE', frozenSnap);
    expect(plan1.changed).toBe(true);
    expect(plan1.stopSpatialAudio).toBe(true);
    expect(plan1.clearPerceptionData).toBe(true);
    expect(plan1.notification).not.toBeNull();

    const dispatched = dispatcher.dispatch({ ...plan1.notification!, now: T });
    expect(dispatched.delivered).toBe(true);
    expect(dispatched.ariaChannel).toBe('assertive');
    expect(dispatched.ttsSpoken).toBe(true);
    expect(speakMock).toHaveBeenCalledTimes(1);
    // Truthfully reports vibration did NOT fire on unsupported device
    expect(dispatched.hapticTriggered).toBe(false);

    // Next 1s health tick while still UNAVAILABLE -> NO repeated notification spam!
    const plan2 = planEchoSafeTransition('UNAVAILABLE', frozenSnap);
    expect(plan2.changed).toBe(false);
    expect(plan2.stopSpatialAudio).toBe(true);
    expect(plan2.notification).toBeNull();
  });

  it('P0-2: handles ACTIVE -> DEGRADED -> UNAVAILABLE -> INITIALIZING -> ACTIVE lifecycle transitions cleanly', () => {
    const T = 60_000;
    const degradedSnap = evaluateEchoSafe(
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
    expect(degradedSnap.state).toBe('DEGRADED');
    const pDeg = planEchoSafeTransition('ACTIVE', degradedSnap);
    expect(pDeg.notification?.priority).toBe('EMERGENCY');
    expect(pDeg.clearPerceptionData).toBe(true);

    const initSnap = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        initializing: true,
      },
      T
    );
    const pRec = planEchoSafeTransition('UNAVAILABLE', initSnap);
    expect(pRec.notification?.text).toContain('重新连接');

    const activeSnap = evaluateEchoSafe(
      {
        ...createInitialSignals(),
        requestedLive: true,
        cameraPermission: 'granted',
        cameraTrackLive: true,
        lastFrameAt: T - 100,
        modelState: 'ready',
        lastInferenceAt: T - 200,
        audioState: 'running',
      },
      T
    );
    const pAct = planEchoSafeTransition('INITIALIZING', activeSnap);
    expect(pAct.notification?.haptic).toBe('confirm');
    expect(pAct.clearPerceptionData).toBe(false);
  });
});
