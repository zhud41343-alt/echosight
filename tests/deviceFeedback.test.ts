import { describe, it, expect, vi } from 'vitest';
import {
  HapticFeedbackController,
  ScreenWakeLockController,
  WakeLockSentinelLike,
} from '../src/engine/deviceFeedback';
import { AudioSpeechScheduler, SpeechBackend } from '../src/engine/audioScheduler';

describe('Mobile BLV Device Feedback, Screen Wake Lock & Screen-Reader Anti-Collision', () => {
  it('HapticFeedbackController triggers event-driven patterns with cooldown and safe fallback', () => {
    const vibrateMock = vi.fn(() => true);
    let now = 1000;
    const ctrl = new HapticFeedbackController({ vibrate: vibrateMock }, () => now);

    expect(ctrl.isSupported()).toBe(true);
    expect(ctrl.trigger('confirm')).toBe(true);
    expect(vibrateMock).toHaveBeenLastCalledWith([30]);

    // Within 200ms confirm cooldown -> skipped
    now += 50;
    expect(ctrl.trigger('confirm')).toBe(false);

    // Warning & unavailable patterns fire independently
    expect(ctrl.trigger('warning')).toBe(true);
    expect(vibrateMock).toHaveBeenLastCalledWith([60, 50, 60]);
    expect(ctrl.trigger('unavailable')).toBe(true);
    expect(vibrateMock).toHaveBeenLastCalledWith([90, 60, 90, 60, 120]);

    // Disabled -> no vibration
    ctrl.setEnabled(false);
    now += 5000;
    expect(ctrl.trigger('warning')).toBe(false);

    // Unsupported device -> safe no-op
    const unsupported = new HapticFeedbackController({});
    expect(unsupported.isSupported()).toBe(false);
    expect(unsupported.trigger('confirm')).toBe(false);
  });

  it('ScreenWakeLockController acquires, releases on hide, and re-acquires on foreground return', async () => {
    const releaseMock = vi.fn(async () => {});
    const requestMock = vi.fn(
      async (): Promise<WakeLockSentinelLike> => ({
        released: false,
        release: releaseMock,
        addEventListener: () => {},
      })
    );

    const ctrl = new ScreenWakeLockController({ wakeLock: { request: requestMock } });
    expect(ctrl.isSupported()).toBe(true);

    // Acquire on session start
    expect(await ctrl.acquire()).toBe(true);
    expect(ctrl.isHeld).toBe(true);
    expect(requestMock).toHaveBeenCalledTimes(1);

    // Page goes hidden -> release
    expect(await ctrl.handleVisibilityChange(false, true)).toBe(false);
    expect(ctrl.isHeld).toBe(false);
    expect(releaseMock).toHaveBeenCalledTimes(1);

    // Page returns visible while camera live -> re-acquire automatically
    expect(await ctrl.handleVisibilityChange(true, true)).toBe(true);
    expect(ctrl.isHeld).toBe(true);
    expect(requestMock).toHaveBeenCalledTimes(2);

    // Graceful failure on battery saver / NotAllowedError
    const failing = new ScreenWakeLockController({
      wakeLock: { request: async () => Promise.reject(new Error('NotAllowedError')) },
    });
    expect(await failing.acquire()).toBe(false);
    expect(failing.isHeld).toBe(false);
  });

  it('AudioSpeechScheduler screenReaderSafeMode prevents VoiceOver/TalkBack double-speech collision', () => {
    const speakMock = vi.fn();
    const cancelMock = vi.fn();
    const backend: SpeechBackend = {
      isAvailable: () => true,
      isSpeaking: () => false,
      cancel: cancelMock,
      speak: speakMock,
    };

    const scheduler = new AudioSpeechScheduler(backend, () => 1000);
    let lastNotified = '';
    scheduler.subscribe((s) => {
      lastNotified = s.lastText;
    });

    scheduler.setScreenReaderSafeMode(true);
    const res = scheduler.schedule({
      text: '正前方约 2 米有椅子',
      priority: 'USER_QUERY',
    });

    // Built-in speechSynthesis MUST NOT speak; ARIA subscriber still receives text
    expect(res.accepted).toBe(false);
    expect(res.reason).toBe('screen_reader_delegated');
    expect(speakMock).not.toHaveBeenCalled();
    expect(lastNotified).toBe('正前方约 2 米有椅子');
  });
});
