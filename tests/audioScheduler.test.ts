import { describe, it, expect } from 'vitest';
import { AudioSpeechScheduler, SpeechBackend } from '../src/engine/audioScheduler';

function createMockBackendWithLateCallbacks() {
  const spoken: string[] = [];
  const callbacks: Array<() => void> = [];
  let speaking = false;
  let cancelCount = 0;

  const backend: SpeechBackend = {
    isAvailable: () => true,
    isSpeaking: () => speaking,
    cancel: () => {
      cancelCount++;
      speaking = false;
    },
    speak: (text, _rate, onDone) => {
      spoken.push(text);
      speaking = true;
      callbacks.push(() => {
        speaking = false;
        onDone();
      });
    },
  };

  return {
    backend,
    spoken,
    getCancelCount: () => cancelCount,
    triggerDoneForIndex: (idx: number) => callbacks[idx]?.(),
  };
}

describe('Unified AudioSpeechScheduler V3 (race-free playbackId + unified clock + spatial coordination)', () => {
  it('P0-3: late onend/onerror callback from a cancelled utterance NEVER clobbers newer EMERGENCY speech', () => {
    let clockNow = 1000;
    const mock = createMockBackendWithLateCallbacks();
    const sched = new AudioSpeechScheduler(mock.backend, () => clockNow);

    // 1. Start AMBIENT utterance (index 0)
    const r1 = sched.schedule({ text: '普通环境播报', priority: 'AMBIENT' });
    expect(r1.accepted).toBe(true);
    expect(sched.getCurrentPriority()).toBe('AMBIENT');

    // 2. EMERGENCY preempts AMBIENT (index 1)
    clockNow = 1200;
    const r2 = sched.schedule({ text: '摄像头已断开', priority: 'EMERGENCY' });
    expect(r2.accepted).toBe(true);
    expect(r2.reason).toBe('preempted_and_spoken');
    expect(sched.getCurrentPriority()).toBe('EMERGENCY');

    // 3. Browser fires LATE onend/onerror for the cancelled AMBIENT utterance (index 0)!
    clockNow = 1300;
    mock.triggerDoneForIndex(0);

    // State MUST STILL belong to the active EMERGENCY utterance!
    expect(sched.getCurrentPriority()).toBe('EMERGENCY');

    // A new AMBIENT message must still be blocked by the active EMERGENCY utterance
    clockNow = 5000;
    const r3 = sched.schedule({ text: '另一条普通播报', priority: 'AMBIENT' });
    expect(r3.accepted).toBe(false);
    expect(r3.reason).toBe('dropped_lower_priority');
  });

  it('blocks continuous spatial scan sound during active EMERGENCY/USER_QUERY speech and when muted', () => {
    let clockNow = 2000;
    const mock = createMockBackendWithLateCallbacks();
    const sched = new AudioSpeechScheduler(mock.backend, () => clockNow);

    expect(sched.shouldAllowSpatialCue('continuous')).toBe(true);

    sched.schedule({ text: '紧急提醒', priority: 'EMERGENCY' });
    clockNow = 2200;
    expect(sched.shouldAllowSpatialCue('continuous')).toBe(false);

    mock.triggerDoneForIndex(0);
    clockNow = 2400;
    expect(sched.shouldAllowSpatialCue('continuous')).toBe(true);

    sched.setMuted(true);
    clockNow = 3000;
    expect(sched.shouldAllowSpatialCue('continuous')).toBe(false);
    expect(sched.shouldAllowSpatialCue('finder')).toBe(false);

    sched.setMuted(false);
    clockNow = 4000;
    expect(sched.shouldAllowSpatialCue('finder')).toBe(true);
  });

  it('enforces deduplication cooldown on repeated ambient announcements', () => {
    let clockNow = 1000;
    const mock = createMockBackendWithLateCallbacks();
    const sched = new AudioSpeechScheduler(mock.backend, () => clockNow);

    const r1 = sched.schedule({
      text: '左前方检测到椅子',
      priority: 'AMBIENT',
      dedupeKey: 'chair-left',
      cooldownMs: 8000,
    });
    expect(r1.accepted).toBe(true);
    mock.triggerDoneForIndex(0);

    clockNow = 5000;
    const r2 = sched.schedule({
      text: '左前方检测到椅子',
      priority: 'AMBIENT',
      dedupeKey: 'chair-left',
      cooldownMs: 8000,
    });
    expect(r2.accepted).toBe(false);
    expect(r2.reason).toBe('dropped_cooldown');
  });
});
