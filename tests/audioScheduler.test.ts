import { describe, it, expect } from 'vitest';
import { AudioSpeechScheduler, SpeechBackend } from '../src/engine/audioScheduler';

function createMockBackend() {
  const spoken: string[] = [];
  let speaking = false;
  let endCb: (() => void) | null = null;
  let cancelCount = 0;

  const backend: SpeechBackend = {
    isAvailable: () => true,
    isSpeaking: () => speaking,
    cancel: () => {
      cancelCount++;
      speaking = false;
      endCb = null;
    },
    speak: (text, _rate, onEnd) => {
      spoken.push(text);
      speaking = true;
      endCb = () => {
        speaking = false;
        onEnd();
      };
    },
  };

  return {
    backend,
    spoken,
    getCancelCount: () => cancelCount,
    finishCurrent: () => endCb?.(),
  };
}

describe('Unified AudioSpeechScheduler priority & cooldown guarantees', () => {
  it('blocks AMBIENT speech from interrupting an active EMERGENCY alert', () => {
    const mock = createMockBackend();
    const sched = new AudioSpeechScheduler(mock.backend);

    const r1 = sched.schedule({ text: '紧急提示', priority: 'EMERGENCY', now: 1000 });
    expect(r1.accepted).toBe(true);

    const r2 = sched.schedule({ text: '普通环境播报', priority: 'AMBIENT', now: 1200 });
    expect(r2.accepted).toBe(false);
    expect(r2.reason).toBe('dropped_lower_priority');
    expect(mock.spoken).toEqual(['紧急提示']);
  });

  it('allows EMERGENCY to preempt active AMBIENT narration', () => {
    const mock = createMockBackend();
    const sched = new AudioSpeechScheduler(mock.backend);

    sched.schedule({ text: '普通环境播报', priority: 'AMBIENT', now: 1000 });
    const r2 = sched.schedule({ text: '摄像头已断开', priority: 'EMERGENCY', now: 1500 });
    expect(r2.accepted).toBe(true);
    expect(r2.reason).toBe('preempted_and_spoken');
    expect(mock.getCancelCount()).toBe(1);
  });

  it('enforces deduplication cooldown on repeated ambient announcements', () => {
    const mock = createMockBackend();
    const sched = new AudioSpeechScheduler(mock.backend);

    const r1 = sched.schedule({
      text: '左前方检测到椅子',
      priority: 'AMBIENT',
      dedupeKey: 'chair-left',
      cooldownMs: 8000,
      now: 1000,
    });
    expect(r1.accepted).toBe(true);
    mock.finishCurrent();

    const r2 = sched.schedule({
      text: '左前方检测到椅子',
      priority: 'AMBIENT',
      dedupeKey: 'chair-left',
      cooldownMs: 8000,
      now: 5000,
    });
    expect(r2.accepted).toBe(false);
    expect(r2.reason).toBe('dropped_cooldown');
  });
});
