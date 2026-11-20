// Unified Audio & Speech Priority Scheduler
// Prevents competing voice broadcasts across modules, enforces priority preemption
// (EMERGENCY > USER_QUERY > AMBIENT), deduplication, cooldowns, and clean screen-reader coexistence.

export type SpeechPriority = 'EMERGENCY' | 'USER_QUERY' | 'AMBIENT';

const PRIORITY_RANK: Record<SpeechPriority, number> = {
  EMERGENCY: 3,
  USER_QUERY: 2,
  AMBIENT: 1,
};

export interface SpeechRequest {
  text: string;
  priority: SpeechPriority;
  dedupeKey?: string;
  cooldownMs?: number;
  rate?: number;
  now?: number; // injectable timestamp for deterministic testing
}

export interface SpeechBackend {
  isAvailable: () => boolean;
  isSpeaking: () => boolean;
  cancel: () => void;
  speak: (text: string, rate: number, onEnd: () => void) => void;
}

export interface ScheduleResult {
  accepted: boolean;
  reason: 'spoken' | 'queued' | 'preempted_and_spoken' | 'dropped_lower_priority' | 'dropped_cooldown' | 'muted' | 'unavailable';
}

function createBrowserSpeechBackend(): SpeechBackend {
  return {
    isAvailable: () => typeof window !== 'undefined' && 'speechSynthesis' in window,
    isSpeaking: () =>
      typeof window !== 'undefined' && 'speechSynthesis' in window
        ? window.speechSynthesis.speaking
        : false,
    cancel: () => {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // Ignore browser speech errors
        }
      }
    },
    speak: (text, rate, onEnd) => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        onEnd();
        return;
      }
      try {
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'zh-CN';
        u.rate = rate;
        u.onend = onEnd;
        u.onerror = onEnd;
        window.speechSynthesis.speak(u);
      } catch {
        onEnd();
      }
    },
  };
}

export class AudioSpeechScheduler {
  private backend: SpeechBackend;
  private currentPriority: SpeechPriority | null = null;
  private activeSpeaking = false;
  private muted = false;
  private lastKeySpokenAt = new Map<string, number>();
  private lastAnyAmbientAt: number | null = null;
  private queue: Array<{ req: SpeechRequest; enqueuedAt: number }> = [];
  private listeners = new Set<(status: { available: boolean; muted: boolean; lastText: string }) => void>();
  private lastText = '';

  constructor(backend?: SpeechBackend) {
    this.backend = backend ?? createBrowserSpeechBackend();
  }

  public isAvailable(): boolean {
    return this.backend.isAvailable();
  }

  public isMuted(): boolean {
    return this.muted;
  }

  public setMuted(muted: boolean) {
    this.muted = muted;
    if (muted) {
      this.clearAll();
    }
    this.notify();
  }

  public clearAll() {
    this.queue = [];
    this.activeSpeaking = false;
    this.currentPriority = null;
    this.backend.cancel();
  }

  public subscribe(
    fn: (status: { available: boolean; muted: boolean; lastText: string }) => void
  ): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify() {
    const snap = { available: this.isAvailable(), muted: this.muted, lastText: this.lastText };
    this.listeners.forEach((fn) => fn(snap));
  }

  public schedule(req: SpeechRequest): ScheduleResult {
    const text = req.text.trim();
    if (!text) return { accepted: false, reason: 'dropped_cooldown' };
    if (this.muted && req.priority !== 'EMERGENCY') {
      return { accepted: false, reason: 'muted' };
    }
    if (!this.backend.isAvailable()) {
      return { accepted: false, reason: 'unavailable' };
    }

    const now = req.now ?? (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const key = req.dedupeKey ?? text;
    const defaultCooldown =
      req.priority === 'EMERGENCY' ? 2500 : req.priority === 'USER_QUERY' ? 0 : 6000;
    const cooldownMs = req.cooldownMs ?? defaultCooldown;

    if (cooldownMs > 0) {
      const prev = this.lastKeySpokenAt.get(key);
      if (prev !== undefined && now - prev < cooldownMs) {
        return { accepted: false, reason: 'dropped_cooldown' };
      }
    }

    // Global ambient rate limit so continuous narration never overwhelms ambient listening
    if (
      req.priority === 'AMBIENT' &&
      this.lastAnyAmbientAt !== null &&
      now - this.lastAnyAmbientAt < 3500
    ) {
      return { accepted: false, reason: 'dropped_cooldown' };
    }

    const isBusy = this.activeSpeaking || this.backend.isSpeaking();
    if (isBusy && this.currentPriority) {
      const incomingRank = PRIORITY_RANK[req.priority];
      const currentRank = PRIORITY_RANK[this.currentPriority];

      if (incomingRank < currentRank) {
        // Never allow ambient chatter to interrupt emergency or user query
        return { accepted: false, reason: 'dropped_lower_priority' };
      }

      if (incomingRank > currentRank || req.priority === 'USER_QUERY') {
        // Higher priority or direct user query preempts current utterance
        this.backend.cancel();
        this.queue = this.queue.filter((item) => PRIORITY_RANK[item.req.priority] >= incomingRank);
        this.speakNow(req, key, now);
        return { accepted: true, reason: 'preempted_and_spoken' };
      }

      // Same priority: queue only if EMERGENCY (keep max 1 pending)
      if (req.priority === 'EMERGENCY') {
        this.queue = [{ req, enqueuedAt: now }];
        return { accepted: true, reason: 'queued' };
      }
      return { accepted: false, reason: 'dropped_lower_priority' };
    }

    this.speakNow(req, key, now);
    return { accepted: true, reason: 'spoken' };
  }

  private speakNow(req: SpeechRequest, key: string, now: number) {
    this.activeSpeaking = true;
    this.currentPriority = req.priority;
    this.lastKeySpokenAt.set(key, now);
    if (req.priority === 'AMBIENT') {
      this.lastAnyAmbientAt = now;
    }
    this.lastText = req.text;
    this.notify();

    this.backend.speak(req.text, req.rate ?? 1.04, () => {
      this.activeSpeaking = false;
      this.currentPriority = null;
      const next = this.queue.shift();
      if (next) {
        this.schedule({ ...next.req, now: Date.now() });
      }
    });
  }
}

export const audioScheduler = new AudioSpeechScheduler();
