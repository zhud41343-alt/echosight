// Unified Audio & Speech Priority Scheduler (V3 — Race-Free)
// Fixes P0-3:
// - Unique monotonic playbackId per utterance so late onend/onerror callbacks from cancelled
//   utterances can NEVER clobber a newer utterance's activeSpeaking / currentPriority state.
// - Unified monotonic clock (no mixing of performance.now() and Date.now()).
// - Coordinates spatial audio cues (finder / low-freq / continuous) with active speech & mute state.

export type SpeechPriority = 'EMERGENCY' | 'USER_QUERY' | 'AMBIENT';
export type SpatialCueKind = 'continuous' | 'finder' | 'low_freq';

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
  ttlMs?: number; // Drop queued item if older than ttlMs
  now?: number; // Injectable monotonic timestamp for deterministic testing
}

export interface SpeechBackend {
  isAvailable: () => boolean;
  isSpeaking: () => boolean;
  cancel: () => void;
  speak: (text: string, rate: number, onDone: () => void) => void;
}

export interface ScheduleResult {
  accepted: boolean;
  playbackId?: number;
  reason:
    | 'spoken'
    | 'queued'
    | 'preempted_and_spoken'
    | 'dropped_lower_priority'
    | 'dropped_cooldown'
    | 'muted'
    | 'unavailable';
}

function defaultMonotonicClock(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
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
    speak: (text, rate, onDone) => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        onDone();
        return;
      }
      try {
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'zh-CN';
        u.rate = rate;
        let finished = false;
        const finishOnce = () => {
          if (finished) return;
          finished = true;
          onDone();
        };
        u.onend = finishOnce;
        u.onerror = finishOnce;
        window.speechSynthesis.speak(u);
      } catch {
        onDone();
      }
    },
  };
}

export class AudioSpeechScheduler {
  private backend: SpeechBackend;
  private clock: () => number;
  private nextPlaybackId = 0;
  private activePlaybackId: number | null = null;
  private currentPriority: SpeechPriority | null = null;
  private muted = false;
  private audioOutputReady = true;
  private lastKeySpokenAt = new Map<string, number>();
  private lastAnyAmbientAt: number | null = null;
  private lastSpatialCueAt = new Map<SpatialCueKind, number>();
  private queue: Array<{ req: SpeechRequest; enqueuedAt: number }> = [];
  private listeners = new Set<
    (status: { available: boolean; muted: boolean; audioOutputReady: boolean; lastText: string }) => void
  >();
  private lastText = '';

  constructor(backend?: SpeechBackend, clock: () => number = defaultMonotonicClock) {
    this.backend = backend ?? createBrowserSpeechBackend();
    this.clock = clock;
  }

  public isAvailable(): boolean {
    return this.backend.isAvailable();
  }

  public isMuted(): boolean {
    return this.muted;
  }

  public setAudioOutputReady(ready: boolean) {
    if (this.audioOutputReady !== ready) {
      this.audioOutputReady = ready;
      this.notify();
    }
  }

  public getCurrentPriority(): SpeechPriority | null {
    return this.activePlaybackId !== null ? this.currentPriority : null;
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
    // Advance activePlaybackId to null so any late callbacks from cancelled speech are ignored
    this.activePlaybackId = null;
    this.currentPriority = null;
    this.lastSpatialCueAt.clear();
    this.backend.cancel();
  }

  public subscribe(
    fn: (status: { available: boolean; muted: boolean; audioOutputReady: boolean; lastText: string }) => void
  ): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify() {
    const snap = {
      available: this.isAvailable(),
      muted: this.muted,
      audioOutputReady: this.audioOutputReady,
      lastText: this.lastText,
    };
    this.listeners.forEach((fn) => fn(snap));
  }

  /**
   * P0-3 Unified Output Coordination:
   * ALL spatial cues (continuous, finder, low_freq) yield to EMERGENCY or USER_QUERY speech,
   * and are blocked when muted or when audio output context is suspended/unavailable.
   */
  public shouldAllowSpatialCue(kind: SpatialCueKind, now?: number): boolean {
    if (this.muted || !this.audioOutputReady) return false;
    const t = now ?? this.clock();

    // No spatial cue may compete with active EMERGENCY or USER_QUERY speech
    if (
      this.activePlaybackId !== null &&
      (this.currentPriority === 'EMERGENCY' || this.currentPriority === 'USER_QUERY')
    ) {
      return false;
    }

    // Continuous scan also yields to active AMBIENT speech so narration stays clear
    if (kind === 'continuous' && this.activePlaybackId !== null) {
      return false;
    }

    const minGap = kind === 'continuous' ? 80 : kind === 'finder' ? 850 : 1400;
    const prev = this.lastSpatialCueAt.get(kind);
    if (prev !== undefined && t - prev < minGap) {
      return false;
    }
    this.lastSpatialCueAt.set(kind, t);
    return true;
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

    const now = req.now ?? this.clock();
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

    if (
      req.priority === 'AMBIENT' &&
      this.lastAnyAmbientAt !== null &&
      now - this.lastAnyAmbientAt < 3500
    ) {
      return { accepted: false, reason: 'dropped_cooldown' };
    }

    const isBusy = this.activePlaybackId !== null;
    if (isBusy && this.currentPriority) {
      const incomingRank = PRIORITY_RANK[req.priority];
      const currentRank = PRIORITY_RANK[this.currentPriority];

      if (incomingRank < currentRank) {
        return { accepted: false, reason: 'dropped_lower_priority' };
      }

      if (incomingRank > currentRank || req.priority === 'USER_QUERY') {
        // Invalidate current playbackId BEFORE cancel so late onend/onerror from old speech is ignored
        this.activePlaybackId = null;
        this.currentPriority = null;
        this.backend.cancel();
        this.queue = this.queue.filter((item) => PRIORITY_RANK[item.req.priority] >= incomingRank);
        const id = this.speakNow(req, key, now);
        return { accepted: true, playbackId: id, reason: 'preempted_and_spoken' };
      }

      if (req.priority === 'EMERGENCY') {
        this.queue = [{ req, enqueuedAt: now }];
        return { accepted: true, reason: 'queued' };
      }
      return { accepted: false, reason: 'dropped_lower_priority' };
    }

    const id = this.speakNow(req, key, now);
    return { accepted: true, playbackId: id, reason: 'spoken' };
  }

  private speakNow(req: SpeechRequest, key: string, now: number): number {
    const id = ++this.nextPlaybackId;
    this.activePlaybackId = id;
    this.currentPriority = req.priority;
    this.lastKeySpokenAt.set(key, now);
    if (req.priority === 'AMBIENT') {
      this.lastAnyAmbientAt = now;
    }
    this.lastText = req.text;
    this.notify();

    this.backend.speak(req.text, req.rate ?? 1.04, () => {
      // P0-3 Race Guard: ignore stale onend/onerror callbacks from previously cancelled utterances
      if (this.activePlaybackId !== id) {
        return;
      }
      this.activePlaybackId = null;
      this.currentPriority = null;

      const tNow = this.clock();
      while (this.queue.length > 0) {
        const next = this.queue.shift()!;
        const ttl = next.req.ttlMs ?? 4000;
        if (tNow - next.enqueuedAt <= ttl) {
          this.schedule({ ...next.req, now: tNow });
          break;
        }
      }
    });

    return id;
  }
}

export const audioScheduler = new AudioSpeechScheduler();
