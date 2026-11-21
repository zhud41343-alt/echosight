// Mobile Device Feedback & Lifecycle Helpers (WCAG 2.2 + BLV Haptic & Wake Lock Best Practices)
// - Event-driven haptics (never continuous vibration to avoid tactile fatigue)
// - Screen Wake Lock API with automatic visibilitychange re-acquisition for active camera sessions

export type HapticCueKind = 'confirm' | 'warning' | 'unavailable';

const HAPTIC_PATTERNS: Record<HapticCueKind, number[]> = {
  confirm: [30],
  warning: [60, 50, 60],
  unavailable: [90, 60, 90, 60, 120],
};

export interface VibratorLike {
  vibrate?: (pattern: number | number[]) => boolean;
}

export class HapticFeedbackController {
  private lastFiredAt = new Map<HapticCueKind, number>();
  private enabled = true;
  private vibrator: VibratorLike | null;
  private clock: () => number;

  constructor(
    vibrator?: VibratorLike | null,
    clock: () => number = () => (typeof performance !== 'undefined' ? performance.now() : Date.now())
  ) {
    this.vibrator =
      vibrator !== undefined
        ? vibrator
        : typeof navigator !== 'undefined'
          ? (navigator as unknown as VibratorLike)
          : null;
    this.clock = clock;
  }

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  public isEnabled(): boolean {
    return this.enabled;
  }

  public isSupported(): boolean {
    return Boolean(this.vibrator && typeof this.vibrator.vibrate === 'function');
  }

  public trigger(kind: HapticCueKind, now?: number): boolean {
    if (!this.enabled || !this.isSupported()) return false;
    const t = now ?? this.clock();
    const cooldownMs = kind === 'confirm' ? 200 : kind === 'warning' ? 2500 : 3000;
    const prev = this.lastFiredAt.get(kind);
    if (prev !== undefined && t - prev < cooldownMs) {
      return false;
    }
    this.lastFiredAt.set(kind, t);
    try {
      return Boolean(this.vibrator!.vibrate!(HAPTIC_PATTERNS[kind]));
    } catch {
      return false;
    }
  }
}

export interface WakeLockSentinelLike {
  released?: boolean;
  release: () => Promise<void>;
  addEventListener?: (type: 'release', listener: () => void) => void;
}

export interface WakeLockNavigatorLike {
  wakeLock?: {
    request: (type: 'screen') => Promise<WakeLockSentinelLike>;
  };
}

/**
 * Manages the Screen Wake Lock API so the smartphone screen does not auto-dim/lock
 * during active blind-assist camera sessions, with:
 * - Monotonic generation guard: if stop/release happens while request('screen') is pending,
 *   the late-resolving sentinel is IMMEDIATELY released (zero leaked wake locks).
 * - Single-flight deduplication: concurrent acquire() calls share the same request.
 * - Automatic re-acquire on visibilitychange when session is still active.
 */
export class ScreenWakeLockController {
  private sentinel: WakeLockSentinelLike | null = null;
  private nav: WakeLockNavigatorLike | null;
  private generation = 0;
  private inFlight: Promise<boolean> | null = null;

  constructor(nav?: WakeLockNavigatorLike | null) {
    this.nav =
      nav !== undefined
        ? nav
        : typeof navigator !== 'undefined'
          ? (navigator as unknown as WakeLockNavigatorLike)
          : null;
  }

  public isSupported(): boolean {
    return Boolean(this.nav?.wakeLock && typeof this.nav.wakeLock.request === 'function');
  }

  public get isHeld(): boolean {
    return this.sentinel !== null && !this.sentinel.released;
  }

  public async acquire(isSessionStillValid?: () => boolean): Promise<boolean> {
    if (!this.isSupported()) return false;
    if (isSessionStillValid && !isSessionStillValid()) return false;
    if (this.isHeld) return true;
    if (this.inFlight) return this.inFlight;

    const myGen = ++this.generation;
    const task = (async (): Promise<boolean> => {
      try {
        const sentinel = await this.nav!.wakeLock!.request('screen');
        // P1-1: If release() was called or session invalidated while awaiting, release immediately!
        if (this.generation !== myGen || (isSessionStillValid && !isSessionStillValid())) {
          try {
            await sentinel.release();
          } catch {
            // Ignore release errors
          }
          return false;
        }
        this.sentinel = sentinel;
        sentinel.addEventListener?.('release', () => {
          if (this.sentinel === sentinel) {
            this.sentinel = null;
          }
        });
        return true;
      } catch {
        // Gracefully ignore NotAllowedError (e.g. battery saver / background tab)
        if (this.generation === myGen) {
          this.sentinel = null;
        }
        return false;
      } finally {
        if (this.generation === myGen) {
          this.inFlight = null;
        }
      }
    })();

    this.inFlight = task;
    return task;
  }

  public async release(): Promise<void> {
    // Invalidate any pending acquire() first so late-resolving sentinels self-release
    this.generation += 1;
    this.inFlight = null;
    const s = this.sentinel;
    this.sentinel = null;
    if (s) {
      try {
        await s.release();
      } catch {
        // Ignore already-released sentinel
      }
    }
  }

  public async handleVisibilityChange(
    isVisible: boolean,
    shouldHold: boolean,
    isSessionStillValid?: () => boolean
  ): Promise<boolean> {
    if (!isVisible) {
      await this.release();
      return false;
    }
    if (shouldHold) {
      return this.acquire(isSessionStillValid);
    }
    return false;
  }
}

export const hapticFeedback = new HapticFeedbackController();
