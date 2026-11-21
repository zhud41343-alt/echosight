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
 * during active blind-assist camera sessions, with automatic re-acquire on visibilitychange.
 */
export class ScreenWakeLockController {
  private sentinel: WakeLockSentinelLike | null = null;
  private nav: WakeLockNavigatorLike | null;

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

  public async acquire(): Promise<boolean> {
    if (!this.isSupported()) return false;
    if (this.isHeld) return true;
    try {
      const sentinel = await this.nav!.wakeLock!.request('screen');
      this.sentinel = sentinel;
      sentinel.addEventListener?.('release', () => {
        if (this.sentinel === sentinel) {
          this.sentinel = null;
        }
      });
      return true;
    } catch {
      // Gracefully ignore NotAllowedError (e.g. battery saver / background tab)
      this.sentinel = null;
      return false;
    }
  }

  public async release(): Promise<void> {
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

  public async handleVisibilityChange(isVisible: boolean, shouldHold: boolean): Promise<boolean> {
    if (!isVisible) {
      await this.release();
      return false;
    }
    if (shouldHold) {
      return this.acquire();
    }
    return false;
  }
}

export const hapticFeedback = new HapticFeedbackController();
