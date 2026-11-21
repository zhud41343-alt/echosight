// Unified Accessible Notification Dispatcher & EchoSafe State Transition Coordinator (V6)
// Fixes P0-1 & P0-2:
// - Decouples message validity & ARIA Live delivery from browser TTS (`speechSynthesis`) acceptance,
//   so enabling Screen-Reader Anti-Collision Mode NEVER drops automatic environment updates.
// - Routes critical failures (e.g. video freeze -> UNAVAILABLE) through assertive ARIA + TTS + Haptics
//   with transition deduplication so users are notified even on devices without vibration.

import { AudioSpeechScheduler, SpeechPriority } from './audioScheduler';
import { HapticCueKind, HapticFeedbackController } from './deviceFeedback';
import { EchoSafeSnapshot, SystemModeState } from './echoSafe';

export type AriaLiveChannel = 'polite' | 'assertive' | 'none';

export interface NotificationRequest {
  text: string;
  priority: SpeechPriority;
  dedupeKey?: string;
  cooldownMs?: number;
  rate?: number;
  haptic?: HapticCueKind;
  now?: number;
}

export interface NotificationDispatchResult {
  /** True whenever a valid, non-duplicate message was produced for the user */
  delivered: boolean;
  text: string | null;
  ariaChannel: AriaLiveChannel;
  ttsAttempted: boolean;
  ttsSpoken: boolean;
  hapticTriggered: boolean;
  dropReason?: 'empty' | 'cooldown';
}

export class AccessibleNotificationDispatcher {
  private scheduler: AudioSpeechScheduler;
  private haptics: HapticFeedbackController;
  private clock: () => number;
  private lastKeyAt = new Map<string, number>();
  private lastAmbientAt: number | null = null;
  private listeners = new Set<(res: NotificationDispatchResult) => void>();

  constructor(
    scheduler: AudioSpeechScheduler,
    haptics: HapticFeedbackController,
    clock: () => number = () => (typeof performance !== 'undefined' ? performance.now() : Date.now())
  ) {
    this.scheduler = scheduler;
    this.haptics = haptics;
    this.clock = clock;
  }

  public subscribe(fn: (res: NotificationDispatchResult) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  public reset(): void {
    this.lastKeyAt.clear();
    this.lastAmbientAt = null;
  }

  public dispatch(req: NotificationRequest): NotificationDispatchResult {
    const text = req.text.trim();
    if (!text) {
      return {
        delivered: false,
        text: null,
        ariaChannel: 'none',
        ttsAttempted: false,
        ttsSpoken: false,
        hapticTriggered: false,
        dropReason: 'empty',
      };
    }

    const now = req.now ?? this.clock();
    const key = req.dedupeKey ?? text;
    const defaultCooldown =
      req.priority === 'EMERGENCY' ? 2500 : req.priority === 'USER_QUERY' ? 0 : 6000;
    const cooldownMs = req.cooldownMs ?? defaultCooldown;

    if (cooldownMs > 0) {
      const prev = this.lastKeyAt.get(key);
      if (prev !== undefined && now - prev < cooldownMs) {
        return {
          delivered: false,
          text: null,
          ariaChannel: 'none',
          ttsAttempted: false,
          ttsSpoken: false,
          hapticTriggered: false,
          dropReason: 'cooldown',
        };
      }
    }

    // Prevent high-frequency ambient spam in BOTH TTS and VoiceOver/TalkBack ARIA Live
    if (
      req.priority === 'AMBIENT' &&
      this.lastAmbientAt !== null &&
      now - this.lastAmbientAt < 3500
    ) {
      return {
        delivered: false,
        text: null,
        ariaChannel: 'none',
        ttsAttempted: false,
        ttsSpoken: false,
        hapticTriggered: false,
        dropReason: 'cooldown',
      };
    }

    this.lastKeyAt.set(key, now);
    if (req.priority === 'AMBIENT') {
      this.lastAmbientAt = now;
    }

    // Reserve assertive role="alert" strictly for EMERGENCY; polite role="status" for queries & ambient
    const ariaChannel: AriaLiveChannel = req.priority === 'EMERGENCY' ? 'assertive' : 'polite';

    // Attempt browser TTS (automatically suppressed if screenReaderSafeMode is ON or TTS unavailable)
    const ttsRes = this.scheduler.schedule({
      text,
      priority: req.priority,
      dedupeKey: key,
      cooldownMs: 0, // Deduplication already enforced uniformly above
      rate: req.rate ?? 1.04,
      now,
    });

    const hapticTriggered = req.haptic ? this.haptics.trigger(req.haptic, now) : false;

    const result: NotificationDispatchResult = {
      delivered: true,
      text,
      ariaChannel,
      ttsAttempted: !this.scheduler.isScreenReaderSafeMode(),
      ttsSpoken: ttsRes.accepted,
      hapticTriggered,
    };

    this.listeners.forEach((fn) => fn(result));
    return result;
  }
}

export interface StateTransitionPlan {
  changed: boolean;
  stopSpatialAudio: boolean;
  clearPerceptionData: boolean;
  notification: NotificationRequest | null;
}

/**
 * P0-2: Unified EchoSafe state transition feedback planner.
 * Guarantees that critical transitions (e.g. ACTIVE/DEGRADED -> UNAVAILABLE on video freeze)
 * ALWAYS produce coordinated ARIA + TTS + Haptic feedback without repeating every 1s tick.
 */
export function planEchoSafeTransition(
  prevState: SystemModeState,
  nextSnap: EchoSafeSnapshot
): StateTransitionPlan {
  const nextState = nextSnap.state;
  const changed = prevState !== nextState;

  const stopSpatialAudio = nextState === 'UNAVAILABLE' || nextState === 'IDLE';
  const clearPerceptionData = !nextSnap.canTrustLivePerception;

  if (!changed) {
    return {
      changed: false,
      stopSpatialAudio,
      clearPerceptionData,
      notification: null,
    };
  }

  // Transition into UNAVAILABLE from an active/degraded/initializing live session (e.g. video freeze, track ended, background)
  if (
    nextState === 'UNAVAILABLE' &&
    (prevState === 'ACTIVE' || prevState === 'DEGRADED' || prevState === 'INITIALIZING')
  ) {
    return {
      changed: true,
      stopSpatialAudio: true,
      clearPerceptionData: true,
      notification: {
        text: `实时感知已失效：${nextSnap.userSummary}。已停止环境声场，请勿依赖旧识别结果。`,
        priority: 'EMERGENCY',
        dedupeKey: `transition:UNAVAILABLE:${nextSnap.userSummary}`,
        cooldownMs: 2500,
        haptic: 'unavailable',
      },
    };
  }

  // Transition from ACTIVE to DEGRADED (e.g. model failure, slow inference, suspended audio)
  if (nextState === 'DEGRADED' && prevState === 'ACTIVE') {
    return {
      changed: true,
      stopSpatialAudio: false,
      clearPerceptionData: true,
      notification: {
        text: `系统已降级：${nextSnap.userSummary}`,
        priority: 'EMERGENCY',
        dedupeKey: `transition:DEGRADED:${nextSnap.userSummary}`,
        cooldownMs: 4000,
        haptic: 'warning',
      },
    };
  }

  // Transition into ACTIVE from INITIALIZING or DEGRADED
  if (nextState === 'ACTIVE' && (prevState === 'INITIALIZING' || prevState === 'DEGRADED')) {
    return {
      changed: true,
      stopSpatialAudio: false,
      clearPerceptionData: false,
      notification: {
        text: '实时感知已就绪。方位相对当前摄像头画面，仅作辅助参考。',
        priority: 'USER_QUERY',
        dedupeKey: 'transition:ACTIVE',
        cooldownMs: 3000,
        haptic: 'confirm',
      },
    };
  }

  // Recovery attempt UNAVAILABLE -> INITIALIZING
  if (nextState === 'INITIALIZING' && prevState === 'UNAVAILABLE') {
    return {
      changed: true,
      stopSpatialAudio: false,
      clearPerceptionData: true,
      notification: {
        text: '正在尝试重新连接实时感知画面。',
        priority: 'USER_QUERY',
        dedupeKey: 'transition:RECOVERING',
        cooldownMs: 2000,
      },
    };
  }

  return {
    changed: true,
    stopSpatialAudio,
    clearPerceptionData,
    notification: null,
  };
}
