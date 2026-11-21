// EchoSafe V5 — Unified Perception Reliability State Machine, Session Guard, Frame Tracker & Packet Validator
// Fixes:
// - P0-1 (V3): Never treats lastFrameAt === null or lastInferenceAt === null as fresh/ACTIVE.
// - P0-2 (V3): SessionGuard generation tokens prevent stale async camera/model/inference writebacks.
// - P0-4 (V3/V4): PerceptionPacket + isPerceptionPacketUsable enforce uniform validity checks everywhere.
// - P0-1 (V5): FrameTracker assigns monotonic frameId to each real video frame so AI never infers the same frame twice.
// - P0-2 (V5): canEmitContinuousSpatialAudio blocks Sobel/HRTF continuous audio immediately on freeze or UNAVAILABLE.

export type SystemModeState =
  | 'IDLE'
  | 'INITIALIZING'
  | 'ACTIVE'
  | 'DEGRADED'
  | 'UNAVAILABLE'
  | 'DEMO';

export const STATE_LABELS_ZH: Record<SystemModeState, string> = {
  IDLE: '未启动',
  INITIALIZING: '正在准备',
  ACTIVE: '实时感知运行中',
  DEGRADED: '降级运行',
  UNAVAILABLE: '当前不可用',
  DEMO: '模拟演示模式',
};

export type PerceptionStatus = 'DETECTED' | 'UNKNOWN' | 'DEGRADED' | 'UNAVAILABLE';

export type DataSourceKind = 'REAL_CAMERA' | 'UPLOADED_IMAGE' | 'SIMULATED_SCENE' | 'UNAVAILABLE';

export interface EchoSafeSignals {
  requestedLive: boolean;
  isDemoMode: boolean;
  initializing: boolean;
  sessionStartedAt: number | null;
  cameraPermission: 'unknown' | 'granted' | 'denied' | 'unavailable';
  cameraTrackLive: boolean;
  /** Timestamp of the last REAL video frame advance (NOT repeated drawImage) */
  lastFrameAt: number | null;
  modelState: 'idle' | 'loading' | 'ready' | 'failed';
  /** Timestamp of the last SUCCESSFUL AI inference on a real frame */
  lastInferenceAt: number | null;
  lastInferenceDurationMs: number | null;
  consecutiveInferenceFailures: number;
  pageHidden: boolean;
  audioState: 'running' | 'suspended' | 'unavailable';
  microphoneAvailable: boolean;
}

export interface EchoSafeSnapshot {
  state: SystemModeState;
  stateLabelZh: string;
  perceptionStatus: PerceptionStatus;
  dataSource: DataSourceKind;
  reasons: string[];
  userSummary: string;
  canTrustLivePerception: boolean;
  frameFresh: boolean;
  inferenceFresh: boolean;
  evaluatedAt: number;
}

export interface EchoSafeThresholds {
  firstFrameGraceMs: number;
  frameStaleMs: number;
  inferenceStaleMs: number;
  inferenceTimeoutMs: number;
  maxConsecutiveFailures: number;
}

export const DEFAULT_ECHOSAFE_THRESHOLDS: EchoSafeThresholds = {
  firstFrameGraceMs: 4000,
  frameStaleMs: 2500,
  inferenceStaleMs: 4500,
  inferenceTimeoutMs: 3500,
  maxConsecutiveFailures: 3,
};

export function createInitialSignals(): EchoSafeSignals {
  return {
    requestedLive: false,
    isDemoMode: false,
    initializing: false,
    sessionStartedAt: null,
    cameraPermission: 'unknown',
    cameraTrackLive: false,
    lastFrameAt: null,
    modelState: 'idle',
    lastInferenceAt: null,
    lastInferenceDurationMs: null,
    consecutiveInferenceFailures: 0,
    pageHidden: typeof document !== 'undefined' ? document.hidden : false,
    audioState: 'suspended',
    microphoneAvailable: false,
  };
}

/**
 * Pure, conservative EchoSafe evaluator (V3).
 * Guarantees:
 * 1. lastFrameAt === null NEVER enters ACTIVE.
 * 2. lastInferenceAt === null NEVER enters ACTIVE (enters DEGRADED while awaiting first inference).
 * 3. Frozen video (no real frame advance > frameStaleMs) immediately enters UNAVAILABLE.
 */
export function evaluateEchoSafe(
  signals: EchoSafeSignals,
  now: number,
  thresholds: EchoSafeThresholds = DEFAULT_ECHOSAFE_THRESHOLDS
): EchoSafeSnapshot {
  const reasons: string[] = [];

  if (signals.isDemoMode) {
    return {
      state: 'DEMO',
      stateLabelZh: STATE_LABELS_ZH.DEMO,
      perceptionStatus: 'UNKNOWN',
      dataSource: 'SIMULATED_SCENE',
      reasons: ['当前处于明确标记的模拟演示模式，输出不代表真实周围环境。'],
      userSummary: '模拟演示模式（非真实摄像头画面）',
      canTrustLivePerception: false,
      frameFresh: false,
      inferenceFresh: false,
      evaluatedAt: now,
    };
  }

  if (!signals.requestedLive && !signals.initializing) {
    return {
      state: 'IDLE',
      stateLabelZh: STATE_LABELS_ZH.IDLE,
      perceptionStatus: 'UNAVAILABLE',
      dataSource: 'UNAVAILABLE',
      reasons: ['实时感知尚未启动。'],
      userSummary: '未启动',
      canTrustLivePerception: false,
      frameFresh: false,
      inferenceFresh: false,
      evaluatedAt: now,
    };
  }

  if (signals.initializing) {
    return {
      state: 'INITIALIZING',
      stateLabelZh: STATE_LABELS_ZH.INITIALIZING,
      perceptionStatus: 'UNKNOWN',
      dataSource: 'UNAVAILABLE',
      reasons: ['正在申请设备权限并加载端侧模型。'],
      userSummary: '正在准备权限与模型',
      canTrustLivePerception: false,
      frameFresh: false,
      inferenceFresh: false,
      evaluatedAt: now,
    };
  }

  // Hard unavailability checks
  if (signals.pageHidden) {
    reasons.push('页面已切入后台，浏览器无法保证持续摄像头采集。');
  }
  if (signals.cameraPermission === 'denied') {
    reasons.push('摄像头权限被拒绝，无法采集真实环境画面。');
  } else if (signals.cameraPermission === 'unavailable') {
    reasons.push('当前设备未检测到可用摄像头。');
  } else if (!signals.cameraTrackLive) {
    reasons.push('摄像头视频轨道已中断或停止。');
  }

  if (reasons.length > 0) {
    return {
      state: 'UNAVAILABLE',
      stateLabelZh: STATE_LABELS_ZH.UNAVAILABLE,
      perceptionStatus: 'UNAVAILABLE',
      dataSource: 'UNAVAILABLE',
      reasons,
      userSummary: reasons[0],
      canTrustLivePerception: false,
      frameFresh: false,
      inferenceFresh: false,
      evaluatedAt: now,
    };
  }

  // P0-1: Strict frame freshness — null lastFrameAt is NEVER fresh
  if (signals.lastFrameAt === null) {
    const waitedMs =
      signals.sessionStartedAt !== null ? Math.max(0, now - signals.sessionStartedAt) : Infinity;
    if (waitedMs <= thresholds.firstFrameGraceMs) {
      return {
        state: 'INITIALIZING',
        stateLabelZh: STATE_LABELS_ZH.INITIALIZING,
        perceptionStatus: 'UNKNOWN',
        dataSource: 'UNAVAILABLE',
        reasons: ['等待摄像头首帧画面到达。'],
        userSummary: '正在等待摄像头首帧画面',
        canTrustLivePerception: false,
        frameFresh: false,
        inferenceFresh: false,
        evaluatedAt: now,
      };
    }
    return {
      state: 'UNAVAILABLE',
      stateLabelZh: STATE_LABELS_ZH.UNAVAILABLE,
      perceptionStatus: 'UNAVAILABLE',
      dataSource: 'UNAVAILABLE',
      reasons: ['摄像头已打开但超时未收到有效视频帧。'],
      userSummary: '未收到有效视频帧，实时感知不可用',
      canTrustLivePerception: false,
      frameFresh: false,
      inferenceFresh: false,
      evaluatedAt: now,
    };
  }

  const frameAge = now - signals.lastFrameAt;
  const frameFresh = frameAge >= 0 && frameAge <= thresholds.frameStaleMs;
  if (!frameFresh) {
    return {
      state: 'UNAVAILABLE',
      stateLabelZh: STATE_LABELS_ZH.UNAVAILABLE,
      perceptionStatus: 'UNAVAILABLE',
      dataSource: 'UNAVAILABLE',
      reasons: ['视频画面已冻结或停止更新（数据过期）。'],
      userSummary: '视频画面已冻结，实时感知不可用',
      canTrustLivePerception: false,
      frameFresh: false,
      inferenceFresh: false,
      evaluatedAt: now,
    };
  }

  // Model & inference freshness checks (null lastInferenceAt is NEVER treated as fresh)
  const inferenceFresh =
    signals.modelState === 'ready' &&
    signals.lastInferenceAt !== null &&
    now - signals.lastInferenceAt >= 0 &&
    now - signals.lastInferenceAt <= thresholds.inferenceStaleMs;

  if (signals.modelState === 'failed') {
    reasons.push('端侧物体识别模型加载失败，当前无法识别具体物体。');
  } else if (signals.modelState === 'loading' || signals.modelState === 'idle') {
    reasons.push('端侧物体识别模型尚未就绪。');
  } else if (signals.lastInferenceAt === null) {
    reasons.push('模型已就绪，正在等待首次视觉推理完成。');
  } else if (!inferenceFresh) {
    reasons.push('视觉识别结果已过期，暂不输出确定性描述。');
  }

  if (signals.consecutiveInferenceFailures >= thresholds.maxConsecutiveFailures) {
    reasons.push('视觉推理连续失败，识别结果已降级。');
  }
  if (
    signals.lastInferenceDurationMs !== null &&
    signals.lastInferenceDurationMs > thresholds.inferenceTimeoutMs
  ) {
    reasons.push(`单次推理耗时过长（${Math.round(signals.lastInferenceDurationMs)}ms）。`);
  }
  if (signals.audioState !== 'running') {
    reasons.push('浏览器音频未激活或已暂停，可能听不到提示音。');
  }

  if (reasons.length > 0) {
    return {
      state: 'DEGRADED',
      stateLabelZh: STATE_LABELS_ZH.DEGRADED,
      perceptionStatus: 'DEGRADED',
      dataSource: 'REAL_CAMERA',
      reasons,
      userSummary: `降级运行：${reasons[0]}`,
      canTrustLivePerception: false,
      frameFresh: true,
      inferenceFresh,
      evaluatedAt: now,
    };
  }

  return {
    state: 'ACTIVE',
    stateLabelZh: STATE_LABELS_ZH.ACTIVE,
    perceptionStatus: 'UNKNOWN',
    dataSource: 'REAL_CAMERA',
    reasons: [],
    userSummary: '实时感知运行中（仅作辅助参考，不可替代盲杖与现场判断）',
    canTrustLivePerception: true,
    frameFresh: true,
    inferenceFresh: true,
    evaluatedAt: now,
  };
}

/**
 * P0-2: Monotonic session generation guard.
 * Every start increments sessionId; stop/unmount/mode-switch invalidates it so
 * late-resolving promises (camera permission, model load, inference) can NEVER write back.
 */
export class SessionGuard {
  private current = 0;
  private active = false;

  public begin(): number {
    this.current += 1;
    this.active = true;
    return this.current;
  }

  public invalidate(): void {
    this.current += 1;
    this.active = false;
  }

  public isValid(sessionId: number): boolean {
    return this.active && sessionId === this.current;
  }

  public get currentId(): number {
    return this.current;
  }

  public get isActive(): boolean {
    return this.active;
  }
}

/**
 * P0-1 (V5): Monotonic real video frame tracker.
 * Guarantees each distinct camera frame gets a unique increasing frameId,
 * preventing AI inference from running twice on the same video frame when
 * camera FPS < inference scheduling cadence.
 */
export interface FrameToken {
  frameId: number;
  capturedAt: number;
}

export class FrameTracker {
  private latestFrameId = 0;
  private latestFrameCapturedAt: number | null = null;
  private lastInferredFrameId = 0;

  public recordNewFrame(capturedAt: number): number {
    this.latestFrameId += 1;
    this.latestFrameCapturedAt = capturedAt;
    return this.latestFrameId;
  }

  public claimNextFrameForInference(
    now: number,
    maxFrameAgeMs = DEFAULT_ECHOSAFE_THRESHOLDS.frameStaleMs
  ): FrameToken | null {
    if (this.latestFrameId === 0 || this.latestFrameCapturedAt === null) {
      return null;
    }
    if (this.latestFrameId <= this.lastInferredFrameId) {
      return null;
    }
    const age = now - this.latestFrameCapturedAt;
    if (age < -50 || age > maxFrameAgeMs) {
      return null;
    }
    this.lastInferredFrameId = this.latestFrameId;
    return {
      frameId: this.latestFrameId,
      capturedAt: this.latestFrameCapturedAt,
    };
  }

  public getSnapshot(): {
    latestFrameId: number;
    latestFrameCapturedAt: number | null;
    lastInferredFrameId: number;
  } {
    return {
      latestFrameId: this.latestFrameId,
      latestFrameCapturedAt: this.latestFrameCapturedAt,
      lastInferredFrameId: this.lastInferredFrameId,
    };
  }

  public reset(): void {
    this.latestFrameId = 0;
    this.latestFrameCapturedAt = null;
    this.lastInferredFrameId = 0;
  }
}

/**
 * P0-2 (V5): Strict gate for continuous spatial audio sonification.
 * Ensures continuous HRTF/Sobel audio NEVER plays when video is frozen,
 * camera track ended, page is hidden, audio context is not running,
 * or EchoSafe is UNAVAILABLE/IDLE.
 */
export function canEmitContinuousSpatialAudio(
  signals: EchoSafeSignals,
  now: number,
  thresholds: EchoSafeThresholds = DEFAULT_ECHOSAFE_THRESHOLDS
): { allowed: boolean; reason?: string } {
  if (signals.isDemoMode) {
    return { allowed: false, reason: '模拟模式不走实时连续声场' };
  }
  if (!signals.requestedLive || signals.initializing) {
    return { allowed: false, reason: '实时感知未就绪' };
  }
  if (signals.pageHidden) {
    return { allowed: false, reason: '页面已进入后台' };
  }
  if (signals.cameraPermission !== 'granted' || !signals.cameraTrackLive) {
    return { allowed: false, reason: '摄像头不可用或已断开' };
  }
  if (signals.audioState !== 'running') {
    return { allowed: false, reason: '音频上下文未运行' };
  }
  if (signals.lastFrameAt === null) {
    return { allowed: false, reason: '尚未收到真实视频帧' };
  }
  const frameAge = now - signals.lastFrameAt;
  if (frameAge < -50 || frameAge > thresholds.frameStaleMs) {
    return { allowed: false, reason: '视频帧已过期或冻结' };
  }
  const snap = evaluateEchoSafe(signals, now, thresholds);
  if (snap.state === 'UNAVAILABLE' || snap.state === 'IDLE' || snap.state === 'INITIALIZING') {
    return { allowed: false, reason: snap.userSummary };
  }
  return { allowed: true };
}

/**
 * P0-4 & P1-2 (V6): Unified perception result packet with strict runtime contract.
 * Real camera packets MUST carry a valid positive integer frameId from FrameTracker.
 */
export interface PerceptionPacket<T> {
  dataSource: DataSourceKind;
  sessionId: number;
  frameId: number;
  capturedAt: number;
  inferredAt: number;
  expiresAt: number;
  safeState: SystemModeState;
  detections: T[];
}

export function isPerceptionPacketUsable<T>(
  packet: PerceptionPacket<T> | null,
  expectedSessionId: number,
  currentSafeState: SystemModeState,
  now: number,
  maxFrameAgeMs = 3000,
  minFrameId = 1
): boolean {
  if (!packet) return false;
  if (packet.dataSource !== 'REAL_CAMERA') return false;
  if (packet.sessionId !== expectedSessionId) return false;
  if (packet.safeState !== 'ACTIVE' || currentSafeState !== 'ACTIVE') return false;

  // P1-2 (V6): Runtime enforcement of required positive integer frameId & monotonic floor
  if (
    typeof packet.frameId !== 'number' ||
    !Number.isInteger(packet.frameId) ||
    packet.frameId <= 0 ||
    packet.frameId < minFrameId
  ) {
    return false;
  }

  const { capturedAt, inferredAt, expiresAt } = packet;
  if (
    !Number.isFinite(capturedAt) ||
    !Number.isFinite(inferredAt) ||
    !Number.isFinite(expiresAt) ||
    !Number.isFinite(now)
  ) {
    return false;
  }

  // Enforce monotonic temporal ordering: capturedAt <= inferredAt <= expiresAt
  if (capturedAt > inferredAt || inferredAt > expiresAt) return false;

  // Reject future timestamps (allow tiny 50ms clock jitter)
  if (capturedAt > now + 50 || inferredAt > now + 50) return false;

  // Reject stale raw camera frame AND expired inference packet
  if (now - capturedAt > maxFrameAgeMs) return false;
  if (now > expiresAt) return false;

  return true;
}

export class SingleFlightRunner {
  private inFlight = false;

  public get isRunning(): boolean {
    return this.inFlight;
  }

  public async run<T>(task: () => Promise<T>): Promise<{ skipped: boolean; value?: T }> {
    if (this.inFlight) {
      return { skipped: true };
    }
    this.inFlight = true;
    try {
      const value = await task();
      return { skipped: false, value };
    } finally {
      this.inFlight = false;
    }
  }
}

export interface PerformanceTelemetry {
  visionFps: number;
  aiFps: number;
  avgInferenceMs: number;
  p95InferenceMs: number;
  maxInferenceMs: number;
  lastInferenceMs: number | null;
  avgE2eMs: number;
  p95E2eMs: number;
  modelLoadMs: number | null;
  mediaInterruptions: number;
  memoryUsedMb: number | null;
}

export class RuntimeMetricsTracker {
  private visionTimestamps: number[] = [];
  private aiTimestamps: number[] = [];
  private inferenceDurations: number[] = [];
  private e2eDurations: number[] = [];
  private modelLoadMs: number | null = null;
  private mediaInterruptions = 0;

  public recordVisionFrame(now: number) {
    this.visionTimestamps.push(now);
    const cutoff = now - 2000;
    while (this.visionTimestamps.length > 0 && this.visionTimestamps[0] < cutoff) {
      this.visionTimestamps.shift();
    }
  }

  public recordInference(now: number, durationMs: number, e2eMs?: number) {
    this.aiTimestamps.push(now);
    const cutoff = now - 5000;
    while (this.aiTimestamps.length > 0 && this.aiTimestamps[0] < cutoff) {
      this.aiTimestamps.shift();
    }
    this.inferenceDurations.push(durationMs);
    if (this.inferenceDurations.length > 50) this.inferenceDurations.shift();
    if (e2eMs !== undefined && Number.isFinite(e2eMs) && e2eMs >= 0) {
      this.e2eDurations.push(e2eMs);
      if (this.e2eDurations.length > 50) this.e2eDurations.shift();
    }
  }

  public setModelLoadMs(ms: number) {
    this.modelLoadMs = Math.round(ms);
  }

  public recordMediaInterruption() {
    this.mediaInterruptions += 1;
  }

  public getSnapshot(now: number): PerformanceTelemetry {
    const vCutoff = now - 2000;
    while (this.visionTimestamps.length > 0 && this.visionTimestamps[0] < vCutoff) {
      this.visionTimestamps.shift();
    }
    const aCutoff = now - 5000;
    while (this.aiTimestamps.length > 0 && this.aiTimestamps[0] < aCutoff) {
      this.aiTimestamps.shift();
    }

    const stats = (arr: number[]) => {
      if (arr.length === 0) return { avg: 0, p95: 0, max: 0 };
      const sum = arr.reduce((a, b) => a + b, 0);
      const sorted = [...arr].sort((a, b) => a - b);
      const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
      return {
        avg: Math.round(sum / arr.length),
        p95: Math.round(sorted[idx]),
        max: Math.round(sorted[sorted.length - 1]),
      };
    };

    const inf = stats(this.inferenceDurations);
    const e2e = stats(this.e2eDurations);

    let memoryUsedMb: number | null = null;
    const perfWithMem = performance as unknown as { memory?: { usedJSHeapSize: number } };
    if (perfWithMem?.memory?.usedJSHeapSize) {
      memoryUsedMb = Math.round(perfWithMem.memory.usedJSHeapSize / (1024 * 1024));
    }

    return {
      visionFps: Math.round((this.visionTimestamps.length / 2) * 10) / 10,
      aiFps: Math.round((this.aiTimestamps.length / 5) * 10) / 10,
      avgInferenceMs: inf.avg,
      p95InferenceMs: inf.p95,
      maxInferenceMs: inf.max,
      lastInferenceMs:
        this.inferenceDurations.length > 0
          ? Math.round(this.inferenceDurations[this.inferenceDurations.length - 1])
          : null,
      avgE2eMs: e2e.avg,
      p95E2eMs: e2e.p95,
      modelLoadMs: this.modelLoadMs,
      mediaInterruptions: this.mediaInterruptions,
      memoryUsedMb,
    };
  }
}
