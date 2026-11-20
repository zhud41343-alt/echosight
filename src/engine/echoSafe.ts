// EchoSafe — Unified Perception Reliability State Machine & Fail-Safe Evaluator
// Ensures strict separation between live sensor inputs and demo data, tracks sensor/model/page
// liveness & staleness, prevents concurrent inference pile-ups, and enforces fail-safe reporting
// (never inferring "safe / clear / flat ground" from empty detections).

export type SystemModeState =
  | 'IDLE'
  | 'INITIALIZING'
  | 'ACTIVE'
  | 'DEGRADED'
  | 'UNAVAILABLE'
  | 'DEMO';

export type PerceptionStatus = 'DETECTED' | 'UNKNOWN' | 'DEGRADED' | 'UNAVAILABLE';

export type DataSourceKind = 'live_camera' | 'uploaded_image' | 'demo_preset' | 'none';

export interface EchoSafeSignals {
  requestedLive: boolean;
  isDemoMode: boolean;
  initializing: boolean;
  cameraPermission: 'unknown' | 'granted' | 'denied' | 'unavailable';
  cameraTrackLive: boolean;
  lastFrameAt: number | null; // ms timestamp
  modelState: 'idle' | 'loading' | 'ready' | 'failed';
  lastInferenceAt: number | null;
  lastInferenceDurationMs: number | null;
  consecutiveInferenceFailures: number;
  pageHidden: boolean;
  audioState: 'running' | 'suspended' | 'unavailable';
  microphoneAvailable: boolean;
}

export interface EchoSafeSnapshot {
  state: SystemModeState;
  perceptionStatus: PerceptionStatus;
  dataSource: DataSourceKind;
  reasons: string[];
  userSummary: string;
  canTrustLivePerception: boolean;
  frameStale: boolean;
  inferenceStale: boolean;
  evaluatedAt: number;
}

export interface EchoSafeThresholds {
  frameStaleMs: number;
  inferenceStaleMs: number;
  inferenceTimeoutMs: number;
  maxConsecutiveFailures: number;
}

export const DEFAULT_ECHOSAFE_THRESHOLDS: EchoSafeThresholds = {
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
 * Pure, deterministic evaluator for EchoSafe reliability state transitions.
 * Never treats `active === true` alone as healthy.
 */
export function evaluateEchoSafe(
  signals: EchoSafeSignals,
  now: number,
  thresholds: EchoSafeThresholds = DEFAULT_ECHOSAFE_THRESHOLDS
): EchoSafeSnapshot {
  const reasons: string[] = [];

  // 1. Explicit Demo Mode (strictly isolated from live perception)
  if (signals.isDemoMode) {
    return {
      state: 'DEMO',
      perceptionStatus: 'UNKNOWN',
      dataSource: 'demo_preset',
      reasons: ['当前处于明确标记的模拟演示模式，输出不代表真实周围环境。'],
      userSummary: '演示模式（模拟画面，非真实环境）',
      canTrustLivePerception: false,
      frameStale: false,
      inferenceStale: false,
      evaluatedAt: now,
    };
  }

  // 2. Idle
  if (!signals.requestedLive && !signals.initializing) {
    return {
      state: 'IDLE',
      perceptionStatus: 'UNAVAILABLE',
      dataSource: 'none',
      reasons: ['感知系统尚未启动。'],
      userSummary: '未启动',
      canTrustLivePerception: false,
      frameStale: false,
      inferenceStale: false,
      evaluatedAt: now,
    };
  }

  // 3. Initializing
  if (signals.initializing) {
    return {
      state: 'INITIALIZING',
      perceptionStatus: 'UNKNOWN',
      dataSource: 'none',
      reasons: ['正在申请设备权限并加载端侧推理模型。'],
      userSummary: '正在初始化权限与视觉模型',
      canTrustLivePerception: false,
      frameStale: false,
      inferenceStale: false,
      evaluatedAt: now,
    };
  }

  // 4. Hard unavailability checks (Fail-Safe)
  if (signals.pageHidden) {
    reasons.push('页面已切入后台，浏览器无法保证持续摄像头采集与后台助行。');
  }
  if (signals.cameraPermission === 'denied') {
    reasons.push('摄像头权限被拒绝，无法采集真实环境画面。');
  } else if (signals.cameraPermission === 'unavailable') {
    reasons.push('当前设备未检测到可用摄像头。');
  } else if (!signals.cameraTrackLive) {
    reasons.push('摄像头视频轨道已中断或停止。');
  }

  const frameStale =
    signals.lastFrameAt !== null && now - signals.lastFrameAt > thresholds.frameStaleMs;
  if (frameStale) {
    reasons.push('视频画面已停止更新（数据过期）。');
  }

  if (
    signals.pageHidden ||
    signals.cameraPermission === 'denied' ||
    signals.cameraPermission === 'unavailable' ||
    !signals.cameraTrackLive ||
    frameStale
  ) {
    return {
      state: 'UNAVAILABLE',
      perceptionStatus: 'UNAVAILABLE',
      dataSource: 'none',
      reasons,
      userSummary: reasons[0] || '实时感知不可用，请勿依赖当前输出。',
      canTrustLivePerception: false,
      frameStale,
      inferenceStale: true,
      evaluatedAt: now,
    };
  }

  // 5. Degradation checks (camera is live, but AI / audio / mic / latency degraded)
  const inferenceStale =
    signals.modelState === 'ready' &&
    signals.lastInferenceAt !== null &&
    now - signals.lastInferenceAt > thresholds.inferenceStaleMs;

  if (signals.modelState === 'failed') {
    reasons.push('端侧物体识别模型加载失败，当前仅剩基础边缘轮廓分析（无法识别具体物体）。');
  } else if (signals.modelState === 'loading') {
    reasons.push('端侧物体识别模型仍在加载中。');
  }

  if (signals.consecutiveInferenceFailures >= thresholds.maxConsecutiveFailures) {
    reasons.push('视觉推理连续失败，识别结果已降级。');
  }
  if (
    signals.lastInferenceDurationMs !== null &&
    signals.lastInferenceDurationMs > thresholds.inferenceTimeoutMs
  ) {
    reasons.push(`单次推理耗时过长（${Math.round(signals.lastInferenceDurationMs)}ms），实时性下降。`);
  }
  if (inferenceStale) {
    reasons.push('物体识别结果已过期，暂不输出确定性描述。');
  }
  if (signals.audioState !== 'running') {
    reasons.push('音频输出未激活或被浏览器暂停，可能听不到提示音。');
  }

  if (reasons.length > 0) {
    return {
      state: 'DEGRADED',
      perceptionStatus: 'DEGRADED',
      dataSource: 'live_camera',
      reasons,
      userSummary: `降级运行：${reasons[0]}`,
      canTrustLivePerception: false,
      frameStale: false,
      inferenceStale,
      evaluatedAt: now,
    };
  }

  // 6. Active & healthy
  return {
    state: 'ACTIVE',
    perceptionStatus: 'UNKNOWN', // Caller upgrades to 'DETECTED' only when valid items are present
    dataSource: 'live_camera',
    reasons: [],
    userSummary: '实时感知运行中（辅助参考，不可替代盲杖与现场判断）',
    canTrustLivePerception: true,
    frameStale: false,
    inferenceStale: false,
    evaluatedAt: now,
  };
}

/**
 * Single-flight async guard: prevents overlapping vision inferences from piling up
 * on slower mobile GPUs.
 */
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
  lastInferenceMs: number | null;
  modelLoadMs: number | null;
  mediaInterruptions: number;
  memoryUsedMb: number | null;
}

/**
 * Tracks real vision loop FPS separately from AI inference FPS & P95 latency.
 */
export class RuntimeMetricsTracker {
  private visionTimestamps: number[] = [];
  private aiTimestamps: number[] = [];
  private inferenceDurations: number[] = [];
  private modelLoadMs: number | null = null;
  private mediaInterruptions = 0;

  public recordVisionFrame(now: number) {
    this.visionTimestamps.push(now);
    const cutoff = now - 2000;
    while (this.visionTimestamps.length > 0 && this.visionTimestamps[0] < cutoff) {
      this.visionTimestamps.shift();
    }
  }

  public recordInference(now: number, durationMs: number) {
    this.aiTimestamps.push(now);
    const cutoff = now - 5000;
    while (this.aiTimestamps.length > 0 && this.aiTimestamps[0] < cutoff) {
      this.aiTimestamps.shift();
    }
    this.inferenceDurations.push(durationMs);
    if (this.inferenceDurations.length > 50) {
      this.inferenceDurations.shift();
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

    const visionFps = Math.round((this.visionTimestamps.length / 2) * 10) / 10;
    const aiFps = Math.round((this.aiTimestamps.length / 5) * 10) / 10;

    let avgInferenceMs = 0;
    let p95InferenceMs = 0;
    if (this.inferenceDurations.length > 0) {
      const sum = this.inferenceDurations.reduce((a, b) => a + b, 0);
      avgInferenceMs = Math.round(sum / this.inferenceDurations.length);
      const sorted = [...this.inferenceDurations].sort((a, b) => a - b);
      const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
      p95InferenceMs = Math.round(sorted[idx]);
    }

    const lastInferenceMs =
      this.inferenceDurations.length > 0
        ? Math.round(this.inferenceDurations[this.inferenceDurations.length - 1])
        : null;

    let memoryUsedMb: number | null = null;
    const perfWithMem = performance as unknown as {
      memory?: { usedJSHeapSize: number };
    };
    if (perfWithMem?.memory?.usedJSHeapSize) {
      memoryUsedMb = Math.round(perfWithMem.memory.usedJSHeapSize / (1024 * 1024));
    }

    return {
      visionFps,
      aiFps,
      avgInferenceMs,
      p95InferenceMs,
      lastInferenceMs,
      modelLoadMs: this.modelLoadMs,
      mediaInterruptions: this.mediaInterruptions,
      memoryUsedMb,
    };
  }
}
