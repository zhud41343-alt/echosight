// On-Device Multimodal AI Object Detector (TensorFlow.js COCO-SSD)
// Fail-Safe & Calibrated Monocular Estimation Architecture:
// - Strictly isolates live camera inference from demo presets (`allowDemoFallback` must be false in live mode)
// - Separates uncalibrated coarse range estimation from calibrated pinhole estimation
// - Rejects clipped / tiny / malformed bounding boxes as '距离未知'
// - Routes all speech through the unified AudioSpeechScheduler

import type { PresetSceneId } from './visionProcessor';
import { analyzeTrafficLightState, TrafficLightStabilizer } from './blindAssistTools';
import { audioScheduler } from './audioScheduler';

export type SpatialDirection = '正左侧' | '左前方' | '正前方' | '右前方' | '正右侧';

export interface CameraCalibration {
  calibrated: boolean;
  /** Focal length in pixels for the reference calibration height */
  focalLengthPx: number;
  /** Reference image height in pixels at which focalLengthPx was calibrated */
  referenceHeightPx: number;
  /** Reference image width in pixels at calibration time */
  referenceWidthPx?: number;
  /** Explicit physical distance (m) used during calibration */
  knownDistanceMeters?: number;
  /** Explicit user-measured physical height (m) of the reference object */
  knownHeightMeters?: number;
  heightSource?: 'user_measured' | 'class_default';
  /** MediaStreamTrack deviceId or camera label bound to this calibration */
  cameraDeviceId?: string;
  deviceLabel?: string;
  /** True ONLY after real tape-measure error evaluation validates an interval */
  measuredErrorValidated?: boolean;
}

const CALIBRATION_STORAGE_KEY = 'echosight:camera_calibration_v2';

export interface CalibrationComputeParams {
  labelEn: string;
  bbox: [number, number, number, number];
  knownDistanceMeters: number;
  knownHeightMeters: number;
  canvasW: number;
  canvasH: number;
  cameraDeviceId?: string;
  deviceLabel?: string;
}

/**
 * P0-3 (V6): Computes pinhole focal length from explicit user-measured physical height & distance.
 * Rejects boxes clipped on any edge, tiny boxes, extreme tilt/aspect distortion, or out-of-range inputs.
 */
export function computeCameraCalibrationFromDetection(
  params: CalibrationComputeParams
): { ok: true; calibration: CameraCalibration } | { ok: false; reason: string } {
  const {
    bbox,
    knownDistanceMeters,
    knownHeightMeters,
    canvasW,
    canvasH,
    cameraDeviceId,
    deviceLabel,
  } = params;
  const [x, y, w, h] = bbox;

  if (!Number.isFinite(knownDistanceMeters) || knownDistanceMeters < 0.5 || knownDistanceMeters > 10) {
    return { ok: false, reason: '已知距离必须在 0.5–10 米之间。' };
  }
  if (!Number.isFinite(knownHeightMeters) || knownHeightMeters < 0.15 || knownHeightMeters > 3.0) {
    return { ok: false, reason: '参考物真实高度必须在 0.15–3.0 米之间。' };
  }
  if (h < Math.max(24, canvasH * 0.1) || w < 14) {
    return { ok: false, reason: '目标框过小，请靠近或选择更清晰参考物。' };
  }
  // Reject clipping on ANY edge (top, bottom, left, right)
  if (x <= 2 || x + w >= canvasW - 2 || y <= 2 || y + h >= canvasH - 2) {
    return { ok: false, reason: '参考物检测框贴边裁切，无法用于焦距标定。' };
  }
  // Reject extreme aspect ratios indicating severe perspective distortion or occlusion
  const aspect = w / Math.max(1, h);
  if (aspect > 4.5 || aspect < 0.12) {
    return { ok: false, reason: '检测框纵横比异常（可能存在严重遮挡或倾斜），已拒判。' };
  }

  const focalLengthPx = Math.round((h * knownDistanceMeters) / knownHeightMeters);
  if (!Number.isFinite(focalLengthPx) || focalLengthPx < canvasH * 0.35 || focalLengthPx > canvasH * 3.5) {
    return { ok: false, reason: '推算焦距超出合理物理范围，请核对距离与高度测量值。' };
  }

  return {
    ok: true,
    calibration: {
      calibrated: true,
      focalLengthPx,
      referenceHeightPx: canvasH,
      referenceWidthPx: canvasW,
      knownDistanceMeters,
      knownHeightMeters,
      heightSource: 'user_measured',
      cameraDeviceId,
      deviceLabel,
      measuredErrorValidated: false,
    },
  };
}

/**
 * Verifies whether a saved calibration still applies to the active camera device & resolution.
 * Switching cameras (front/back) or changing aspect ratio automatically invalidates reuse.
 */
export function isCalibrationApplicable(
  cal: CameraCalibration | null | undefined,
  currentCanvasW: number,
  currentCanvasH: number,
  currentCameraDeviceId?: string
): boolean {
  if (!cal || !cal.calibrated || cal.focalLengthPx <= 0 || cal.referenceHeightPx <= 0) {
    return false;
  }
  if (cal.cameraDeviceId && currentCameraDeviceId && cal.cameraDeviceId !== currentCameraDeviceId) {
    return false;
  }
  if (cal.referenceWidthPx && cal.referenceWidthPx > 0 && currentCanvasW > 0 && currentCanvasH > 0) {
    const savedAspect = cal.referenceWidthPx / cal.referenceHeightPx;
    const currentAspect = currentCanvasW / currentCanvasH;
    if (Math.abs(savedAspect - currentAspect) / Math.max(savedAspect, currentAspect) > 0.12) {
      return false;
    }
  }
  return true;
}

export function loadSavedCameraCalibration(): CameraCalibration | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(CALIBRATION_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<CameraCalibration>;
    if (
      parsed.calibrated === true &&
      typeof parsed.focalLengthPx === 'number' &&
      parsed.focalLengthPx > 0 &&
      typeof parsed.referenceHeightPx === 'number' &&
      parsed.referenceHeightPx > 0
    ) {
      return {
        calibrated: true,
        focalLengthPx: parsed.focalLengthPx,
        referenceHeightPx: parsed.referenceHeightPx,
        referenceWidthPx: parsed.referenceWidthPx,
        knownDistanceMeters: parsed.knownDistanceMeters,
        knownHeightMeters: parsed.knownHeightMeters,
        heightSource: parsed.heightSource ?? 'user_measured',
        cameraDeviceId: parsed.cameraDeviceId,
        deviceLabel: parsed.deviceLabel,
        measuredErrorValidated: Boolean(parsed.measuredErrorValidated),
      };
    }
    return null;
  } catch {
    return null;
  }
}

export function saveCameraCalibration(cal: CameraCalibration): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(cal));
  } catch {
    // Ignore storage quota errors
  }
}

export function clearSavedCameraCalibration(): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(CALIBRATION_STORAGE_KEY);
  } catch {
    // Ignore
  }
}

export interface DistanceEstimate {
  valid: boolean;
  calibrated: boolean;
  estimatedMeters: number | null;
  band: '近距离' | '中距离' | '较远处' | '未知';
  displayText: string;
  invalidReason?: 'clipped_vertical' | 'bbox_too_small' | 'unknown_class_height' | 'abnormal_aspect';
}

export interface SpatialDetection {
  labelEn: string;
  labelZh: string;
  score: number; // 0..1
  bbox: [number, number, number, number]; // [x, y, w, h]
  direction: SpatialDirection;
  clockFace: string;
  distance: '近距离' | '中距离' | '较远处' | '未知';
  distanceMeters: number | null;
  distanceText: string;
  distanceCalibrated: boolean;
  priority: 'danger' | 'obstacle' | 'normal';
  source: 'tfjs' | 'demo_preset';
  detectedAt: number;
}

export const COCO_ZH_MAP: Record<string, string> = {
  person: '行人',
  bicycle: '自行车',
  car: '汽车',
  motorcycle: '摩托车',
  airplane: '飞机',
  bus: '公交车',
  train: '列车',
  truck: '卡车',
  boat: '船只',
  'traffic light': '交通信号灯',
  'fire hydrant': '消防栓',
  'stop sign': '停车标志',
  'parking meter': '咪表立柱',
  bench: '长椅',
  bird: '小鸟',
  cat: '猫',
  dog: '狗',
  horse: '马',
  sheep: '羊',
  cow: '牛',
  elephant: '大象',
  bear: '熊',
  zebra: '斑马',
  giraffe: '长颈鹿',
  backpack: '背包',
  umbrella: '雨伞',
  handbag: '手提包',
  tie: '领带',
  suitcase: '行李箱',
  frisbee: '飞盘',
  skis: '滑雪板',
  snowboard: '单板',
  'sports ball': '球类',
  kite: '风筝',
  'baseball bat': '球棒',
  'baseball glove': '手套',
  skateboard: '滑板',
  surfboard: '冲浪板',
  'tennis racket': '球拍',
  bottle: '水瓶',
  'wine glass': '玻璃杯',
  cup: '水杯',
  fork: '叉子',
  knife: '刀具',
  spoon: '勺子',
  bowl: '碗',
  banana: '香蕉',
  apple: '苹果',
  sandwich: '三明治',
  orange: '橙子',
  broccoli: '西兰花',
  carrot: '胡萝卜',
  'hot dog': '热狗',
  pizza: '披萨',
  donut: '甜甜圈',
  cake: '蛋糕',
  chair: '椅子',
  couch: '沙发',
  'potted plant': '盆栽绿植',
  bed: '床',
  'dining table': '桌子',
  toilet: '马桶',
  tv: '电视屏幕',
  laptop: '笔记本电脑',
  mouse: '鼠标',
  remote: '遥控器',
  keyboard: '键盘',
  'cell phone': '手机',
  microwave: '微波炉',
  oven: '烤箱',
  toaster: '烤面包机',
  sink: '洗手池',
  refrigerator: '冰箱',
  book: '书本',
  clock: '时钟',
  vase: '花瓶',
  scissors: '剪刀',
  'teddy bear': '毛绒玩具',
  'hair drier': '吹风机',
  toothbrush: '牙刷',
};

export const REAL_HEIGHT_M: Record<string, number> = {
  person: 1.68,
  bicycle: 1.05,
  car: 1.5,
  motorcycle: 1.15,
  bus: 3.0,
  train: 3.4,
  truck: 2.8,
  'traffic light': 0.9,
  'fire hydrant': 0.85,
  'stop sign': 0.75,
  'parking meter': 1.2,
  bench: 0.8,
  dog: 0.55,
  cat: 0.3,
  backpack: 0.45,
  suitcase: 0.65,
  chair: 0.85,
  couch: 0.85,
  'potted plant': 0.65,
  bed: 0.65,
  'dining table': 0.75,
  toilet: 0.75,
  tv: 0.55,
  laptop: 0.28,
  'cell phone': 0.15,
  refrigerator: 1.75,
  cup: 0.13,
  bottle: 0.22,
  'wine glass': 0.18,
  bowl: 0.1,
  book: 0.22,
  vase: 0.28,
  scissors: 0.16,
  knife: 0.22,
  doorway: 2.0,
};

const DANGER_CLASSES = new Set([
  'car',
  'bus',
  'truck',
  'motorcycle',
  'bicycle',
  'train',
  'fire hydrant',
  'knife',
  'scissors',
]);

const OBSTACLE_CLASSES = new Set([
  'person',
  'traffic light',
  'stop sign',
  'parking meter',
  'bench',
  'chair',
  'couch',
  'dining table',
  'bed',
  'potted plant',
  'suitcase',
  'refrigerator',
  'dog',
  'doorway',
]);

/**
 * Monocular distance estimator with explicit validity checks and calibration separation.
 * - Rejects vertically clipped boxes (partial occlusion at top/bottom edge), tiny boxes, or extreme aspect ratios.
 * - When uncalibrated, returns conservative range labels instead of false decimal precision.
 */
export function estimateMonocularDistance(
  labelEn: string,
  bbox: [number, number, number, number],
  canvasW: number,
  canvasH: number,
  calibration?: CameraCalibration | null
): DistanceEstimate {
  const [, y, w, h] = bbox;
  const realH = REAL_HEIGHT_M[labelEn];

  if (!realH) {
    return {
      valid: false,
      calibrated: Boolean(calibration?.calibrated),
      estimatedMeters: null,
      band: '未知',
      displayText: '距离未知',
      invalidReason: 'unknown_class_height',
    };
  }

  if (h < Math.max(14, canvasH * 0.04) || w < 8) {
    return {
      valid: false,
      calibrated: Boolean(calibration?.calibrated),
      estimatedMeters: null,
      band: '未知',
      displayText: '距离未知',
      invalidReason: 'bbox_too_small',
    };
  }

  // If object touches both or either vertical edge heavily, bbox height is truncated
  const clippedTop = y <= 2;
  const clippedBottom = y + h >= canvasH - 2;
  if ((clippedTop && clippedBottom) || (clippedBottom && h / canvasH > 0.78)) {
    return {
      valid: false,
      calibrated: Boolean(calibration?.calibrated),
      estimatedMeters: null,
      band: '未知',
      displayText: '距离未知（部分裁切）',
      invalidReason: 'clipped_vertical',
    };
  }

  const aspect = w / Math.max(1, h);
  if (aspect > 6.5 || aspect < 0.08) {
    return {
      valid: false,
      calibrated: Boolean(calibration?.calibrated),
      estimatedMeters: null,
      band: '未知',
      displayText: '距离未知',
      invalidReason: 'abnormal_aspect',
    };
  }

  const isCalibrated = Boolean(calibration?.calibrated && calibration.focalLengthPx > 0);
  const focalPx = isCalibrated
    ? calibration!.focalLengthPx * (canvasH / Math.max(1, calibration!.referenceHeightPx))
    : canvasH * 0.98; // Uncalibrated generic smartphone approximation

  const rawMeters = (realH * focalPx) / h;
  const clamped = Math.round(Math.max(0.3, Math.min(18, rawMeters)) * 100) / 100;

  const band: '近距离' | '中距离' | '较远处' =
    clamped <= 2.0 ? '近距离' : clamped <= 5.0 ? '中距离' : '较远处';

  if (!isCalibrated) {
    // Conservative range only when camera is uncalibrated
    const rangeText =
      band === '近距离'
        ? '较近（未标定粗估）'
        : band === '中距离'
          ? '中距（未标定粗估）'
          : '较远（未标定粗估）';
    return {
      valid: true,
      calibrated: false,
      estimatedMeters: clamped,
      band,
      displayText: rangeText,
    };
  }

  if (!calibration?.measuredErrorValidated) {
    // P0-3 (V6): Never claim a fixed ±15% verified error interval before real tape-measure evaluation
    return {
      valid: true,
      calibrated: true,
      estimatedMeters: clamped,
      band,
      displayText: `${band}（实验估算约${clamped.toFixed(1)}米·未实测误差）`,
    };
  }

  // Only when backed by real measured error validation
  const low = Math.max(0.3, Math.round(clamped * 0.85 * 10) / 10);
  const high = Math.round(clamped * 1.15 * 10) / 10;
  return {
    valid: true,
    calibrated: true,
    estimatedMeters: clamped,
    band,
    displayText: `约${low}–${high}米`,
  };
}

export interface DistanceEvalSample {
  referenceMeters: number;
  predictedMeters: number | null;
}

export interface DistanceBandStat {
  band: string;
  count: number;
  validCount: number;
  maeMeters: number | null;
  p90ErrorMeters: number | null;
}

export interface DistanceErrorReport {
  total: number;
  validCount: number;
  unknownCount: number;
  maeMeters: number | null;
  p90ErrorMeters: number | null;
  byBand: DistanceBandStat[];
}

function calcErrors(pairs: Array<{ ref: number; pred: number }>): {
  mae: number | null;
  p90: number | null;
} {
  if (pairs.length === 0) return { mae: null, p90: null };
  const absErrors = pairs.map((p) => Math.abs(p.pred - p.ref)).sort((a, b) => a - b);
  const mae = absErrors.reduce((a, b) => a + b, 0) / absErrors.length;
  const p90Idx = Math.min(absErrors.length - 1, Math.floor(absErrors.length * 0.9));
  return {
    mae: Math.round(mae * 1000) / 1000,
    p90: Math.round(absErrors[p90Idx] * 1000) / 1000,
  };
}

/**
 * Computes MAE, P90 absolute error, and per-distance-band statistics.
 */
export function computeDistanceErrorStats(samples: DistanceEvalSample[]): DistanceErrorReport {
  const validPairs: Array<{ ref: number; pred: number }> = [];
  for (const s of samples) {
    if (s.predictedMeters !== null && Number.isFinite(s.predictedMeters)) {
      validPairs.push({ ref: s.referenceMeters, pred: s.predictedMeters });
    }
  }
  const overall = calcErrors(validPairs);

  const bands: Array<{ label: string; min: number; max: number }> = [
    { label: '0.5–2m', min: 0, max: 2.01 },
    { label: '2–5m', min: 2.01, max: 5.01 },
    { label: '5–10m', min: 5.01, max: 12.0 },
  ];

  const byBand: DistanceBandStat[] = bands.map((b) => {
    const inBand = samples.filter((s) => s.referenceMeters >= b.min && s.referenceMeters < b.max);
    const validInBand = inBand
      .filter((s) => s.predictedMeters !== null)
      .map((s) => ({ ref: s.referenceMeters, pred: s.predictedMeters! }));
    const st = calcErrors(validInBand);
    return {
      band: b.label,
      count: inBand.length,
      validCount: validInBand.length,
      maeMeters: st.mae,
      p90ErrorMeters: st.p90,
    };
  });

  return {
    total: samples.length,
    validCount: validPairs.length,
    unknownCount: samples.length - validPairs.length,
    maeMeters: overall.mae,
    p90ErrorMeters: overall.p90,
    byBand,
  };
}

function computeDirectionMeta(
  bbox: [number, number, number, number],
  canvasW: number
): { direction: SpatialDirection; clockFace: string } {
  const [x, , w] = bbox;
  const centerX = (x + w / 2) / Math.max(1, canvasW);
  if (centerX < 0.2) return { direction: '正左侧', clockFace: '9点方向' };
  if (centerX < 0.39) return { direction: '左前方', clockFace: '10点方向' };
  if (centerX <= 0.61) return { direction: '正前方', clockFace: '12点方向' };
  if (centerX <= 0.8) return { direction: '右前方', clockFace: '2点方向' };
  return { direction: '正右侧', clockFace: '3点方向' };
}

type CocoModel = {
  detect: (
    img: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement,
    maxNumBoxes?: number,
    minScore?: number
  ) => Promise<Array<{ class: string; score: number; bbox: [number, number, number, number] }>>;
};

let loadedModel: CocoModel | null = null;
let loadedModelVariant: 'mobilenet_v2' | 'lite_mobilenet_v2' | null = null;
let lastModelLoadMs: number | null = null;
let loadFailed = false;
let inFlightLoadPromise: Promise<ModelLoadResult> | null = null;

const trafficLightStabilizer = new TrafficLightStabilizer(3);

export function computeBboxIoU(
  a: [number, number, number, number],
  b: [number, number, number, number]
): number {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  const union = aw * ah + bw * bh - inter;
  return union > 0 ? inter / union : 0;
}

interface TrackedTarget {
  trackId: number;
  labelEn: string;
  bbox: [number, number, number, number];
  hits: number;
  missedFrames: number;
  lastDetection: SpatialDetection;
}

/**
 * P0-1 Spatial Multi-Object Tracker (IoU + spatial proximity matching).
 * Fixes class-only matching flaws:
 * - Two same-class objects at different positions (e.g. left chair vs right chair) maintain
 *   independent tracks and NEVER falsely confirm each other.
 * - High-confidence fast movers (>= 0.68) pass immediately without over-smoothing lag.
 * - Medium-confidence targets (0.45..0.68) require 2 consecutive spatially-matched frames.
 * - Automatically expires disappeared targets and resets cleanly on session/mode changes.
 */
export class SpatialObjectTracker {
  private tracks: TrackedTarget[] = [];
  private nextTrackId = 1;
  private lastProcessedFrameId: number | null = null;

  public update(
    detections: SpatialDetection[],
    frameW = 640,
    frameH = 480,
    frameId?: number
  ): SpatialDetection[] {
    // P0-1 (V5): Duplicate or out-of-order frameId MUST NEVER increment hits or confirm tracks
    if (
      frameId !== undefined &&
      this.lastProcessedFrameId !== null &&
      frameId <= this.lastProcessedFrameId
    ) {
      return this.tracks
        .filter((tr) => tr.missedFrames === 0 && (tr.hits >= 2 || tr.lastDetection.score >= 0.68))
        .map((tr) => tr.lastDetection);
    }
    if (frameId !== undefined) {
      this.lastProcessedFrameId = frameId;
    }

    const diag = Math.hypot(Math.max(1, frameW), Math.max(1, frameH));
    const matchedTrackIds = new Set<number>();
    const confirmed: SpatialDetection[] = [];

    for (const det of detections) {
      let bestTrack: TrackedTarget | null = null;
      let bestScore = -1;

      const detCx = det.bbox[0] + det.bbox[2] / 2;
      const detCy = det.bbox[1] + det.bbox[3] / 2;

      for (const tr of this.tracks) {
        if (tr.labelEn !== det.labelEn || matchedTrackIds.has(tr.trackId)) continue;
        const iou = computeBboxIoU(tr.bbox, det.bbox);
        const trCx = tr.bbox[0] + tr.bbox[2] / 2;
        const trCy = tr.bbox[1] + tr.bbox[3] / 2;
        const normDist = Math.hypot(detCx - trCx, detCy - trCy) / diag;

        // Require meaningful spatial overlap (IoU >= 0.2) or tight center proximity (< 14% diag)
        if (iou >= 0.2 || normDist <= 0.14) {
          const matchScore = iou * 2 + (1 - normDist);
          if (matchScore > bestScore) {
            bestScore = matchScore;
            bestTrack = tr;
          }
        }
      }

      if (bestTrack) {
        matchedTrackIds.add(bestTrack.trackId);
        // Smooth bbox with adaptive EMA so overlay boxes glide cleanly without lagging fast movers
        const alpha = det.score >= 0.75 ? 0.82 : 0.62;
        const smoothedBbox: [number, number, number, number] = [
          bestTrack.bbox[0] * (1 - alpha) + det.bbox[0] * alpha,
          bestTrack.bbox[1] * (1 - alpha) + det.bbox[1] * alpha,
          bestTrack.bbox[2] * (1 - alpha) + det.bbox[2] * alpha,
          bestTrack.bbox[3] * (1 - alpha) + det.bbox[3] * alpha,
        ];
        const smoothedDet: SpatialDetection = { ...det, bbox: smoothedBbox };
        bestTrack.bbox = smoothedBbox;
        bestTrack.hits += 1;
        bestTrack.missedFrames = 0;
        bestTrack.lastDetection = smoothedDet;
        if (bestTrack.hits >= 2 || det.score >= 0.68) {
          confirmed.push(smoothedDet);
        }
      } else {
        const newTrack: TrackedTarget = {
          trackId: this.nextTrackId++,
          labelEn: det.labelEn,
          bbox: det.bbox,
          hits: 1,
          missedFrames: 0,
          lastDetection: det,
        };
        this.tracks.push(newTrack);
        matchedTrackIds.add(newTrack.trackId);
        // Fast-moving / very clear target (>= 0.68) confirms immediately without lag
        if (det.score >= 0.68) {
          confirmed.push(det);
        }
      }
    }

    // Age unmatched tracks and prune disappeared objects (max 1 missed frame tolerance)
    this.tracks = this.tracks.filter((tr) => {
      if (matchedTrackIds.has(tr.trackId)) return true;
      tr.missedFrames += 1;
      return tr.missedFrames <= 1;
    });

    return confirmed;
  }

  public getActiveTrackCount(): number {
    return this.tracks.length;
  }

  public reset() {
    this.tracks = [];
    this.nextTrackId = 1;
    this.lastProcessedFrameId = null;
  }
}

export interface ModelLoadResult {
  /** True ONLY when the neural network actually loaded and can infer */
  modelReady: boolean;
  /** Backwards-compatible alias for demo/studio callers */
  ready: boolean;
  fallback: boolean;
  variant: 'mobilenet_v2' | 'lite_mobilenet_v2' | 'none';
  loadMs: number;
}

export async function loadOnDeviceAIModel(
  onStatus?: (msg: string) => void,
  forceRetry = false
): Promise<ModelLoadResult> {
  if (loadedModel && loadedModelVariant) {
    onStatus?.(
      loadedModelVariant === 'mobilenet_v2'
        ? '端侧 MobileNet v2 就绪（本地 WebGL 推理）'
        : '端侧轻量 COCO-SSD 就绪（本地 WebGL 推理）'
    );
    return {
      modelReady: true,
      ready: true,
      fallback: false,
      variant: loadedModelVariant,
      loadMs: lastModelLoadMs ?? 0,
    };
  }
  if (loadFailed && !forceRetry) {
    onStatus?.('端侧神经网络加载失败（实时识别已降级为不可用）');
    return {
      modelReady: false,
      ready: false,
      fallback: true,
      variant: 'none',
      loadMs: lastModelLoadMs ?? 0,
    };
  }
  if (inFlightLoadPromise) {
    return inFlightLoadPromise;
  }
  if (forceRetry) {
    loadFailed = false;
  }

  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  inFlightLoadPromise = (async (): Promise<ModelLoadResult> => {
    try {
      onStatus?.('正在启动本地 TensorFlow.js 推理引擎...');
      const tf = await import('@tensorflow/tfjs');
      await tf.ready();
      onStatus?.('正在加载端侧 COCO-SSD（80 类物体）...');
      const cocoSsd = await import('@tensorflow-models/coco-ssd');
      try {
        loadedModel = (await cocoSsd.load({ base: 'mobilenet_v2' })) as unknown as CocoModel;
        loadedModelVariant = 'mobilenet_v2';
      } catch {
        loadedModel = (await cocoSsd.load({ base: 'lite_mobilenet_v2' })) as unknown as CocoModel;
        loadedModelVariant = 'lite_mobilenet_v2';
      }
      lastModelLoadMs = Math.round(
        (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0
      );
      onStatus?.(
        loadedModelVariant === 'mobilenet_v2'
          ? `端侧 MobileNet v2 就绪（耗时 ${lastModelLoadMs}ms，本地 WebGL 推理）`
          : `端侧轻量 COCO-SSD 就绪（耗时 ${lastModelLoadMs}ms）`
      );
      return {
        modelReady: true,
        ready: true,
        fallback: false,
        variant: loadedModelVariant,
        loadMs: lastModelLoadMs,
      };
    } catch {
      loadFailed = true;
      lastModelLoadMs = Math.round(
        (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0
      );
      onStatus?.('端侧神经网络加载失败：实时识别不可用，请勿依赖物体识别结果');
      return {
        modelReady: false,
        ready: false,
        fallback: true,
        variant: 'none',
        loadMs: lastModelLoadMs,
      };
    } finally {
      inFlightLoadPromise = null;
    }
  })();

  return inFlightLoadPromise;
}

export function getLoadedModelState(): 'idle' | 'ready' | 'failed' {
  if (loadedModel) return 'ready';
  if (loadFailed) return 'failed';
  return 'idle';
}

function getDemoPresetDetections(
  preset: PresetSceneId,
  w: number,
  h: number,
  now: number
): SpatialDetection[] {
  if (preset === 'crosswalk') {
    return [
      {
        labelEn: 'car',
        labelZh: '模拟车辆（演示数据）',
        score: 0.91,
        bbox: [w * 0.67, h * 0.43, w * 0.26, h * 0.28],
        direction: '右前方',
        clockFace: '2点方向',
        distance: '近距离',
        distanceMeters: null,
        distanceText: '演示场景',
        distanceCalibrated: false,
        priority: 'danger',
        source: 'demo_preset',
        detectedAt: now,
      },
      {
        labelEn: 'traffic light',
        labelZh: '模拟信号灯（绿色像素·仅演示）',
        score: 0.94,
        bbox: [w * 0.11, h * 0.11, w * 0.1, h * 0.25],
        direction: '左前方',
        clockFace: '10点方向',
        distance: '中距离',
        distanceMeters: null,
        distanceText: '演示场景',
        distanceCalibrated: false,
        priority: 'obstacle',
        source: 'demo_preset',
        detectedAt: now,
      },
    ];
  }
  if (preset === 'room') {
    return [
      {
        labelEn: 'dining table',
        labelZh: '模拟餐桌（演示数据）',
        score: 0.92,
        bbox: [w * 0.07, h * 0.3, w * 0.36, h * 0.58],
        direction: '左前方',
        clockFace: '10点方向',
        distance: '近距离',
        distanceMeters: null,
        distanceText: '演示场景',
        distanceCalibrated: false,
        priority: 'obstacle',
        source: 'demo_preset',
        detectedAt: now,
      },
      {
        labelEn: 'doorway',
        labelZh: '模拟门框（演示数据）',
        score: 0.89,
        bbox: [w * 0.55, h * 0.13, w * 0.24, h * 0.68],
        direction: '右前方',
        clockFace: '2点方向',
        distance: '中距离',
        distanceMeters: null,
        distanceText: '演示场景',
        distanceCalibrated: false,
        priority: 'obstacle',
        source: 'demo_preset',
        detectedAt: now,
      },
    ];
  }
  if (preset === 'mountain') {
    return [
      {
        labelEn: 'moon',
        labelZh: '模拟光源（演示数据）',
        score: 0.95,
        bbox: [w * 0.72, h * 0.12, w * 0.12, h * 0.2],
        direction: '右前方',
        clockFace: '2点方向',
        distance: '较远处',
        distanceMeters: null,
        distanceText: '演示场景',
        distanceCalibrated: false,
        priority: 'normal',
        source: 'demo_preset',
        detectedAt: now,
      },
    ];
  }
  return [
    {
      labelEn: 'target',
      labelZh: '模拟校准靶标（演示数据）',
      score: 0.96,
      bbox: [w * 0.34, h * 0.25, w * 0.32, h * 0.5],
      direction: '正前方',
      clockFace: '12点方向',
      distance: '中距离',
      distanceMeters: null,
      distanceText: '演示场景',
      distanceCalibrated: false,
      priority: 'normal',
      source: 'demo_preset',
      detectedAt: now,
    },
  ];
}

function rankDetections(items: SpatialDetection[]): SpatialDetection[] {
  const prioWeight = { danger: 3, obstacle: 2, normal: 1 };
  const bandWeight = { 近距离: 3, 中距离: 2, 较远处: 1, 未知: 0 };
  return [...items].sort((a, b) => {
    const scoreA = prioWeight[a.priority] * 10 + bandWeight[a.distance] * 3 + a.score;
    const scoreB = prioWeight[b.priority] * 10 + bandWeight[b.distance] * 3 + b.score;
    return scoreB - scoreA;
  });
}

const NOISY_SMALL_CLASSES = new Set([
  'tie',
  'toothbrush',
  'remote',
  'fork',
  'spoon',
  'knife',
  'scissors',
  'bird',
  'kite',
  'hair drier',
  'mouse',
]);

/**
 * Runs object detection.
 * CRITICAL DATA ISOLATION RULE:
 * When `allowDemoFallback` is false (all live camera / blind-assist modes),
 * demo preset boxes are NEVER returned.
 */
export async function detectSpatialObjects(
  canvas: HTMLCanvasElement,
  allowDemoFallback: boolean,
  presetId: PresetSceneId,
  calibration?: CameraCalibration | null
): Promise<SpatialDetection[]> {
  const now = typeof performance !== 'undefined' ? performance.now() : Date.now();

  if (loadedModel && !loadFailed) {
    const rawPreds = await loadedModel.detect(canvas, 6, 0.5);
    // Suppress frequent indoor false positives on tiny/ambiguous classes unless high confidence
    const preds = rawPreds.filter((p) =>
      NOISY_SMALL_CLASSES.has(p.class) ? p.score >= 0.65 : p.score >= 0.5
    );
    if (preds.length > 0) {
      const mapped: SpatialDetection[] = preds.map((p) => {
        const dir = computeDirectionMeta(p.bbox, canvas.width);
        const dist = estimateMonocularDistance(
          p.class,
          p.bbox,
          canvas.width,
          canvas.height,
          calibration
        );
        const priority: 'danger' | 'obstacle' | 'normal' = DANGER_CLASSES.has(p.class)
          ? 'danger'
          : OBSTACLE_CLASSES.has(p.class)
            ? 'obstacle'
            : 'normal';

        let labelZh = COCO_ZH_MAP[p.class] || p.class;
        // Experimental color check ONLY — explicitly labeled as pixel color, not crossing clearance
        if (p.class === 'traffic light') {
          const rawColor = analyzeTrafficLightState(canvas, p.bbox);
          const stableColor = trafficLightStabilizer.push(rawColor);
          if (stableColor !== '未知') {
            labelZh = `交通信号灯（${stableColor}·仅供参考）`;
          }
        }

        return {
          labelEn: p.class,
          labelZh,
          score: p.score,
          bbox: p.bbox,
          direction: dir.direction,
          clockFace: dir.clockFace,
          distance: dist.band,
          distanceMeters: dist.estimatedMeters,
          distanceText: dist.displayText,
          distanceCalibrated: dist.calibrated,
          priority,
          source: 'tfjs',
          detectedAt: now,
        };
      });
      return rankDetections(mapped);
    }
    return [];
  }

  if (allowDemoFallback) {
    return rankDetections(getDemoPresetDetections(presetId, canvas.width, canvas.height, now));
  }

  return [];
}

export function speakCue(
  text: string,
  rate = 1.04,
  priority: 'EMERGENCY' | 'USER_QUERY' | 'AMBIENT' = 'USER_QUERY'
) {
  audioScheduler.schedule({
    text,
    priority,
    rate,
  });
}

export interface SpatialNarrationPayload {
  text: string;
  priority: 'EMERGENCY' | 'USER_QUERY' | 'AMBIENT';
  dedupeKey: string;
  cooldownMs: number;
}

/**
 * Pure narration builder decoupled from browser TTS acceptance (P0-1 V6).
 * NEVER infers "safe / clear / walkable" from an empty detection array.
 */
export function buildSpatialDetectionNotification(
  detections: SpatialDetection[],
  force = false,
  fallbackStatusCue?: string
): SpatialNarrationPayload | null {
  if (detections.length === 0) {
    if (!force && !fallbackStatusCue) return null;
    const text =
      fallbackStatusCue ||
      '未检测到已支持类别的物体。这不代表前方无障碍，请结合盲杖与现场确认。';
    return {
      text,
      priority: force ? 'USER_QUERY' : 'AMBIENT',
      dedupeKey: `empty:${text}`,
      cooldownMs: force ? 0 : 12000,
    };
  }

  const top = detections.slice(0, force ? 3 : 2);
  const signature = top.map((d) => `${d.direction}-${d.distance}-${d.labelZh}`).join('|');
  const hasNearHazard = top.some((d) => d.priority === 'danger' && d.distance === '近距离');

  const summary = top
    .map((d) => {
      // Keep speech crisp & natural without repeating "（未标定粗估）" on every item
      const spokenDist =
        d.distance === '未知'
          ? '距离未知'
          : d.distanceCalibrated
            ? d.distanceText
            : d.distance === '近距离'
              ? '较近'
              : d.distance === '中距离'
                ? '中距'
                : '较远';
      const prefix = d.priority === 'danger' && d.distance === '近距离' ? '注意，' : '';
      return `${prefix}${d.direction}${spokenDist}有${d.labelZh}`;
    })
    .join('；');

  return {
    text: summary,
    priority: force ? 'USER_QUERY' : hasNearHazard ? 'EMERGENCY' : 'AMBIENT',
    dedupeKey: signature,
    cooldownMs: force ? 0 : hasNearHazard ? 6000 : 22000,
  };
}

/**
 * Fail-Safe spatial narration routed through AudioSpeechScheduler.
 * Returns text whenever accepted OR delegated to screen reader ARIA Live (never drops SR updates!).
 */
export function speakSpatialDetections(
  detections: SpatialDetection[],
  force = false,
  fallbackStatusCue?: string
): string | null {
  const payload = buildSpatialDetectionNotification(detections, force, fallbackStatusCue);
  if (!payload) return null;

  const res = audioScheduler.schedule({
    text: payload.text,
    priority: payload.priority,
    dedupeKey: payload.dedupeKey,
    cooldownMs: payload.cooldownMs,
  });

  return res.accepted || res.reason === 'screen_reader_delegated' ? payload.text : null;
}
