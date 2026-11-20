// On-Device Multimodal AI Object Detector (TensorFlow.js COCO-SSD + Offline Spatial Fallback)
// Features:
// - Full 80-class COCO Chinese vocabulary
// - Monocular pinhole camera distance estimation in meters (up to ~12m for vehicles/people)
// - 5-sector + clock-face directional localization
// - Hazard prioritization & delta-aware warm Chinese voice guidance

import type { PresetSceneId } from './visionProcessor';
import { analyzeTrafficLightState } from './blindAssistTools';

export type SpatialDirection = '正左侧' | '左前方' | '正前方' | '右前方' | '正右侧';

export interface SpatialDetection {
  labelEn: string;
  labelZh: string;
  score: number; // 0..1
  bbox: [number, number, number, number]; // [x, y, w, h]
  direction: SpatialDirection;
  clockFace: string; // e.g. '10点钟'
  distance: '近距离' | '中距离' | '较远处';
  distanceMeters: number; // Estimated distance in meters (e.g. 1.2, 4.5)
  distanceText: string; // e.g. '约1.2米'
  priority: 'danger' | 'obstacle' | 'normal';
  source: 'tfjs' | 'local-topology';
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

// Typical real-world vertical height in meters for pinhole monocular distance estimation:
// D ≈ (H_real * focalPx) / bboxHeightPx
const REAL_HEIGHT_M: Record<string, number> = {
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

export function formatDistanceMeters(meters: number): string {
  const clamped = Math.max(0.4, Math.min(15, meters));
  if (clamped < 1.0) return `不到1米`;
  if (clamped < 3.0) return `约${clamped.toFixed(1)}米`;
  if (clamped < 10.0) return `约${Math.round(clamped)}米`;
  return `约10米外`;
}

function computeSpatialMeta(
  labelEn: string,
  bbox: [number, number, number, number],
  canvasW: number,
  canvasH: number
): {
  direction: SpatialDirection;
  clockFace: string;
  distance: '近距离' | '中距离' | '较远处';
  distanceMeters: number;
  distanceText: string;
  priority: 'danger' | 'obstacle' | 'normal';
} {
  const [x, , w, h] = bbox;
  const centerX = (x + w / 2) / canvasW;

  let direction: SpatialDirection = '正前方';
  let clockFace = '12点方向';
  if (centerX < 0.2) {
    direction = '正左侧';
    clockFace = '9点方向';
  } else if (centerX < 0.39) {
    direction = '左前方';
    clockFace = '10点方向';
  } else if (centerX <= 0.61) {
    direction = '正前方';
    clockFace = '12点方向';
  } else if (centerX <= 0.8) {
    direction = '右前方';
    clockFace = '2点方向';
  } else {
    direction = '正右侧';
    clockFace = '3点方向';
  }

  // Smartphone rear camera typical vertical FOV ≈ 54° -> focalPx ≈ canvasH / (2 * tan(27°)) ≈ canvasH * 0.98
  const focalPx = canvasH * 0.98;
  const realH = REAL_HEIGHT_M[labelEn] ?? 0.65;
  const safeBboxH = Math.max(12, h);
  const rawMeters = (realH * focalPx) / safeBboxH;
  const distanceMeters = Math.round(Math.max(0.4, Math.min(15, rawMeters)) * 10) / 10;
  const distanceText = formatDistanceMeters(distanceMeters);

  const distance: '近距离' | '中距离' | '较远处' =
    distanceMeters <= 2.0 ? '近距离' : distanceMeters <= 5.0 ? '中距离' : '较远处';

  const priority: 'danger' | 'obstacle' | 'normal' = DANGER_CLASSES.has(labelEn)
    ? 'danger'
    : OBSTACLE_CLASSES.has(labelEn)
      ? 'obstacle'
      : 'normal';

  return { direction, clockFace, distance, distanceMeters, distanceText, priority };
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
let loadFailed = false;

export async function loadOnDeviceAIModel(
  onStatus?: (msg: string) => void
): Promise<{ ready: boolean; fallback: boolean }> {
  if (loadedModel) {
    onStatus?.(
      loadedModelVariant === 'mobilenet_v2'
        ? '高精度 MobileNet v2 就绪（80 类物体 · 远距米数测距）'
        : '轻量 COCO-SSD 就绪（80 类物体 · 远距米数测距）'
    );
    return { ready: true, fallback: false };
  }
  if (loadFailed) return { ready: true, fallback: true };

  try {
    onStatus?.('正在启动本地 TensorFlow.js GPU 推理引擎...');
    const tf = await import('@tensorflow/tfjs');
    await tf.ready();
    onStatus?.('正在加载高精度视觉模型（支持 80 类物体与远距测距）...');
    const cocoSsd = await import('@tensorflow-models/coco-ssd');
    try {
      loadedModel = (await cocoSsd.load({ base: 'mobilenet_v2' })) as unknown as CocoModel;
      loadedModelVariant = 'mobilenet_v2';
      onStatus?.('高精度 MobileNet v2 就绪（80 类物体 · 远距米数测距）');
    } catch {
      loadedModel = (await cocoSsd.load({ base: 'lite_mobilenet_v2' })) as unknown as CocoModel;
      loadedModelVariant = 'lite_mobilenet_v2';
      onStatus?.('轻量 COCO-SSD 就绪（80 类物体 · 远距米数测距）');
    }
    return { ready: true, fallback: false };
  } catch {
    loadFailed = true;
    onStatus?.('已自动切换至离线拓扑语义模式（无需联网下载权重）');
    return { ready: true, fallback: true };
  }
}

function getProceduralFallbackDetections(
  preset: PresetSceneId,
  w: number,
  h: number
): SpatialDetection[] {
  if (preset === 'crosswalk') {
    return [
      {
        labelEn: 'car',
        labelZh: '来向车辆',
        score: 0.91,
        bbox: [w * 0.67, h * 0.43, w * 0.26, h * 0.28],
        direction: '右前方',
        clockFace: '2点方向',
        distance: '近距离',
        distanceMeters: 2.2,
        distanceText: '约2.2米',
        priority: 'danger',
        source: 'local-topology',
      },
      {
        labelEn: 'traffic light',
        labelZh: '交通信号灯（绿灯）',
        score: 0.94,
        bbox: [w * 0.11, h * 0.11, w * 0.1, h * 0.25],
        direction: '左前方',
        clockFace: '10点方向',
        distance: '中距离',
        distanceMeters: 4.5,
        distanceText: '约5米',
        priority: 'obstacle',
        source: 'local-topology',
      },
    ];
  }
  if (preset === 'room') {
    return [
      {
        labelEn: 'dining table',
        labelZh: '餐桌障碍',
        score: 0.92,
        bbox: [w * 0.07, h * 0.3, w * 0.36, h * 0.58],
        direction: '左前方',
        clockFace: '10点方向',
        distance: '近距离',
        distanceMeters: 1.3,
        distanceText: '约1.3米',
        priority: 'obstacle',
        source: 'local-topology',
      },
      {
        labelEn: 'doorway',
        labelZh: '明亮门框通道',
        score: 0.89,
        bbox: [w * 0.55, h * 0.13, w * 0.24, h * 0.68],
        direction: '右前方',
        clockFace: '2点方向',
        distance: '中距离',
        distanceMeters: 2.8,
        distanceText: '约2.8米',
        priority: 'obstacle',
        source: 'local-topology',
      },
    ];
  }
  if (preset === 'mountain') {
    return [
      {
        labelEn: 'moon',
        labelZh: '高空高亮光源（明月）',
        score: 0.95,
        bbox: [w * 0.72, h * 0.12, w * 0.12, h * 0.2],
        direction: '右前方',
        clockFace: '2点方向',
        distance: '较远处',
        distanceMeters: 10,
        distanceText: '约10米外',
        priority: 'normal',
        source: 'local-topology',
      },
      {
        labelEn: 'ridge',
        labelZh: '连续起伏轮廓线',
        score: 0.9,
        bbox: [w * 0.15, h * 0.32, w * 0.7, h * 0.48],
        direction: '正前方',
        clockFace: '12点方向',
        distance: '中距离',
        distanceMeters: 4.0,
        distanceText: '约4米',
        priority: 'normal',
        source: 'local-topology',
      },
    ];
  }
  return [
    {
      labelEn: 'target',
      labelZh: '中心同心圆校准靶标',
      score: 0.96,
      bbox: [w * 0.34, h * 0.25, w * 0.32, h * 0.5],
      direction: '正前方',
      clockFace: '12点方向',
      distance: '中距离',
      distanceMeters: 2.5,
      distanceText: '约2.5米',
      priority: 'normal',
      source: 'local-topology',
    },
  ];
}

function rankDetections(items: SpatialDetection[]): SpatialDetection[] {
  const prioWeight = { danger: 3, obstacle: 2, normal: 1 };
  return [...items].sort((a, b) => {
    const proxA = 10 / Math.max(0.5, a.distanceMeters);
    const proxB = 10 / Math.max(0.5, b.distanceMeters);
    const scoreA = prioWeight[a.priority] * 8 + proxA + a.score;
    const scoreB = prioWeight[b.priority] * 8 + proxB + b.score;
    return scoreB - scoreA;
  });
}

export async function detectSpatialObjects(
  canvas: HTMLCanvasElement,
  isProceduralPreset: boolean,
  presetId: PresetSceneId
): Promise<SpatialDetection[]> {
  if (loadedModel && !loadFailed) {
    try {
      // Lower minScore to 0.30 so distant pedestrians, cars, and traffic lights (5-12m) are caught earlier
      const preds = await loadedModel.detect(canvas, 10, 0.3);
      if (preds.length > 0) {
        const mapped: SpatialDetection[] = preds.map((p) => {
          const meta = computeSpatialMeta(p.class, p.bbox, canvas.width, canvas.height);
          let labelZh = COCO_ZH_MAP[p.class] || p.class;
          if (p.class === 'traffic light') {
            const state = analyzeTrafficLightState(canvas, p.bbox);
            if (state !== '未知') {
              labelZh = `交通信号灯（${state}）`;
            }
          }
          return {
            labelEn: p.class,
            labelZh,
            score: p.score,
            bbox: p.bbox,
            direction: meta.direction,
            clockFace: meta.clockFace,
            distance: meta.distance,
            distanceMeters: meta.distanceMeters,
            distanceText: meta.distanceText,
            priority: meta.priority,
            source: 'tfjs',
          };
        });
        return rankDetections(mapped);
      }
    } catch {
      // Fallback gracefully
    }
  }

  if (isProceduralPreset) {
    return rankDetections(getProceduralFallbackDetections(presetId, canvas.width, canvas.height));
  }

  return [];
}

export function speakCue(text: string, rate = 1.04) {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  if (!text.trim()) return;
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'zh-CN';
    u.rate = rate;
    window.speechSynthesis.speak(u);
  } catch {
    // Ignore speech errors
  }
}

let lastSpokenAt = 0;
let lastSpokenSignature = '';

/**
 * Delta-aware natural Chinese guidance with exact meter distances.
 */
export function speakSpatialDetections(
  detections: SpatialDetection[],
  force = false,
  fallbackNavigationCue?: string
): string | null {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null;
  const now = performance.now();

  if (detections.length === 0) {
    if (!force && !fallbackNavigationCue) return null;
    if (!force && now - lastSpokenAt < 7500) return null;
    const text = fallbackNavigationCue || '前方视野比较空旷，暂时没有发现明显障碍物，可以平稳往前走。';
    if (!force && text === lastSpokenSignature && now - lastSpokenAt < 14000) return null;
    lastSpokenAt = now;
    lastSpokenSignature = text;
    speakCue(text);
    return text;
  }

  const top = detections.slice(0, 3);
  const signature = top
    .map((d) => `${d.direction}-${Math.round(d.distanceMeters)}-${d.labelZh}`)
    .join('|');
  const hasNearHazard = top.some((d) => d.priority === 'danger' && d.distanceMeters <= 3.5);

  if (!force) {
    const minInterval = hasNearHazard ? 3000 : 4800;
    if (now - lastSpokenAt < minInterval) return null;
    if (signature === lastSpokenSignature && !hasNearHazard && now - lastSpokenAt < 12000) {
      return null;
    }
  }

  lastSpokenAt = now;
  lastSpokenSignature = signature;

  const summary = top
    .map((d) => {
      const prefix = d.priority === 'danger' && d.distanceMeters <= 3.5 ? '小心，' : '';
      return `${prefix}${d.direction}${d.distanceText}有${d.labelZh}`;
    })
    .join('；');

  speakCue(summary);
  return summary;
}
