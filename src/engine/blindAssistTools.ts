// Experimental On-Device Assistive Utilities
// 1. Traffic-light box pixel color classifier (EXPERIMENTAL color demo ONLY — NOT a street-crossing signal)
// 2. Ambient light & center color detector
// 3. Chinese voice command parser

export type TrafficLightColor = '红色像素' | '黄色像素' | '绿色像素' | '未知';

export interface TrafficLightAnalysis {
  color: TrafficLightColor;
  confidence: number; // 0..1
  /** Explicitly never grants pedestrian street-crossing clearance */
  isCrossingClearance: false;
}

/**
 * Pure pixel-array analyzer for unit testing (backlit, dark, glare, multi-lamp interference).
 */
export function classifyTrafficLightPixels(data: Uint8ClampedArray): TrafficLightAnalysis {
  let redScore = 0;
  let yellowScore = 0;
  let greenScore = 0;
  let brightNeutralCount = 0;
  const totalPixels = Math.max(1, data.length / 4);

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);

    if (max > 235 && max - min < 22) {
      brightNeutralCount++;
      continue;
    }
    if (max < 115 || max - min < 50) continue;

    if (r > 170 && g < 110 && b < 110) {
      redScore += r - Math.max(g, b);
    } else if (g > 160 && r < 120 && b < 145) {
      greenScore += g - Math.max(r, b);
    } else if (r > 180 && g > 145 && b < 105) {
      yellowScore += (r + g) / 2 - b;
    }
  }

  // Reject severe glare / overexposure
  if (brightNeutralCount / totalPixels > 0.45) {
    return { color: '未知', confidence: 0, isCrossingClearance: false };
  }

  const totalScore = redScore + yellowScore + greenScore;
  const best = Math.max(redScore, yellowScore, greenScore);
  if (best < 450 || totalScore === 0) {
    return { color: '未知', confidence: 0, isCrossingClearance: false };
  }

  // Reject ambiguous multi-lamp / mixed interference where winner doesn't clearly dominate
  const dominance = best / totalScore;
  if (dominance < 0.68) {
    return { color: '未知', confidence: Math.round(dominance * 100) / 100, isCrossingClearance: false };
  }

  const color: TrafficLightColor =
    best === greenScore ? '绿色像素' : best === redScore ? '红色像素' : '黄色像素';
  return { color, confidence: Math.round(dominance * 100) / 100, isCrossingClearance: false };
}

/**
 * Multi-frame stabilizer: requires N consecutive agreeing frames before reporting a stable color;
 * otherwise returns '未知' to avoid single-frame flicker.
 */
export class TrafficLightStabilizer {
  private history: TrafficLightColor[] = [];
  private requiredFrames: number;

  constructor(requiredFrames = 3) {
    this.requiredFrames = Math.max(2, requiredFrames);
  }

  public push(color: TrafficLightColor): TrafficLightColor {
    this.history.push(color);
    if (this.history.length > this.requiredFrames) {
      this.history.shift();
    }
    if (this.history.length < this.requiredFrames) return '未知';
    const first = this.history[0];
    if (first === '未知') return '未知';
    return this.history.every((c) => c === first) ? first : '未知';
  }

  public reset() {
    this.history = [];
  }
}

export function analyzeTrafficLightState(
  canvas: HTMLCanvasElement,
  bbox: [number, number, number, number]
): TrafficLightColor {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return '未知';

  const [bx, by, bw, bh] = bbox;
  const x = Math.max(0, Math.floor(bx));
  const y = Math.max(0, Math.floor(by));
  const w = Math.max(1, Math.min(canvas.width - x, Math.ceil(bw)));
  const h = Math.max(1, Math.min(canvas.height - y, Math.ceil(bh)));

  const data = ctx.getImageData(x, y, w, h).data;
  return classifyTrafficLightPixels(data).color;
}

export interface LightAndColorReport {
  brightnessPct: number; // 0..100
  lightLabel: string;
  lampHint: string;
  colorName: string;
  centerHex: string;
  speechText: string;
}

function rgbToChineseColor(r: number, g: number, b: number): string {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2 / 255;
  const d = max - min;
  const s = d === 0 ? 0 : d / (255 - Math.abs(max + min - 255));

  if (l < 0.16) return '深黑色';
  if (l > 0.86 && s < 0.2) return '纯白色';
  if (s < 0.15) {
    if (l < 0.38) return '深灰色';
    if (l < 0.68) return '中灰色';
    return '浅灰白色';
  }

  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
    else if (max === g) h = ((b - r) / d + 2) * 60;
    else h = ((r - g) / d + 4) * 60;
  }

  if (h < 16 || h >= 342) return l < 0.35 ? '深红色' : '红色';
  if (h < 42) return l < 0.42 ? '棕色' : '橙色';
  if (h < 68) return l > 0.72 ? '米黄色' : '黄色';
  if (h < 155) return l < 0.35 ? '深绿色' : '绿色';
  if (h < 198) return '青色';
  if (h < 255) return l < 0.35 ? '藏青色' : '蓝色';
  if (h < 300) return '紫色';
  return '粉红色';
}

/**
 * Measure overall ambient brightness (to tell if room lights are on/off)
 * and identify the dominant color in the center of the camera/image (for clothing/items).
 */
export function analyzeLightAndColor(canvas: HTMLCanvasElement): LightAndColorReport {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return {
      brightnessPct: 0,
      lightLabel: '未知',
      lampHint: '无法读取画面亮度',
      colorName: '未知',
      centerHex: '#161512',
      speechText: '暂时无法读取环境光线与颜色',
    };
  }

  const { width, height } = canvas;
  const full = ctx.getImageData(0, 0, width, height).data;
  let lumSum = 0;
  let lumCount = 0;
  for (let i = 0; i < full.length; i += 16) {
    lumSum += 0.2126 * full[i] + 0.7152 * full[i + 1] + 0.0722 * full[i + 2];
    lumCount++;
  }
  const brightnessPct = Math.round((lumSum / Math.max(1, lumCount) / 255) * 100);

  let lightLabel = '光线适中';
  let lampHint = '室内光线柔和';
  if (brightnessPct < 14) {
    lightLabel = '非常暗';
    lampHint = '环境很暗，室内灯光可能已关闭';
  } else if (brightnessPct < 32) {
    lightLabel = '偏暗';
    lampHint = '光线较弱，可能为夜间或阴影处';
  } else if (brightnessPct < 65) {
    lightLabel = '光线适中';
    lampHint = '室内照明正常或室外多云';
  } else if (brightnessPct < 85) {
    lightLabel = '明亮';
    lampHint = '环境明亮，灯光开启或正对窗户';
  } else {
    lightLabel = '强光';
    lampHint = '正对强光源或阳光直射';
  }

  // Sample central 18% patch for clothing / object color identification
  const rw = Math.max(12, Math.floor(width * 0.18));
  const rh = Math.max(12, Math.floor(height * 0.18));
  const rx = Math.floor((width - rw) / 2);
  const ry = Math.floor((height - rh) / 2);
  const patch = ctx.getImageData(rx, ry, rw, rh).data;

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  const pCount = patch.length / 4;
  for (let i = 0; i < patch.length; i += 4) {
    rSum += patch[i];
    gSum += patch[i + 1];
    bSum += patch[i + 2];
  }
  const rAvg = Math.round(rSum / pCount);
  const gAvg = Math.round(gSum / pCount);
  const bAvg = Math.round(bSum / pCount);
  const colorName = rgbToChineseColor(rAvg, gAvg, bAvg);
  const toHex = (n: number) => n.toString(16).padStart(2, '0');
  const centerHex = `#${toHex(rAvg)}${toHex(gAvg)}${toHex(bAvg)}`;

  const speechText = `环境亮度${brightnessPct}%，${lampHint}；画面正中颜色为${colorName}。`;
  return { brightnessPct, lightLabel, lampHint, colorName, centerHex, speechText };
}

export interface FinderTargetOption {
  id: string; // COCO class or 'none'
  label: string;
  keywords: string[];
}

export const FINDER_TARGETS: FinderTargetOption[] = [
  { id: 'cup', label: '水杯', keywords: ['水杯', '杯子', '茶杯', '瓶', '水瓶'] },
  { id: 'cell phone', label: '手机', keywords: ['手机', '电话'] },
  { id: 'chair', label: '椅子', keywords: ['椅子', '座位', '凳子'] },
  { id: 'person', label: '行人', keywords: ['人', '行人', '朋友'] },
  { id: 'car', label: '车辆', keywords: ['车', '汽车', '来车', '车辆'] },
  { id: 'traffic light', label: '红绿灯', keywords: ['红绿灯', '信号灯', '绿灯', '红灯'] },
  { id: 'backpack', label: '背包', keywords: ['包', '背包', '书包'] },
];

export type VoiceCommandIntent =
  | { type: 'read_scene' }
  | { type: 'light_color' }
  | { type: 'find_item'; targetId: string; targetLabel: string }
  | { type: 'clear_finder' }
  | { type: 'start_scan' }
  | { type: 'stop_scan' }
  | { type: 'unknown'; raw: string };

export function parseVoiceCommand(raw: string): VoiceCommandIntent {
  const text = raw.trim();
  if (!text) return { type: 'unknown', raw };

  if (text.includes('取消') || text.includes('不找了') || text.includes('关闭寻物')) {
    return { type: 'clear_finder' };
  }
  if (text.includes('找') || text.includes('在哪')) {
    for (const t of FINDER_TARGETS) {
      if (t.keywords.some((k) => text.includes(k))) {
        return { type: 'find_item', targetId: t.id, targetLabel: t.label };
      }
    }
  }
  if (
    text.includes('灯') && !text.includes('红绿灯') && !text.includes('信号灯') ||
    text.includes('亮') ||
    text.includes('暗') ||
    text.includes('颜色') ||
    text.includes('什么色')
  ) {
    return { type: 'light_color' };
  }
  if (text.includes('停') || text.includes('静音') || text.includes('闭嘴') || text.includes('关闭扫描')) {
    return { type: 'stop_scan' };
  }
  if (text.includes('开始') || text.includes('启动扫描')) {
    return { type: 'start_scan' };
  }
  if (
    text.includes('有什么') ||
    text.includes('前面') ||
    text.includes('周围') ||
    text.includes('红绿灯') ||
    text.includes('信号灯') ||
    text.includes('播报') ||
    text.includes('看看')
  ) {
    return { type: 'read_scene' };
  }
  return { type: 'unknown', raw: text };
}
