// "XiaoGuang" (小光) — Fail-Safe Conversational Assistant Engine
// Strictly separates confirmed perception outputs from unverified safety claims or navigation orders.
// Never declares a path "safe / clear / flat" from empty detections, and never instructs the user
// to step into traffic or sidestep blindly based on edge density alone.

import type { SpatialDetection } from './aiObjectDetector';
import type { SpatialSceneMetrics } from './visionProcessor';
import type { EchoSafeSnapshot } from './echoSafe';
import { FINDER_TARGETS } from './blindAssistTools';

export interface CompanionContext {
  safeState: EchoSafeSnapshot;
  detections: SpatialDetection[];
  metrics: SpatialSceneMetrics;
  lowRegionEdgeDense: boolean; // Sobel lower-center edge density (NOT confirmed ground step)
  highRegionEdgeDense: boolean; // Sobel upper-center edge density (NOT confirmed overhead branch)
}

export type AssistantAction =
  | { type: 'none' }
  | { type: 'set_finder'; targetId: string; targetLabel: string }
  | { type: 'clear_finder' }
  | { type: 'check_light_color' }
  | { type: 'set_scan'; enabled: boolean }
  | { type: 'set_volume'; delta: number | 'mute' | 'unmute' }
  | { type: 'pause_all' }
  | { type: 'resume_all' };

export interface AssistantReply {
  text: string;
  action: AssistantAction;
  mood: 'calm' | 'alert' | 'warm';
}

/**
 * Fail-Safe scene report: reports only verified system state & detected objects,
 * never outputs unverified safety guarantees or blind steering commands.
 */
export function buildDetailedSceneReport(ctx: CompanionContext): {
  text: string;
  mood: 'calm' | 'alert' | 'warm';
} {
  const { safeState, detections, metrics, lowRegionEdgeDense, highRegionEdgeDense } = ctx;

  if (safeState.state === 'IDLE') {
    return {
      text: '感知系统尚未启动，目前没有实时画面。请先启动后再查询环境信息。',
      mood: 'calm',
    };
  }
  if (safeState.state === 'INITIALIZING') {
    return {
      text: '正在申请权限并加载视觉模型，请稍候。',
      mood: 'calm',
    };
  }
  if (safeState.state === 'UNAVAILABLE') {
    return {
      text: `实时感知当前不可用：${safeState.userSummary}请勿依赖屏幕结果，请以盲杖与现场判断为准。`,
      mood: 'alert',
    };
  }
  if (safeState.state === 'DEMO') {
    return {
      text: '当前处于模拟演示模式，画面并非真实摄像头采集，不能用于真实环境判断。',
      mood: 'calm',
    };
  }
  if (safeState.state === 'DEGRADED') {
    return {
      text: `系统处于降级状态：${safeState.userSummary}当前无法可靠识别物体，请以现场判断为准。`,
      mood: 'alert',
    };
  }

  // State is ACTIVE (live camera + fresh inference)
  const parts: string[] = [];
  let mood: 'calm' | 'alert' | 'warm' = 'calm';

  if (detections.length > 0) {
    const top = detections.slice(0, 3);
    const hasNearDanger = top.some((d) => d.priority === 'danger' && d.distance === '近距离');
    if (hasNearDanger) mood = 'alert';

    const desc = top
      .map((d) => `${d.direction}（${d.clockFace}）${d.distanceText}检测到${d.labelZh}`)
      .join('；');
    parts.push(`当前识别到：${desc}。`);
  } else {
    // FAIL-SAFE: Empty detection NEVER means "safe / clear / flat"
    parts.push(
      `未识别到已知类别物体（视觉轮廓密度：左${metrics.leftDensity}%、中${metrics.centerDensity}%、右${metrics.rightDensity}%）。未检出不代表前方无障碍，请结合盲杖确认。`
    );
  }

  if (lowRegionEdgeDense) {
    parts.push('画面下方边缘轮廓较密，具体物体未知，请注意现场确认。');
  } else if (highRegionEdgeDense) {
    parts.push('画面上方边缘轮廓较密，具体物体未知。');
  }

  return { text: parts.join(''), mood };
}

export function respondToUserUtterance(
  rawText: string,
  ctx: CompanionContext
): AssistantReply {
  const text = rawText.trim();
  if (!text) {
    return {
      text: '未识别到有效语音，你可以问“前面识别到什么”或说出控制指令。',
      action: { type: 'none' },
      mood: 'calm',
    };
  }

  // 0. Traffic light / street-crossing question — ALWAYS explicitly refuse crossing clearance first
  if (
    text.includes('能过马路') ||
    text.includes('过马路') ||
    text.includes('可以过街') ||
    text.includes('过街') ||
    text.includes('红绿灯') ||
    text.includes('绿灯') ||
    text.includes('红灯')
  ) {
    return {
      text: '本原型不支持独立过街安全判定。请勿依据识别结果过马路，请结合有声信号灯、盲杖与现场交通声音判断。',
      action: { type: 'none' },
      mood: 'alert',
    };
  }

  // 1. Calm, honest emotional support (without making false safety promises)
  if (
    text.includes('害怕') ||
    text.includes('怕') ||
    text.includes('紧张') ||
    text.includes('慌') ||
    text.includes('不敢')
  ) {
    const statusNote =
      ctx.safeState.state === 'ACTIVE' && ctx.detections.length > 0
        ? `当前识别到最近目标为${ctx.detections[0].direction}的${ctx.detections[0].labelZh}。`
        : `当前感知状态为：${ctx.safeState.userSummary}。`;
    return {
      text: `先放慢节奏，深呼吸，停在原地确认安全再走。${statusNote}本工具仅作辅助参考，请以盲杖和周围真实声音为准。`,
      action: { type: 'none' },
      mood: 'warm',
    };
  }

  if (text.includes('累') || text.includes('辛苦') || text.includes('歇')) {
    return {
      text: '走累了可以先在安全位置休息一下，需要暂停播报时直接说“暂停”即可。',
      action: { type: 'none' },
      mood: 'warm',
    };
  }

  if (text.includes('谢谢') || text.includes('感谢')) {
    return {
      text: '不客气，随时可以问我当前识别状态。',
      action: { type: 'none' },
      mood: 'warm',
    };
  }

  if (text.includes('你在吗') || text.includes('你好') || (text.includes('小光') && text.length <= 5)) {
    return {
      text: `我在。当前系统状态：${ctx.safeState.userSummary}。`,
      action: { type: 'none' },
      mood: 'calm',
    };
  }

  // 2. Explicit voice controls
  if (text.includes('取消寻物') || text.includes('不找了') || text.includes('别找了')) {
    return {
      text: '已关闭 3D 寻物信标。',
      action: { type: 'clear_finder' },
      mood: 'calm',
    };
  }

  if (text.includes('静音') || text.includes('闭嘴') || text.includes('安静')) {
    return {
      text: '已静音连续声场。',
      action: { type: 'set_volume', delta: 'mute' },
      mood: 'calm',
    };
  }

  if (text.includes('大声') || text.includes('音量大') || text.includes('听不清')) {
    return {
      text: '已调高声场音量。',
      action: { type: 'set_volume', delta: 0.2 },
      mood: 'calm',
    };
  }

  if (text.includes('小声') || text.includes('音量小') || text.includes('太吵')) {
    return {
      text: '已调低声场音量。',
      action: { type: 'set_volume', delta: -0.2 },
      mood: 'calm',
    };
  }

  if (text.includes('暂停') || text.includes('先停')) {
    return {
      text: '已暂停实时感知。',
      action: { type: 'pause_all' },
      mood: 'calm',
    };
  }

  if (text.includes('继续') || text.includes('开启')) {
    return {
      text: '正在恢复实时感知。',
      action: { type: 'resume_all' },
      mood: 'calm',
    };
  }

  if (text.includes('关掉声波') || text.includes('关闭声场') || text.includes('只听语音')) {
    return {
      text: '已关闭连续 3D 声场，仅保留状态与查询语音。',
      action: { type: 'set_scan', enabled: false },
      mood: 'calm',
    };
  }

  if (text.includes('打开声波') || text.includes('开启声场')) {
    return {
      text: '已打开连续 3D 声场。',
      action: { type: 'set_scan', enabled: true },
      mood: 'calm',
    };
  }

  // 3. Item finder
  if (text.includes('找') || text.includes('在哪')) {
    for (const t of FINDER_TARGETS) {
      if (t.keywords.some((k) => text.includes(k))) {
        return {
          text: `已设定寻物目标：${t.label}。在画面中检测到时会从对应方位发出提示音。`,
          action: { type: 'set_finder', targetId: t.id, targetLabel: t.label },
          mood: 'calm',
        };
      }
    }
  }

  // 4. Light & color
  if (
    ((text.includes('灯') && !text.includes('红绿灯') && !text.includes('信号灯')) ||
      text.includes('亮') ||
      text.includes('天黑') ||
      text.includes('颜色') ||
      text.includes('衣服'))
  ) {
    return {
      text: '正在测量当前画面亮度与正中颜色。',
      action: { type: 'check_light_color' },
      mood: 'calm',
    };
  }

  // 5. Scene report
  const report = buildDetailedSceneReport(ctx);
  return { text: report.text, action: { type: 'none' }, mood: report.mood };
}
