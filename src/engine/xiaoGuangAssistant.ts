// "XiaoGuang" (小光) — Conversational, Empathetic Eyes-Free Companion Engine
// Combines real-time spatial perception (meters + clock direction + ground/overhead cues),
// voice control, and warm emotional companionship for blind users walking without a cane.

import type { SpatialDetection } from './aiObjectDetector';
import type { SpatialSceneMetrics } from './visionProcessor';
import { FINDER_TARGETS, LightAndColorReport } from './blindAssistTools';

export interface CompanionContext {
  detections: SpatialDetection[];
  metrics: SpatialSceneMetrics;
  groundBlocked: boolean; // dense edges in lower center strip (curb/step/low obstacle)
  overheadBlocked: boolean; // dense edges in upper center strip (branch/sign/overhang)
  lightReport: LightAndColorReport | null;
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

export function buildDetailedSceneReport(ctx: CompanionContext): {
  text: string;
  mood: 'calm' | 'alert' | 'warm';
} {
  const { detections, metrics, groundBlocked, overheadBlocked } = ctx;
  const parts: string[] = [];
  let mood: 'calm' | 'alert' | 'warm' = 'calm';

  if (detections.length > 0) {
    const top = detections.slice(0, 3);
    const hasDanger = top.some((d) => d.priority === 'danger' && d.distanceMeters <= 3.5);
    if (hasDanger) mood = 'alert';

    const desc = top
      .map((d) => `${d.direction}（${d.clockFace}）${d.distanceText}有${d.labelZh}`)
      .join('；');
    parts.push(hasDanger ? `慢一点，先注意一下：${desc}。` : `我帮你看了一下：${desc}。`);
  }

  if (groundBlocked) {
    mood = 'alert';
    parts.push('另外注意脚下，正前方1到2米路面轮廓很密，可能有台阶、路沿或低矮障碍，迈步慢一点。');
  } else if (overheadBlocked) {
    mood = 'alert';
    parts.push('头顶上方偏密，小心可能有低垂树枝或广告牌。');
  }

  if (metrics.centerDensity > 45) {
    const steer = metrics.leftDensity <= metrics.rightDensity ? '向左偏两步' : '向右偏两步';
    parts.push(`正前方比较挤，建议${steer}绕过去，我一直帮你盯着。`);
  } else if (parts.length === 0) {
    parts.push('前方大约3到5米内都很开阔，脚下也平整，放心平稳往前走就行，我在呢。');
  } else if (mood !== 'alert') {
    parts.push('整体还算宽敞，按你舒服的节奏走就好。');
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
      text: '我在呢，没听太清楚，你可以再说一遍，或者问我“前面有什么”。',
      action: { type: 'none' },
      mood: 'warm',
    };
  }

  // 1. Emotional support & companionship (情绪价值与街行安抚)
  if (
    text.includes('害怕') ||
    text.includes('怕') ||
    text.includes('紧张') ||
    text.includes('慌') ||
    text.includes('不敢')
  ) {
    const safeHint =
      ctx.detections.length > 0
        ? `现在最近的物体是${ctx.detections[0].direction}${ctx.detections[0].distanceText}的${ctx.detections[0].labelZh}。`
        : '现在前方很空旷，没有紧贴你的障碍。';
    return {
      text: `别怕，深呼吸，小光一直陪着你呢。咱们不着急，慢慢走就行。${safeHint}有我在，放心。`,
      action: { type: 'none' },
      mood: 'warm',
    };
  }

  if (text.includes('累') || text.includes('辛苦') || text.includes('歇')) {
    return {
      text: '走累了吧，你已经走得很棒啦！如果旁边有长椅或安全角落，咱们先歇两分钟喝口水，我继续帮你看着周围。',
      action: { type: 'none' },
      mood: 'warm',
    };
  }

  if (text.includes('谢谢') || text.includes('感谢') || text.includes('真棒') || text.includes('好用')) {
    return {
      text: '不客气呀！能做你的眼睛陪你一起出门，我特别开心。随时叫我。',
      action: { type: 'none' },
      mood: 'warm',
    };
  }

  if (
    text.includes('你在吗') ||
    text.includes('小光') && text.length <= 5 ||
    text.includes('你好') ||
    text.includes('陪我')
  ) {
    return {
      text: '我在呢！一直开着眼睛帮你看路呢。想问前面几米有啥、找东西、或者单纯聊两句，直接说就行。',
      action: { type: 'none' },
      mood: 'warm',
    };
  }

  // 2. Voice Controls (随时操控)
  if (text.includes('取消寻物') || text.includes('不找了') || text.includes('别找了')) {
    return {
      text: '好嘞，已经关掉寻物滴答声了，咱们继续正常看路。',
      action: { type: 'clear_finder' },
      mood: 'calm',
    };
  }

  if (text.includes('静音') || text.includes('闭嘴') || text.includes('安静')) {
    return {
      text: '好的，我先把背景声场调静音，有需要你随时叫我。',
      action: { type: 'set_volume', delta: 'mute' },
      mood: 'calm',
    };
  }

  if (text.includes('大声') || text.includes('音量大') || text.includes('听不清')) {
    return {
      text: '好，我把声场音量调大一点，现在听起来清楚些了吗？',
      action: { type: 'set_volume', delta: 0.2 },
      mood: 'calm',
    };
  }

  if (text.includes('小声') || text.includes('音量小') || text.includes('太吵')) {
    return {
      text: '明白，已经帮你把声场调轻柔一点了。',
      action: { type: 'set_volume', delta: -0.2 },
      mood: 'calm',
    };
  }

  if (text.includes('暂停') || text.includes('先停') || text.includes('休息一下')) {
    return {
      text: '好的，先暂停扫描歇一会儿，随时点屏幕或者说“继续”我就回来。',
      action: { type: 'pause_all' },
      mood: 'warm',
    };
  }

  if (text.includes('继续') || text.includes('开始看路') || text.includes('开启')) {
    return {
      text: '收到，继续帮你盯着前方路况！',
      action: { type: 'resume_all' },
      mood: 'calm',
    };
  }

  if (text.includes('关掉声波') || text.includes('关闭声场') || text.includes('只听语音')) {
    return {
      text: '没问题，已经关掉 3D 声波，只保留我的中文语音提醒。',
      action: { type: 'set_scan', enabled: false },
      mood: 'calm',
    };
  }

  if (text.includes('打开声波') || text.includes('开启声场')) {
    return {
      text: '好嘞，3D 立体声场已经重新打开。',
      action: { type: 'set_scan', enabled: true },
      mood: 'calm',
    };
  }

  // 3. Find specific item (3D 寻物)
  if (text.includes('找') || text.includes('在哪')) {
    for (const t of FINDER_TARGETS) {
      if (t.keywords.some((k) => text.includes(k))) {
        return {
          text: `好，我帮你盯着找${t.label}！你慢慢左右转一下手机，一旦扫到，耳机对应方向就会响起清脆的滴答声。`,
          action: { type: 'set_finder', targetId: t.id, targetLabel: t.label },
          mood: 'calm',
        };
      }
    }
  }

  // 4. Light & color inquiry (测光识色)
  if (
    (text.includes('灯') && !text.includes('红绿灯') && !text.includes('信号灯')) ||
    text.includes('亮') ||
    text.includes('天黑') ||
    text.includes('颜色') ||
    text.includes('衣服')
  ) {
    return {
      text: '我马上帮你测一下光线和正中间的颜色。',
      action: { type: 'check_light_color' },
      mood: 'calm',
    };
  }

  // 5. Road / Distance / Safety / Default live scene answer
  if (
    text.includes('前面') ||
    text.includes('有什么') ||
    text.includes('安全') ||
    text.includes('能走') ||
    text.includes('可以走') ||
    text.includes('几米') ||
    text.includes('多远') ||
    text.includes('障碍') ||
    text.includes('脚下') ||
    text.includes('红绿灯') ||
    text.includes('周围') ||
    text.includes('路况')
  ) {
    const report = buildDetailedSceneReport(ctx);
    return { text: report.text, action: { type: 'none' }, mood: report.mood };
  }

  // Friendly fallback that still gives helpful current situation
  const report = buildDetailedSceneReport(ctx);
  return {
    text: `我听到你说“${text}”啦。顺便告诉你眼前的路况：${report.text}`,
    action: { type: 'none' },
    mood: 'warm',
  };
}
