import { describe, it, expect } from 'vitest';
import {
  buildDetailedSceneReport,
  respondToUserUtterance,
} from '../src/engine/xiaoGuangAssistant';
import { createInitialSignals, evaluateEchoSafe } from '../src/engine/echoSafe';

const FORBIDDEN_PHRASES = [
  '可以安全通过',
  '前方没有障碍',
  '脚下平整',
  '脚下也平整',
  '放心往前走',
  '可以放心走',
  '向左偏两步',
  '向右偏两步',
  '平稳前行',
];

describe('Fail-Safe wording & anti-false-reassurance checks', () => {
  const now = 10_000;
  const activeSnap = evaluateEchoSafe(
    {
      ...createInitialSignals(),
      requestedLive: true,
      cameraPermission: 'granted',
      cameraTrackLive: true,
      lastFrameAt: now - 50,
      modelState: 'ready',
      lastInferenceAt: now - 200,
      audioState: 'running',
    },
    now
  );

  it('never generates safety assurances or steering orders on empty detections', () => {
    const report = buildDetailedSceneReport({
      safeState: activeSnap,
      detections: [],
      metrics: {
        leftDensity: 12,
        centerDensity: 55,
        rightDensity: 15,
        edgeCoverage: 18,
        navigationCue: '轮廓密度参考',
      },
      lowRegionEdgeDense: false,
      highRegionEdgeDense: false,
    });

    for (const bad of FORBIDDEN_PHRASES) {
      expect(report.text).not.toContain(bad);
    }
    expect(report.text).toContain('未检出不代表前方无障碍');
  });

  it('explicitly refuses to grant pedestrian street-crossing clearance', () => {
    const reply = respondToUserUtterance('前面绿灯了能过马路吗', {
      safeState: activeSnap,
      detections: [],
      metrics: {
        leftDensity: 10,
        centerDensity: 10,
        rightDensity: 10,
        edgeCoverage: 10,
        navigationCue: '',
      },
      lowRegionEdgeDense: false,
      highRegionEdgeDense: false,
    });
    expect(reply.mood).toBe('alert');
    expect(reply.text).toContain('不支持独立过街安全判定');
  });

  it('reports unavailability clearly when camera/system is UNAVAILABLE', () => {
    const unavail = evaluateEchoSafe(
      { ...createInitialSignals(), requestedLive: true, cameraPermission: 'denied' },
      now
    );
    const report = buildDetailedSceneReport({
      safeState: unavail,
      detections: [],
      metrics: {
        leftDensity: 0,
        centerDensity: 0,
        rightDensity: 0,
        edgeCoverage: 0,
        navigationCue: '',
      },
      lowRegionEdgeDense: false,
      highRegionEdgeDense: false,
    });
    expect(report.mood).toBe('alert');
    expect(report.text).toContain('不可用');
  });

  it('correctly routes voice intents: 开启声场 vs 继续, 静音 vs 取消静音, 寻物, 测光', () => {
    const baseCtx = {
      safeState: activeSnap,
      detections: [],
      metrics: {
        leftDensity: 10,
        centerDensity: 10,
        rightDensity: 10,
        edgeCoverage: 10,
        navigationCue: '',
      },
      lowRegionEdgeDense: false,
      highRegionEdgeDense: false,
    };

    // "开启声场" must NOT be hijacked by "开启" -> resume_all!
    expect(respondToUserUtterance('开启声场', baseCtx).action).toEqual({
      type: 'set_scan',
      enabled: true,
    });
    expect(respondToUserUtterance('关掉声波', baseCtx).action).toEqual({
      type: 'set_scan',
      enabled: false,
    });
    expect(respondToUserUtterance('继续', baseCtx).action).toEqual({ type: 'resume_all' });
    expect(respondToUserUtterance('暂停', baseCtx).action).toEqual({ type: 'pause_all' });

    // Mute & unmute must both exist and be reachable
    const muteRes = respondToUserUtterance('静音', baseCtx);
    expect(muteRes.action).toEqual({ type: 'set_volume', delta: 'mute' });
    expect(muteRes.text).toContain('取消静音');

    expect(respondToUserUtterance('取消静音', baseCtx).action).toEqual({
      type: 'set_volume',
      delta: 'unmute',
    });

    expect(respondToUserUtterance('帮我找水杯', baseCtx).action).toEqual({
      type: 'set_finder',
      targetId: 'cup',
      targetLabel: '水杯',
    });

    expect(respondToUserUtterance('开灯了吗', baseCtx).action).toEqual({
      type: 'check_light_color',
    });
  });
});
