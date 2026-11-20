import React, { useState } from 'react';
import {
  createInitialSignals,
  EchoSafeSignals,
  evaluateEchoSafe,
} from '../engine/echoSafe';

interface FaultScenario {
  id: string;
  label: string;
  desc: string;
  legacyBehavior: string;
  buildSignals: (now: number) => EchoSafeSignals;
}

const FAULT_SCENARIOS: FaultScenario[] = [
  {
    id: 'cam_denied',
    label: '摄像头权限被拒绝',
    desc: '用户拒绝或浏览器拦截摄像头授权',
    legacyBehavior: '旧版：静默回退绘制内置模拟斑马线并继续播报“右前方有车”，将模拟数据混入实时助行。',
    buildSignals: () => ({
      ...createInitialSignals(),
      requestedLive: true,
      cameraPermission: 'denied',
      cameraTrackLive: false,
      modelState: 'ready',
      audioState: 'running',
    }),
  },
  {
    id: 'track_ended',
    label: '摄像头中途断开 / 遮挡过期',
    desc: '视频轨道中断或超过 2.5s 无新帧更新',
    legacyBehavior: '旧版：仅检查 active === true，继续沿用旧帧或空数组，甚至在 24s 后播报“前方依然开阔”。',
    buildSignals: (now) => ({
      ...createInitialSignals(),
      requestedLive: true,
      cameraPermission: 'granted',
      cameraTrackLive: true,
      lastFrameAt: now - 3800,
      modelState: 'ready',
      audioState: 'running',
    }),
  },
  {
    id: 'model_failed',
    label: '端侧神经网络加载失败',
    desc: '弱网或 WebGL 异常导致 COCO-SSD 未就绪',
    legacyBehavior: '旧版：loadOnDeviceAIModel 仍返回 ready: true，上层误以为 AI 正常并在空结果时播报安全。',
    buildSignals: (now) => ({
      ...createInitialSignals(),
      requestedLive: true,
      cameraPermission: 'granted',
      cameraTrackLive: true,
      lastFrameAt: now - 100,
      modelState: 'failed',
      audioState: 'running',
    }),
  },
  {
    id: 'page_hidden',
    label: '手机锁屏 / 页面切入后台',
    desc: 'document.hidden === true，浏览器挂起媒体流',
    legacyBehavior: '旧版：未监听 visibilitychange，界面仍显示“运行中”，误导用户以为支持后台持续导航。',
    buildSignals: (now) => ({
      ...createInitialSignals(),
      requestedLive: true,
      cameraPermission: 'granted',
      cameraTrackLive: true,
      lastFrameAt: now - 120,
      modelState: 'ready',
      pageHidden: true,
      audioState: 'running',
    }),
  },
];

const CLAIM_ROWS: Array<{
  claim: string;
  status: string;
  evidence: string;
}> = [
  {
    claim: 'EchoSafe 失效保护状态机与数据隔离',
    status: '已实现且已测试',
    evidence: 'src/engine/echoSafe.ts + tests/echoSafe.test.ts 自动化单元测试通过',
  },
  {
    claim: '统一语音/音频优先级调度（防覆盖与冷却）',
    status: '已实现且已测试',
    evidence: 'src/engine/audioScheduler.ts + tests/audioScheduler.test.ts 确定性测试通过',
  },
  {
    claim: '单目测距输入校验与 MAE/P90 误差统计工具',
    status: '仅有模拟验证',
    evidence: 'src/engine/aiObjectDetector.ts + tests/distance.test.ts（0.5–10m 合成基准）；真实相机标定待实测',
  },
  {
    claim: '单路麦克风环境异常声响检测（非 360° 定位）',
    status: '仅有模拟验证',
    evidence: 'src/engine/acousticThreatDetector.ts + tests/acoustic.test.ts；真实街道噪音误报率待实测',
  },
  {
    claim: '交通信号灯框内像素颜色识别（非过街判定）',
    status: '仅有模拟验证',
    evidence: 'src/engine/blindAssistTools.ts + tests/trafficLight.test.ts（逆光/眩光/多灯干扰用例）',
  },
  {
    claim: '真实视障用户受控环境共创实验',
    status: '尚未完成',
    evidence: '已在 docs/USER_STUDY_PROTOCOL.md 建立完整方案与知情同意框架，结果留空待实测',
  },
  {
    claim: '手机锁屏后台持续摄像头导航',
    status: '受浏览器或设备限制，当前无法实现',
    evidence: 'Web 浏览器安全策略会在后台挂起摄像头采集；EchoSafe 会明确进入 UNAVAILABLE 状态',
  },
];

export const ValidationPanel: React.FC = () => {
  const [faultId, setFaultId] = useState<string>('cam_denied');
  const now = 100_000;
  const fault = FAULT_SCENARIOS.find((f) => f.id === faultId) || FAULT_SCENARIOS[0];
  const evaluated = evaluateEchoSafe(fault.buildSignals(now), now);

  return (
    <div className="space-y-6">
      {/* Intro */}
      <div className="panel p-6">
        <div className="eyebrow">06 · Verification & Boundaries</div>
        <h2 className="title-serif text-[22px] mt-1">技术验证、失效保护对比与能力边界</h2>
        <p className="text-[13.5px] text-ink-2 mt-2 max-w-3xl leading-relaxed">
          本页集中呈现 EchoSight 的工程可靠性验证。我们严格区分「真实摄像头数据」与「模拟演示数据」，
          如实标注已测试项与待实测项，不将浏览器研究原型宣传为可替代白手杖的户外助行产品。
        </p>
      </div>

      {/* Interactive EchoSafe vs Legacy Fault Comparison */}
      <div className="panel p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="eyebrow">Innovation 01 · EchoSafe Contrast</div>
            <h3 className="text-[16px] font-semibold text-ink mt-0.5">
              失效保护对比演示：原始实现 vs EchoSafe 状态机
            </h3>
          </div>
          <span className="chip">点击下方故障场景实时计算</span>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
          {FAULT_SCENARIOS.map((f) => (
            <button
              key={f.id}
              type="button"
              data-active={f.id === faultId}
              onClick={() => setFaultId(f.id)}
              className="tile"
            >
              <div className="text-[13px] font-semibold text-ink">{f.label}</div>
              <div className="text-[11px] text-ink-3 mt-0.5">{f.desc}</div>
            </button>
          ))}
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="panel-sunken p-4 space-y-2 border-l-2 border-l-warn">
            <div className="eyebrow text-warn">原始实现（高风险）</div>
            <p className="text-[13px] text-ink-2 leading-relaxed">{fault.legacyBehavior}</p>
          </div>
          <div className="panel-sunken p-4 space-y-2 border-l-2 border-l-ok">
            <div className="flex items-center justify-between">
              <span className="eyebrow text-ok">EchoSafe 状态机输出</span>
              <span className="chip num">{evaluated.state}</span>
            </div>
            <p className="text-[13px] text-ink leading-relaxed">{evaluated.userSummary}</p>
            <div className="text-[11.5px] text-ink-3 num">
              dataSource: {evaluated.dataSource} · canTrustLivePerception: {String(evaluated.canTrustLivePerception)}
            </div>
          </div>
        </div>
      </div>

      {/* Claim-Evidence Matrix */}
      <div className="panel overflow-hidden">
        <div className="px-6 pt-5 pb-4 border-b border-line flex items-center justify-between">
          <h3 className="text-[15px] font-semibold text-ink">能力主张与验证证据矩阵</h3>
          <span className="eyebrow">Claim-Evidence Matrix</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12.5px] border-collapse">
            <thead>
              <tr className="border-b border-line bg-sunken/70 text-ink-2">
                <th className="py-3 px-5 font-semibold">能力 / 机制</th>
                <th className="py-3 px-4 font-semibold whitespace-nowrap">验证状态</th>
                <th className="py-3 px-5 font-semibold">当前证据与边界</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {CLAIM_ROWS.map((row) => (
                <tr key={row.claim}>
                  <td className="py-3 px-5 font-medium text-ink">{row.claim}</td>
                  <td className="py-3 px-4 whitespace-nowrap">
                    <span className="chip">{row.status}</span>
                  </td>
                  <td className="py-3 px-5 text-ink-2">{row.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
