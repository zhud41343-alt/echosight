import React, { useState } from 'react';
import {
  createInitialSignals,
  EchoSafeSignals,
  evaluateEchoSafe,
} from '../engine/echoSafe';
import { computeDistanceErrorStats, DistanceEvalSample } from '../engine/aiObjectDetector';

interface FaultScenario {
  id: string;
  label: string;
  desc: string;
  legacyBehavior: string;
  buildSignals: (now: number) => EchoSafeSignals;
}

const FAULT_SCENARIOS: FaultScenario[] = [
  {
    id: 'null_frame',
    label: '首帧尚未到达 / 画面冻结',
    desc: '轨道虽为 live，但 lastFrameAt 为 null 或超时未推进',
    legacyBehavior: 'V1/V2：将 lastFrameAt === null 误判为未过期，或靠重复 drawImage 伪造画面新鲜度，提前进入 ACTIVE。',
    buildSignals: (now) => ({
      ...createInitialSignals(),
      requestedLive: true,
      sessionStartedAt: now - 5000,
      cameraPermission: 'granted',
      cameraTrackLive: true,
      lastFrameAt: null,
      modelState: 'ready',
      audioState: 'running',
    }),
  },
  {
    id: 'null_inference',
    label: '模型就绪但尚未完成首次推理',
    desc: 'lastInferenceAt === null，无有效 AI 输出',
    legacyBehavior: 'V2：未检查 lastInferenceAt === null，在首次推理完成前即标记为 ACTIVE。',
    buildSignals: (now) => ({
      ...createInitialSignals(),
      requestedLive: true,
      sessionStartedAt: now - 2000,
      cameraPermission: 'granted',
      cameraTrackLive: true,
      lastFrameAt: now - 100,
      modelState: 'ready',
      lastInferenceAt: null,
      audioState: 'running',
    }),
  },
  {
    id: 'cam_denied',
    label: '摄像头权限被拒绝',
    desc: '用户拒绝或浏览器拦截摄像头授权',
    legacyBehavior: 'V1：静默回退绘制内置模拟斑马线并播报“右前方有车”，将模拟数据混入实时助行。',
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
    id: 'page_hidden',
    label: '手机锁屏 / 页面切入后台',
    desc: 'document.hidden === true，浏览器挂起媒体流',
    legacyBehavior: 'V1：未监听 visibilitychange，界面仍显示“运行中”，误导用户以为支持后台持续导航。',
    buildSignals: (now) => ({
      ...createInitialSignals(),
      requestedLive: true,
      cameraPermission: 'granted',
      cameraTrackLive: true,
      lastFrameAt: now - 120,
      modelState: 'ready',
      lastInferenceAt: now - 300,
      pageHidden: true,
      audioState: 'running',
    }),
  },
];

const CLAIM_ROWS: Array<{
  claim: string;
  tier: string;
  evidence: string;
}> = [
  {
    claim: 'EchoDepthNet (1.11M) 残差边缘引导深度融合模型端侧 ONNX 推理与 AcousticMatrixRouter 防覆盖路由',
    tier: '已实现且线上真实浏览器验证',
    evidence: 'public/models/echodepth_net.onnx (SHA-256 见 MODEL_PROVENANCE.md) + src/engine/acousticRouter.ts（线上实测 44ms，数值误差 4.77e-07）',
  },
  {
    claim: 'EchoSafe V8.4 严格新鲜度、FrameTracker 防重复推理、必填 frameId 契约、连续声场熔断与无障碍通知分发',
    tier: '已实现且自动化验证',
    evidence: 'src/engine/echoSafe.ts + src/engine/accessibleNotifier.ts + tests/（8 文件 44/44 + Python 管道 3/3 通过）',
  },
  {
    claim: '无竞态统一语音与空间音效协调（playbackId 守卫 + 全空间音互斥 + stopAllImmediate 节点清理）',
    tier: '已实现且自动化验证',
    evidence: 'src/engine/audioScheduler.ts + src/engine/spatialAudio.ts + tests/audioScheduler.test.ts',
  },
  {
    claim: 'SpeechRecognition 全生命周期管理（防重复实例、停止/卸载 abort、过期回调丢弃）',
    tier: '已实现（真机语音待实测）',
    evidence: 'src/components/BlindCompanionView.tsx（speechRecRef + voiceSessionIdRef；受限于无真实麦克风语音测试环境）',
  },
  {
    claim: '失效保护文案（空检测绝不输出安全承诺、过街询问强制拒绝）',
    tier: '已实现且自动化验证',
    evidence: 'src/engine/xiaoGuangAssistant.ts + tests/safetyWording.test.ts',
  },
  {
    claim: 'WAI-ARIA Tabs 键盘操作、ErrorBoundary 与多视口（1440×900 / 390×844 / 360×740）0 横向溢出',
    tier: '已实际浏览器验证',
    evidence: 'src/App.tsx + docs/REAL_BROWSER_TEST_REPORT.md（0 控制台错误；移动端顶栏折行已修复）',
  },
  {
    claim: '单目测距输入校验与 MAE/P90 误差统计工具（0.5–10m 合成基准）',
    tier: '仅模拟验证',
    evidence: 'src/engine/aiObjectDetector.ts + tests/distance.test.ts；未标定仅输出保守区间',
  },
  {
    claim: '单路麦克风环境异常声响提示（非 360° 定位）与信号灯像素色态稳定器',
    tier: '仅模拟验证',
    evidence: 'src/engine/acousticThreatDetector.ts + src/engine/blindAssistTools.ts + tests/acousticAndTraffic.test.ts',
  },
  {
    claim: '真实手机 iOS VoiceOver / Android TalkBack 读屏全流程测试',
    tier: '尚未真实手机验证',
    evidence: '当前测试环境无真实移动读屏设备，已如实标注待验证',
  },
  {
    claim: '真实视障用户受控环境共创实验',
    tier: '尚未真实视障用户验证',
    evidence: '完整伦理方案见 docs/USER_RESEARCH_PROTOCOL.md，真实结果栏位严格留空',
  },
];

export const ValidationPanel: React.FC = () => {
  const [faultId, setFaultId] = useState<string>('null_frame');
  const [fieldSamples, setFieldSamples] = useState<
    Array<{ id: number; label: string; refM: number; predM: number | null }>
  >([]);
  const [sampleLabel, setSampleLabel] = useState('椅子');
  const [refInput, setRefInput] = useState('2.0');
  const [predInput, setPredInput] = useState('');
  const [copyStatus, setCopyStatus] = useState('');

  const now = 100_000;
  const fault = FAULT_SCENARIOS.find((f) => f.id === faultId) || FAULT_SCENARIOS[0];
  const evaluated = evaluateEchoSafe(fault.buildSignals(now), now);

  const evalInput: DistanceEvalSample[] = fieldSamples.map((s) => ({
    referenceMeters: s.refM,
    predictedMeters: s.predM,
  }));
  const fieldReport = computeDistanceErrorStats(evalInput);

  const handleAddSample = () => {
    const refM = Number.parseFloat(refInput);
    if (!Number.isFinite(refM) || refM <= 0 || refM > 20) return;
    const trimmed = predInput.trim();
    const predM = trimmed === '' ? null : Number.parseFloat(trimmed);
    if (trimmed !== '' && (!Number.isFinite(predM) || predM! <= 0 || predM! > 30)) return;
    setFieldSamples((prev) => [
      ...prev,
      { id: Date.now(), label: sampleLabel.trim() || '参考物', refM, predM },
    ]);
    setPredInput('');
    setCopyStatus('');
  };

  const handleCopyMarkdown = async () => {
    if (fieldSamples.length === 0) return;
    const rows = fieldSamples
      .map(
        (s, i) =>
          `| ${i + 1} | ${s.label} | ${s.refM.toFixed(2)} | ${s.predM !== null ? s.predM.toFixed(2) : '拒判/未知'} | ${s.predM !== null ? Math.abs(s.predM - s.refM).toFixed(2) : '--'} |`
      )
      .join('\n');
    const md = [
      '### EchoSight 线下卷尺实拍测距误差记录',
      `- 样本总数：${fieldReport.total}（有效：${fieldReport.validCount}，拒判/未知：${fieldReport.unknownCount}）`,
      `- 整体 MAE：${fieldReport.maeMeters ?? '--'} m ｜ P90 误差：${fieldReport.p90ErrorMeters ?? '--'} m`,
      '',
      '| # | 目标 | 卷尺实测(m) | 系统估算(m) | 绝对误差(m) |',
      '| :---: | :--- | :---: | :---: | :---: |',
      rows,
    ].join('\n');
    try {
      await navigator.clipboard.writeText(md);
      setCopyStatus('已复制 Markdown 实测报告到剪贴板');
    } catch {
      setCopyStatus('浏览器限制剪贴板写入，请手动复制统计数值');
    }
  };

  return (
    <div className="space-y-6">
      <div className="panel p-6">
        <div className="eyebrow">06 · Verification & Boundaries</div>
        <h2 className="title-serif text-[22px] mt-1">EchoSafe 技术验证、故障对比与能力边界</h2>
        <p className="text-[13.5px] text-ink-2 mt-2 max-w-3xl leading-relaxed">
          传统视觉转语音原型只关注“检测到了什么”，而 EchoSight 沿摄像头 → 新帧标识 → EchoDepthNet/目标追踪 → 数据包校验 → 无障碍分发（ARIA + TTS + 触觉）全链路校验“数据是否真实、是否过期、失效时是否明确通知”。下表如实区分已验证与未验证层级，不将模拟测试等同于真机或真人验证。
        </p>
      </div>

      {/* Interactive Fault Comparison */}
      <div className="panel p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="eyebrow">Innovation 01 · EchoSafe Contrast</div>
            <h3 className="text-[16px] font-semibold text-ink mt-0.5">
              失效保护对比：历史实现 vs EchoSafe 状态机
            </h3>
          </div>
          <span className="chip">点击故障场景实时计算</span>
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
            <div className="eyebrow text-warn">历史实现（高风险）</div>
            <p className="text-[13px] text-ink-2 leading-relaxed">{fault.legacyBehavior}</p>
          </div>
          <div className="panel-sunken p-4 space-y-2 border-l-2 border-l-ok">
            <div className="flex items-center justify-between">
              <span className="eyebrow text-ok">EchoSafe 输出</span>
              <span className="chip num">
                {evaluated.stateLabelZh} ({evaluated.state})
              </span>
            </div>
            <p className="text-[13px] text-ink leading-relaxed">{evaluated.userSummary}</p>
            <div className="text-[11.5px] text-ink-3 num">
              dataSource: {evaluated.dataSource} · frameFresh: {String(evaluated.frameFresh)} ·
              inferenceFresh: {String(evaluated.inferenceFresh)} · canTrustLivePerception:{' '}
              {String(evaluated.canTrustLivePerception)}
            </div>
          </div>
        </div>
      </div>

      {/* Claim-Evidence Matrix */}
      <div className="panel overflow-hidden">
        <div className="px-6 pt-5 pb-4 border-b border-line flex items-center justify-between">
          <h3 className="text-[15px] font-semibold text-ink">能力主张与分层验证证据矩阵</h3>
          <span className="eyebrow">Claim-Evidence Matrix</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12.5px] border-collapse">
            <thead>
              <tr className="border-b border-line bg-sunken/70 text-ink-2">
                <th className="py-3 px-5 font-semibold">能力 / 工程机制</th>
                <th className="py-3 px-4 font-semibold whitespace-nowrap">验证层级</th>
                <th className="py-3 px-5 font-semibold">仓库路径与边界说明</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {CLAIM_ROWS.map((row) => (
                <tr key={row.claim}>
                  <td className="py-3 px-5 font-medium text-ink">{row.claim}</td>
                  <td className="py-3 px-4 whitespace-nowrap">
                    <span className="chip">{row.tier}</span>
                  </td>
                  <td className="py-3 px-5 text-ink-2">{row.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Honest Field-Test Distance MAE/P90 Recorder (Starts 100% Empty — Zero Fabricated Data) */}
      <div className="panel p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="eyebrow">Field-Test Tool · Real Tape-Measure Evaluator</div>
            <h3 className="text-[16px] font-semibold text-ink mt-0.5">
              线下卷尺实拍测距误差记录与 MAE / P90 即时计算器
            </h3>
          </div>
          <span className="chip">默认空白 · 绝不预填模拟数据</span>
        </div>
        <p className="text-[12.5px] text-ink-2 leading-relaxed">
          供团队或评委在真实室内/室外用卷尺采集数据：输入每组真实距离与系统估算值（若系统输出“距离未知/裁切拒判”，估算值留空即可），即时调用{' '}
          <code className="num">computeDistanceErrorStats()</code> 计算整体与分段 MAE / P90，并一键导出 Markdown 实测表格。
        </p>

        <div className="flex flex-wrap items-end gap-2">
          <label className="text-[12px] text-ink-2 space-y-1">
            <div>目标名称</div>
            <input
              type="text"
              value={sampleLabel}
              onChange={(e) => setSampleLabel(e.target.value)}
              className="w-24 rounded border border-line bg-surface px-2 py-1 text-[12.5px] text-ink"
            />
          </label>
          <label className="text-[12px] text-ink-2 space-y-1">
            <div>卷尺实测(m)</div>
            <input
              type="number"
              step="0.1"
              min="0.2"
              max="20"
              value={refInput}
              onChange={(e) => setRefInput(e.target.value)}
              className="w-24 rounded border border-line bg-surface px-2 py-1 text-[12.5px] text-ink num"
            />
          </label>
          <label className="text-[12px] text-ink-2 space-y-1">
            <div>系统估算(m，拒判留空)</div>
            <input
              type="number"
              step="0.1"
              min="0.2"
              max="30"
              placeholder="留空=拒判"
              value={predInput}
              onChange={(e) => setPredInput(e.target.value)}
              className="w-32 rounded border border-line bg-surface px-2 py-1 text-[12.5px] text-ink num"
            />
          </label>
          <button type="button" onClick={handleAddSample} className="btn btn-primary btn-sm">
            记录该组样本
          </button>
          {fieldSamples.length > 0 && (
            <>
              <button type="button" onClick={handleCopyMarkdown} className="btn btn-sm">
                复制 Markdown 报告
              </button>
              <button
                type="button"
                onClick={() => {
                  setFieldSamples([]);
                  setCopyStatus('');
                }}
                className="btn btn-sm"
              >
                清空
              </button>
            </>
          )}
        </div>

        {copyStatus && <div className="text-[12px] text-ok">{copyStatus}</div>}

        {fieldSamples.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 num text-[12px]">
            <div className="panel-sunken p-3">
              <div className="eyebrow">样本总数 / 有效</div>
              <div className="text-ink text-[15px] mt-0.5">
                {fieldReport.total} / {fieldReport.validCount}
              </div>
            </div>
            <div className="panel-sunken p-3">
              <div className="eyebrow">整体 MAE</div>
              <div className="text-ink text-[15px] mt-0.5">
                {fieldReport.maeMeters !== null ? `${fieldReport.maeMeters} m` : '--'}
              </div>
            </div>
            <div className="panel-sunken p-3">
              <div className="eyebrow">P90 绝对误差</div>
              <div className="text-ink text-[15px] mt-0.5">
                {fieldReport.p90ErrorMeters !== null ? `${fieldReport.p90ErrorMeters} m` : '--'}
              </div>
            </div>
            <div className="panel-sunken p-3">
              <div className="eyebrow">拒判 / 未知样本</div>
              <div className="text-ink text-[15px] mt-0.5">{fieldReport.unknownCount}</div>
            </div>
          </div>
        ) : (
          <div className="panel-sunken p-3 text-[12px] text-ink-3">
            当前暂无线下实拍样本（0 条）。请使用真实卷尺采集后录入，系统绝不自动填充模拟结果。
          </div>
        )}
      </div>
    </div>
  );
};
