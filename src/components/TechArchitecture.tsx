import React from 'react';

const STAGES = [
  {
    no: '01',
    title: '多源画面采集',
    desc: '实时摄像头、本地图片、手绘画板与 4 个内置场景。全部在浏览器内处理，不依赖服务器。',
    tag: 'getUserMedia · Canvas',
  },
  {
    no: '02',
    title: 'Sobel 边缘提取',
    desc: 'Rec.709 灰度化后用 3×3 Sobel 卷积求梯度幅值，再结合亮度加权，下采样为 48×16 的声学矩阵。',
    tag: 'Computer Vision',
  },
  {
    no: '03',
    title: '端侧语义识别',
    desc: '可选加载 COCO-SSD（MobileNet v2）在本地 WebGL 推理，识别 80 类常见物体，给出方位与远近并语音播报。',
    tag: 'TensorFlow.js',
  },
  {
    no: '04',
    title: '双耳空间声合成',
    desc: '水平位置映射为 HRTF 方位角，垂直位置映射为音高，边缘强度决定响度，逐列扫描形成连续的“声音图像”。',
    tag: 'Web Audio · HRTF',
  },
];

const FORMULAS = [
  { k: '边缘梯度', f: 'G = √(Gx² + Gy²)，G < T 时置 0' },
  { k: '方位映射', f: 'θ(x) = (2x / W − 1) × 68°' },
  { k: '音高映射', f: 'f(y) = 250 × (2800 / 250)^(1 − y / H) Hz' },
  { k: '回声延迟', f: 'Δt = 50 ms + min(420 ms, d × 55 ms)' },
];

const ROADMAP = [
  {
    phase: 'Phase 1',
    status: '当前',
    title: '浏览器原型',
    desc: '零安装、零成本，用于科普体验与听觉空间感训练。',
  },
  {
    phase: 'Phase 2',
    status: '进行中',
    title: '端侧深度估计',
    desc: '通过 WebGPU / ONNX Runtime Web 运行轻量深度模型，让“远近”也能被听见。',
  },
  {
    phase: 'Phase 3',
    status: '规划',
    title: '可穿戴设备',
    desc: '移植到轻量 AI 眼镜与骨传导耳机，不遮挡环境声，服务真实出行。',
  },
];

const AUDIENCES = [
  {
    group: '视障与弱视人群',
    benefit: '零硬件成本获得环境轮廓、障碍方位与中文物体语音播报，弱视用户可一键开启 WCAG AAA 纯黑高对比大字模式。',
  },
  {
    group: '盲校与康复训练机构',
    benefit: '通过「指尖听画」与「黑暗迷宫」作为感官替代教具，低成本训练低龄视障儿童建立听觉空间地图。',
  },
  {
    group: '普通公众与高校科普',
    benefit: '戴上耳机 10 秒体验全黑回声定位，打破对视障群体的认知隔阂，推动无障碍环境的社会共情。',
  },
  {
    group: '无障碍与 HCI 研究者',
    benefit: '提供可实时调参的 Sobel + HRTF + 五声/对数频谱实验台，方便验证跨模态听觉编码方案。',
  },
];

export const TechArchitecture: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Intro */}
      <div className="panel p-6 sm:p-8">
        <div className="grid gap-6 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <div className="eyebrow">05 · How it works</div>
            <h2 className="title-serif text-[24px] leading-snug mt-2">
              从一帧画面，
              <br />
              到一段可以“听见”的空间。
            </h2>
          </div>
          <div className="lg:col-span-7 text-[14px] leading-[1.85] text-ink-2 space-y-3">
            <p>
              中国有超过 <span className="num text-ink">1,700 万</span>{' '}
              视障人士。现有辅助工具多数停留在“识别物体 → 语音播报名称”，告诉用户“有什么”，却很难告诉用户“在哪里、是什么形状”。
            </p>
            <p>
              EchoSight 借鉴神经科学中的感官替代（Sensory Substitution）研究，把连续的视觉轮廓直接编码为双耳声音，
              让大脑通过训练逐渐学会“用耳朵读图”。
            </p>
          </div>
        </div>
      </div>

      {/* Pipeline */}
      <div className="panel overflow-hidden">
        <div className="px-6 pt-5 pb-4 border-b border-line flex items-center justify-between">
          <h3 className="text-[14px] font-semibold">处理流水线</h3>
          <span className="eyebrow">Pipeline</span>
        </div>
        <ol className="grid sm:grid-cols-2 lg:grid-cols-4">
          {STAGES.map((st, i) => (
            <li
              key={st.no}
              className={`p-6 space-y-3 border-line ${i > 0 ? 'lg:border-l' : ''} ${
                i % 2 === 1 ? 'sm:border-l' : ''
              } ${i >= 2 ? 'sm:border-t lg:border-t-0' : ''} ${i > 0 ? 'border-t sm:border-t-0' : ''}`}
            >
              <div className="flex items-center justify-between">
                <span className="num text-[28px] leading-none text-ink">{st.no}</span>
                {i < STAGES.length - 1 && (
                  <span className="hidden lg:block num text-ink-3" aria-hidden="true">
                    →
                  </span>
                )}
              </div>
              <div className="text-[15px] font-semibold text-ink">{st.title}</div>
              <p className="text-[13px] leading-[1.75] text-ink-2">{st.desc}</p>
              <div className="eyebrow pt-1">{st.tag}</div>
            </li>
          ))}
        </ol>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Formulas */}
        <div className="panel p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-[14px] font-semibold">核心映射公式</h3>
            <span className="eyebrow">Mapping</span>
          </div>
          <dl className="mt-4 divide-y divide-line border-y border-line">
            {FORMULAS.map((m) => (
              <div key={m.k} className="grid grid-cols-[84px_1fr] gap-3 py-3">
                <dt className="text-[12.5px] text-ink-3">{m.k}</dt>
                <dd className="num text-[13px] text-ink">{m.f}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-4 text-[12.5px] leading-relaxed text-ink-3">
            高处 = 高音、左侧 = 左耳，这一对应与人类天然的跨模态直觉一致，降低学习门槛。
          </p>
        </div>

        {/* Roadmap */}
        <div className="panel p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-[14px] font-semibold">演进路线</h3>
            <span className="eyebrow">Roadmap</span>
          </div>
          <ol className="mt-4 relative">
            {ROADMAP.map((r, i) => (
              <li key={r.phase} className="relative pl-7 pb-5 last:pb-0">
                {i < ROADMAP.length - 1 && (
                  <span className="absolute left-[5px] top-4 bottom-0 w-px bg-line-strong" aria-hidden="true" />
                )}
                <span
                  className={`absolute left-0 top-1.5 h-[11px] w-[11px] rounded-full border-2 ${
                    i === 0 ? 'border-signal bg-signal' : 'border-ink-3 bg-surface'
                  }`}
                  aria-hidden="true"
                />
                <div className="flex items-center gap-2">
                  <span className="num text-[12px] text-ink-3">{r.phase}</span>
                  <span
                    className={`text-[11px] px-1.5 py-0.5 rounded ${
                      i === 0 ? 'bg-signal-soft text-signal-ink' : 'bg-sunken text-ink-3'
                    }`}
                  >
                    {r.status}
                  </span>
                </div>
                <div className="mt-1 text-[14px] font-semibold text-ink">{r.title}</div>
                <p className="mt-0.5 text-[13px] leading-relaxed text-ink-2">{r.desc}</p>
              </li>
            ))}
          </ol>
        </div>
      </div>

      {/* Target Audiences & Social Impact */}
      <div className="panel overflow-hidden">
        <div className="px-6 pt-5 pb-4 border-b border-line flex items-center justify-between">
          <h3 className="text-[14px] font-semibold">适用人群与社会价值</h3>
          <span className="eyebrow">Who Benefits</span>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4">
          {AUDIENCES.map((a, i) => (
            <div
              key={a.group}
              className={`p-5 space-y-2 border-line ${i > 0 ? 'lg:border-l' : ''} ${
                i % 2 === 1 ? 'sm:border-l' : ''
              } ${i >= 2 ? 'sm:border-t lg:border-t-0' : ''} ${i > 0 ? 'border-t sm:border-t-0' : ''}`}
            >
              <div className="text-[14px] font-semibold text-ink">{a.group}</div>
              <p className="text-[12.5px] leading-[1.7] text-ink-2">{a.benefit}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
