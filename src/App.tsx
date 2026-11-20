import React, { useEffect, useRef, useState } from 'react';
import { Contrast, Headphones, Play, Volume2 } from 'lucide-react';
import { VisionScanner } from './components/VisionScanner';
import { BlindSceneTheater } from './components/BlindSceneTheater';
import { AcousticCanvas } from './components/AcousticCanvas';
import { DarkMaze } from './components/DarkMaze';
import { TechArchitecture } from './components/TechArchitecture';
import { spatialAudio } from './engine/spatialAudio';

type TabId = 'scanner' | 'theater' | 'canvas' | 'maze' | 'arch';

const TABS: { id: TabId; no: string; label: string; sub: string }[] = [
  { id: 'scanner', no: '01', label: '声画扫描', sub: '摄像头 / 图片 → 立体声场' },
  { id: 'theater', no: '02', label: '场景剧场', sub: '盲人戴耳机动态实景模拟' },
  { id: 'canvas', no: '03', label: '指尖听画', sub: '画一笔，听一笔' },
  { id: 'maze', no: '04', label: '黑暗迷宫', sub: '只用耳朵找到出口' },
  { id: 'arch', no: '05', label: '原理与硬件', sub: '算法、穿戴方案与路线' },
];

/** Small hand-drawn style waveform used as a quiet brand mark. */
const WaveMark: React.FC<{ className?: string }> = ({ className }) => (
  <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
    <rect x="0.5" y="0.5" width="31" height="31" rx="8" fill="#1b1a17" />
    <path
      d="M6 16 h3 l2 -5 l3 11 l3 -14 l3 12 l2 -6 l2 2 h2"
      fill="none"
      stroke="#f3ede2"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="26" cy="16" r="1.8" fill="#e4572e" />
  </svg>
);

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>('scanner');
  const [calibrating, setCalibrating] = useState(false);
  const [demoProgress, setDemoProgress] = useState<number | null>(null);
  const stopDemoRef = useRef<(() => void) | null>(null);
  const [highContrast, setHighContrast] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem('echosight:hc') === '1';
  });

  useEffect(() => {
    return () => {
      stopDemoRef.current?.();
    };
  }, []);

  const handlePlayDiagonalDemo = () => {
    stopDemoRef.current?.();
    setDemoProgress(0);
    stopDemoRef.current = spatialAudio.playDiagonalDemo(
      (p) => setDemoProgress(p),
      () => setDemoProgress(null),
      0.74
    );
  };

  useEffect(() => {
    const root = document.documentElement;
    if (highContrast) {
      root.setAttribute('data-high-contrast', 'true');
      window.localStorage.setItem('echosight:hc', '1');
    } else {
      root.removeAttribute('data-high-contrast');
      window.localStorage.removeItem('echosight:hc');
    }
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) {
      meta.setAttribute('content', highContrast ? '#000000' : '#f2efe8');
    }
  }, [highContrast]);

  // Number keys 1-4 switch modules when not focused on inputs
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.metaKey ||
        e.ctrlKey ||
        e.altKey
      ) {
        return;
      }
      if (e.key === '1') setActiveTab('scanner');
      else if (e.key === '2') setActiveTab('theater');
      else if (e.key === '3') setActiveTab('canvas');
      else if (e.key === '4') setActiveTab('maze');
      else if (e.key === '5') setActiveTab('arch');
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const handleCalibrateHeadphones = () => {
    spatialAudio.playStereoCalibration(0.75);
    setCalibrating(true);
    window.setTimeout(() => setCalibrating(false), 1300);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-line bg-paper">
        <div className="mx-auto max-w-6xl px-5 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <WaveMark className="h-8 w-8 shrink-0" />
            <div className="flex items-baseline gap-2.5 min-w-0">
              <span className="title-serif text-[19px] leading-none">听见光影</span>
              <span className="num text-[12px] text-ink-3 tracking-wide hidden sm:inline">EchoSight</span>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <span className="hidden lg:inline-flex items-center gap-1.5 text-[12.5px] text-ink-2 mr-1">
              <Headphones className="h-4 w-4" />
              建议佩戴耳机
            </span>
            <button
              type="button"
              aria-pressed={highContrast}
              onClick={() => setHighContrast((v) => !v)}
              className={`btn btn-sm ${highContrast ? 'btn-active' : ''}`}
              title="切换面向弱视用户的 WCAG AAA 纯黑高对比大字模式"
            >
              <Contrast className="h-3.5 w-3.5" />
              {highContrast ? '标准纸面' : '弱视高对比'}
            </button>
            <button
              type="button"
              onClick={handleCalibrateHeadphones}
              className={`btn btn-sm ${calibrating ? 'btn-active' : ''}`}
              title="依次在左、中、右播放提示音，确认耳机左右声道"
            >
              <span className={`led ${calibrating ? 'led-on' : ''}`} />
              <Volume2 className="h-3.5 w-3.5" />
              {calibrating ? '左 · 中 · 右' : '耳机校准'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-5">
        {/* Hero (shown on primary scanner tab to keep other workspaces single-screen) */}
        {activeTab === 'scanner' && (
          <section className="pt-7 pb-6 border-b border-line space-y-5">
            <div className="grid gap-6 lg:grid-cols-12">
              <div className="lg:col-span-8">
                <div className="eyebrow">Sensory Substitution · 帮视障者“用耳朵看世界”</div>
                <h1 className="title-serif mt-2.5 text-[26px] sm:text-[34px] leading-[1.24]">
                  如果世界没有光，声音也能勾勒出万物的轮廓。
                </h1>
                <p className="mt-3 max-w-2xl text-[14px] leading-[1.75] text-ink-2">
                  EchoSight 把摄像头画面实时翻译成 3D 立体声，帮助视障朋友感知前方障碍物的方向与形状，也让普通人通过“黑暗迷宫”亲身体会用耳朵辨位。
                </p>
              </div>

              <dl className="lg:col-span-4 self-end grid grid-cols-2 border-t border-l border-line">
                {[
                  { k: '方位范围', v: '±68°', d: 'HRTF 双耳定位' },
                  { k: '声学矩阵', v: '48×16', d: 'Sobel 边缘采样' },
                  { k: '实时处理', v: '60 fps', d: '纯前端计算' },
                  { k: '数据隐私', v: '本地', d: '画面不出设备' },
                ].map((s) => (
                  <div key={s.k} className="border-r border-b border-line px-3.5 py-2.5">
                    <dt className="eyebrow">{s.k}</dt>
                    <dd className="num mt-1 text-[20px] text-ink leading-none">{s.v}</dd>
                    <dd className="mt-1 text-[11.5px] text-ink-3">{s.d}</dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* 3-Second Instant Understanding Bar */}
            <div className="panel-sunken px-4 py-3 flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[12.5px] text-ink-2">
                <span className="eyebrow text-ink">3 秒听懂规则</span>
                <span>
                  <strong className="text-ink">左 / 右位置</strong> → 左 / 右耳发声
                </span>
                <span>
                  <strong className="text-ink">高 / 低位置</strong> → 音调高 / 低
                </span>
                <span>
                  <strong className="text-ink">轮廓越亮</strong> → 声音越响
                </span>
              </div>

              <div className="flex items-center gap-3">
                {/* Mini interactive diagonal oscilloscope */}
                <div className="screen w-[108px] h-[34px] shrink-0" aria-hidden="true">
                  <svg viewBox="0 0 108 34" className="w-full h-full block">
                    <line
                      x1="10"
                      y1="28"
                      x2="98"
                      y2="6"
                      stroke="rgba(243, 237, 226, 0.45)"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                    />
                    {demoProgress !== null && (
                      <circle
                        cx={10 + demoProgress * 88}
                        cy={28 - demoProgress * 22}
                        r="3.8"
                        fill="#ff8a3d"
                      />
                    )}
                  </svg>
                </div>
                <button
                  type="button"
                  onClick={handlePlayDiagonalDemo}
                  className={`btn btn-sm ${demoProgress !== null ? 'btn-signal' : 'btn-primary'}`}
                >
                  <Play className="h-3.5 w-3.5" />
                  {demoProgress !== null ? '正在演示：左下 → 右上' : '3 秒试听演示'}
                </button>
              </div>
            </div>
          </section>
        )}

        {/* Module Navigation */}
        <nav
          role="tablist"
          aria-label="EchoSight 功能模块"
          className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 border-b border-line"
        >
          {TABS.map((t) => {
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => setActiveTab(t.id)}
                className={`group relative text-left px-1 sm:px-2 py-4 transition-colors ${
                  active ? 'text-ink' : 'text-ink-3 hover:text-ink-2'
                }`}
              >
                <div className="flex items-baseline gap-2.5">
                  <span className={`num text-[12px] ${active ? 'text-signal-ink' : ''}`}>{t.no}</span>
                  <span className="text-[15px] font-semibold">{t.label}</span>
                  <span className="kbd ml-auto mr-2 hidden lg:inline-flex opacity-0 group-hover:opacity-100 transition-opacity">
                    {Number(t.no)}
                  </span>
                </div>
                <div className="mt-1 pl-[30px] text-[12px] text-ink-3">{t.sub}</div>
                <span
                  className={`absolute left-0 right-3 -bottom-px h-[2px] transition-colors ${
                    active ? 'bg-signal' : 'bg-transparent'
                  }`}
                />
              </button>
            );
          })}
        </nav>

        {/* Active Module */}
        <section className="py-8" role="tabpanel">
          {activeTab === 'scanner' && <VisionScanner />}
          {activeTab === 'theater' && <BlindSceneTheater />}
          {activeTab === 'canvas' && <AcousticCanvas />}
          {activeTab === 'maze' && <DarkMaze />}
          {activeTab === 'arch' && <TechArchitecture />}
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-line">
        <div className="mx-auto max-w-6xl px-5 py-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-[12.5px] text-ink-3">
          <div>
            <span className="title-serif text-[14px] text-ink mr-2">听见光影</span>
            2026 Qoder 高校创造力大赛 · Hack4Good 技术向善赛道
          </div>
          <div>全部计算在浏览器本地完成，不上传任何画面</div>
        </div>
      </footer>
    </div>
  );
}

export default App;
