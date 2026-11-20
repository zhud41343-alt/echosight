import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, Trash2, Undo2, Download, Volume2 } from 'lucide-react';
import { ScaleMode, spatialAudio } from '../engine/spatialAudio';
import { processFrameToAcousticMatrix } from '../engine/visionProcessor';

const W = 640;
const H = 340;
const COLS = 48;
const ROWS = 16;
const MAX_HISTORY = 15;

export const AcousticCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDrawingRef = useRef(false);
  const lastSoundTimeRef = useRef(0);
  const historyRef = useRef<ImageData[]>([]);

  const [isPlaying, setIsPlaying] = useState(false);
  const [scanCol, setScanCol] = useState(-1);
  const [scaleMode, setScaleMode] = useState<ScaleMode>('pentatonic');
  const [brushSize, setBrushSize] = useState(7);
  const [speedMs, setSpeedMs] = useState(60);
  const [canUndo, setCanUndo] = useState(false);
  const [cursorHud, setCursorHud] = useState<{
    azimuthDeg: number;
    freqHz: number;
    noteName: string;
  } | null>(null);

  const pushHistory = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx) return;
    const snap = ctx.getImageData(0, 0, W, H);
    historyRef.current.push(snap);
    if (historyRef.current.length > MAX_HISTORY) {
      historyRef.current.shift();
    }
    setCanUndo(historyRef.current.length > 0);
  };

  const handleUndo = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx || historyRef.current.length === 0) return;
    setIsPlaying(false);
    const prev = historyRef.current.pop();
    if (prev) {
      ctx.putImageData(prev, 0, 0);
    }
    setCanUndo(historyRef.current.length > 0);
  };

  const clearCanvas = (saveSnap = true) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    if (saveSnap) pushHistory();
    ctx.fillStyle = '#161512';
    ctx.fillRect(0, 0, W, H);
    setIsPlaying(false);
    setScanCol(-1);
  };

  const drawTemplate = (type: 'wave' | 'star' | 'stairs', saveSnap = true) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    if (saveSnap) pushHistory();
    ctx.fillStyle = '#161512';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#f3ede2';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    if (type === 'wave') {
      for (let x = 40; x <= W - 40; x += 6) {
        const y = H / 2 + Math.sin((x / W) * Math.PI * 3.5) * 90;
        if (x === 40) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    } else if (type === 'stairs') {
      const steps = 6;
      const sw = (W - 100) / steps;
      const sh = (H - 100) / steps;
      for (let i = 0; i < steps; i++) {
        const x = 50 + i * sw;
        const y = H - 55 - i * sh;
        ctx.moveTo(x, y);
        ctx.lineTo(x + sw * 0.75, y);
      }
    } else {
      const cx = W / 2;
      const cy = H / 2;
      const spikes = 5;
      const outer = 115;
      const inner = 52;
      for (let i = 0; i <= spikes * 2; i++) {
        const r = i % 2 === 0 ? outer : inner;
        const a = (i * Math.PI) / spikes - Math.PI / 2;
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  };

  useEffect(() => {
    clearCanvas(false);
    drawTemplate('wave', false);
  }, []);

  // Playback loop
  useEffect(() => {
    if (!isPlaying) {
      setScanCol(-1);
      return;
    }
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx) return;

    const imgData = ctx.getImageData(0, 0, W, H);
    const { matrix } = processFrameToAcousticMatrix(imgData, COLS, ROWS, 20, 0.35);

    let current = 0;
    const timer = window.setInterval(() => {
      setScanCol(current);
      spatialAudio.playScanColumn(matrix[current], current / (COLS - 1), speedMs / 1000, {
        scaleMode,
        useHRTF: true,
        volume: 0.72,
      });
      current = (current + 1) % COLS;
    }, speedMs);

    return () => window.clearInterval(timer);
  }, [isPlaying, scaleMode, speedMs]);

  const getCoords = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(W, ((e.clientX - rect.left) / rect.width) * W)),
      y: Math.max(0, Math.min(H, ((e.clientY - rect.top) / rect.height) * H)),
    };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    spatialAudio.ensureContext();
    setIsPlaying(false);
    pushHistory();
    isDrawingRef.current = true;
    const { x, y } = getCoords(e);
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.beginPath();
    ctx.moveTo(x, y);
    const info = spatialAudio.playBrushTone(x / W, y / H, {
      scaleMode,
      useHRTF: true,
      volume: 0.7,
    });
    setCursorHud(info);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return;
    const { x, y } = getCoords(e);
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.strokeStyle = '#ff8a3d';
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineTo(x, y);
    ctx.stroke();

    const now = performance.now();
    if (now - lastSoundTimeRef.current > 50) {
      lastSoundTimeRef.current = now;
      const info = spatialAudio.playBrushTone(x / W, y / H, {
        scaleMode,
        useHRTF: true,
        volume: 0.65,
      });
      setCursorHud(info);
    }
  };

  const handlePointerUp = () => {
    isDrawingRef.current = false;
  };

  const handleExportPng = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `echosight-acoustic-sketch-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="panel p-5 sm:p-6 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="eyebrow">03 · Sketch</div>
          <h2 className="title-serif text-[20px] mt-1">指尖听画板</h2>
          <p className="text-[13px] text-ink-2 mt-1">
            上方音高清脆、下方低沉；从左画到右，声像会在双耳之间平滑移动。
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => {
              spatialAudio.ensureContext();
              setIsPlaying((p) => !p);
            }}
            className={`btn ${isPlaying ? 'btn-signal' : 'btn-primary'}`}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isPlaying ? '停止回放' : '扫描回放整幅画'}
          </button>
          <button
            type="button"
            disabled={!canUndo}
            onClick={handleUndo}
            className="btn"
          >
            <Undo2 className="h-4 w-4" />
            撤销
          </button>
          <button
            type="button"
            onClick={() => clearCanvas(true)}
            className="btn"
          >
            <Trash2 className="h-4 w-4" />
            清空
          </button>
          <button
            type="button"
            onClick={handleExportPng}
            className="btn"
            title="导出当前声学素描图片"
          >
            <Download className="h-4 w-4" />
            导出 PNG
          </button>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="panel-sunken px-4 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="eyebrow">音阶映射</span>
          <div className="seg">
            {[
              { id: 'pentatonic' as ScaleMode, label: '五声音阶' },
              { id: 'chromatic' as ScaleMode, label: '半音阶' },
              { id: 'voice_log' as ScaleMode, label: 'vOICe 对数谱' },
            ].map((m) => (
              <button
                key={m.id}
                type="button"
                data-active={scaleMode === m.id}
                onClick={() => setScaleMode(m.id)}
                className="seg-item"
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-5 text-[12.5px] text-ink-2">
          <label className="flex items-center gap-2.5">
            <span>笔触</span>
            <input
              type="range"
              min={3}
              max={16}
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="slider w-24"
            />
            <span className="num text-ink w-8 text-right">{brushSize}px</span>
          </label>
          <label className="flex items-center gap-2.5">
            <span>步进</span>
            <input
              type="range"
              min={30}
              max={110}
              value={speedMs}
              onChange={(e) => setSpeedMs(Number(e.target.value))}
              className="slider w-24"
            />
            <span className="num text-ink w-10 text-right">{speedMs}ms</span>
          </label>
        </div>
      </div>

      {/* Canvas Container */}
      <div className="screen">
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
          className="w-full h-auto block touch-none cursor-crosshair"
        />
        {scanCol >= 0 && (
          <div
            className="pointer-events-none absolute top-0 bottom-0 w-[2px] bg-amber"
            style={{ left: `${(scanCol / COLS) * 100}%` }}
          />
        )}
        <div className="pointer-events-none absolute bottom-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2">
          <div className="screen-label inline-flex items-center gap-1.5">
            <Volume2 className="h-3.5 w-3.5 text-amber" />
            拖动鼠标或手指绘制，实时听见 3D 音高
          </div>
          {cursorHud && (
            <div className="screen-label">
              {cursorHud.azimuthDeg > 0 ? `+${cursorHud.azimuthDeg}` : cursorHud.azimuthDeg}° ·{' '}
              {cursorHud.freqHz} Hz ({cursorHud.noteName})
            </div>
          )}
        </div>
      </div>

      {/* Quick Templates */}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <span className="eyebrow mr-1">示例图形</span>
        {[
          { id: 'wave' as const, label: '正弦曲线' },
          { id: 'stairs' as const, label: '阶梯音阶' },
          { id: 'star' as const, label: '五角星' },
        ].map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => {
              setIsPlaying(false);
              drawTemplate(t.id, true);
            }}
            className="btn btn-sm"
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
};
