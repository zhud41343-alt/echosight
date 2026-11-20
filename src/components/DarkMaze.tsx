import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Radio,
  Eye,
  EyeOff,
  RotateCcw,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Volume2,
} from 'lucide-react';
import { spatialAudio } from '../engine/spatialAudio';

// 0 = empty, 1 = wall, 2 = exit
const MAZE_LEVELS: { name: string; desc: string; grid: number[][]; exit: { x: number; y: number } }[] = [
  {
    name: '第 1 关 · 初探回声廊',
    desc: '熟悉 3D 回声与出口信标',
    exit: { x: 9, y: 7 },
    grid: [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
      [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
      [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
      [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
      [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
      [1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
  },
  {
    name: '第 2 关 · 十字回声厅',
    desc: '多岔路口听声辨别通路',
    exit: { x: 9, y: 1 },
    grid: [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1],
      [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
      [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
      [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
      [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
      [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
  },
  {
    name: '第 3 关 · 环绕暗室',
    desc: '纯听觉空间拓扑导航',
    exit: { x: 5, y: 4 },
    grid: [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
      [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
      [1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1],
      [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
      [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
  },
];

const ROWS = 9;
const COLS = 11;
const CELL = 44;
const CANVAS_W = COLS * CELL;
const CANVAS_H = ROWS * CELL;

interface EchoDot {
  x: number;
  y: number;
  type: 'wall' | 'exit';
  createdAt: number;
}

interface Ripple {
  x: number;
  y: number;
  createdAt: number;
}

export const DarkMaze: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [levelIdx, setLevelIdx] = useState(0);
  const [player, setPlayer] = useState({ x: 1, y: 1 });
  const playerRef = useRef({ x: 1, y: 1 });
  const [blindfold, setBlindfold] = useState(true);
  const [steps, setSteps] = useState(0);
  const [pings, setPings] = useState(0);
  const [won, setWon] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [started, setStarted] = useState(false);
  const [bestSteps, setBestSteps] = useState<Record<number, number>>({});
  const [liveStatus, setLiveStatus] = useState('黑暗迷宫已就绪，按空格键发射 3D 声呐或使用方向键移动。');

  const echoesRef = useRef<EchoDot[]>([]);
  const ripplesRef = useRef<Ripple[]>([]);

  const level = MAZE_LEVELS[levelIdx];
  const mazeGrid = level.grid;
  const exitPos = level.exit;

  // Elapsed timer while actively playing
  useEffect(() => {
    if (!started || won) return;
    const id = window.setInterval(() => {
      setElapsedSec((s) => s + 1);
    }, 1000);
    return () => window.clearInterval(id);
  }, [started, won]);

  const speakGuidance = useCallback(
    (px: number, py: number) => {
      const dx = exitPos.x - px;
      const dy = exitPos.y - py;
      const horiz = dx > 0 ? '偏右' : dx < 0 ? '偏左' : '正中';
      const vert = dy > 0 ? '下方' : dy < 0 ? '上方' : '同排';
      const openDirs: string[] = [];
      if (py > 0 && mazeGrid[py - 1][px] !== 1) openDirs.push('上');
      if (py < ROWS - 1 && mazeGrid[py + 1][px] !== 1) openDirs.push('下');
      if (px > 0 && mazeGrid[py][px - 1] !== 1) openDirs.push('左');
      if (px < COLS - 1 && mazeGrid[py][px + 1] !== 1) openDirs.push('右');
      const msg = `出口在${horiz}${vert}，当前可通行方向：${openDirs.join('、') || '无'}`;
      setLiveStatus(msg);
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(msg);
          u.lang = 'zh-CN';
          u.rate = 1.05;
          window.speechSynthesis.speak(u);
        } catch {
          // Ignore speech errors
        }
      }
    },
    [exitPos.x, exitPos.y, mazeGrid]
  );

  const emitSonar = useCallback(
    (px: number, py: number) => {
      spatialAudio.ensureContext();
      setStarted(true);
      const now = performance.now();
      ripplesRef.current.push({ x: (px + 0.5) * CELL, y: (py + 0.5) * CELL, createdAt: now });

      // Cast 8 omnidirectional acoustic rays
      const rayAngles = [
        -Math.PI * 0.75,
        -Math.PI * 0.5,
        -Math.PI * 0.25,
        0,
        Math.PI * 0.25,
        Math.PI * 0.5,
        Math.PI * 0.75,
        Math.PI,
      ];

      const audioRays: { angleOffsetRad: number; distance: number; hitType: 'wall' | 'exit' }[] = [];

      for (const angle of rayAngles) {
        const dx = Math.sin(angle);
        const dy = -Math.cos(angle);
        let dist = 0.2;
        let hitType: 'wall' | 'exit' = 'wall';
        let hitX = px + 0.5;
        let hitY = py + 0.5;

        while (dist < 8) {
          const rx = px + 0.5 + dx * dist;
          const ry = py + 0.5 + dy * dist;
          const gx = Math.floor(rx);
          const gy = Math.floor(ry);
          if (gx < 0 || gy < 0 || gx >= COLS || gy >= ROWS) break;
          const cell = mazeGrid[gy][gx];
          if (cell === 1 || cell === 2) {
            hitType = cell === 2 ? 'exit' : 'wall';
            hitX = rx;
            hitY = ry;
            break;
          }
          dist += 0.12;
        }

        echoesRef.current.push({
          x: hitX * CELL,
          y: hitY * CELL,
          type: hitType,
          createdAt: now,
        });

        audioRays.push({ angleOffsetRad: angle, distance: dist, hitType });
      }

      const relDx = exitPos.x - px;
      const relDy = exitPos.y - py;
      const exitDist = Math.hypot(relDx, relDy);

      spatialAudio.playEcholocationPulse(audioRays, { dx: relDx, dy: relDy, dist: exitDist }, 0.78);
      setPings((p) => p + 1);
    },
    [exitPos.x, exitPos.y, mazeGrid]
  );

  const move = useCallback(
    (dx: number, dy: number) => {
      if (won) return;
      spatialAudio.ensureContext();
      setStarted(true);

      const prev = playerRef.current;
      const nx = prev.x + dx;
      const ny = prev.y + dy;
      if (nx < 0 || ny < 0 || nx >= COLS || ny >= ROWS || mazeGrid[ny][nx] === 1) {
        spatialAudio.playCollisionThud(0.75);
        echoesRef.current.push({
          x: (prev.x + 0.5 + dx * 0.5) * CELL,
          y: (prev.y + 0.5 + dy * 0.5) * CELL,
          type: 'wall',
          createdAt: performance.now(),
        });
        setLiveStatus('前方碰壁，请尝试其他方向。');
        return;
      }

      playerRef.current = { x: nx, y: ny };
      setPlayer({ x: nx, y: ny });
      const nextSteps = steps + 1;
      setSteps(nextSteps);

      if (mazeGrid[ny][nx] === 2) {
        setWon(true);
        spatialAudio.playVictoryChord(0.8);
        setBestSteps((prevBest) => {
          const existing = prevBest[levelIdx];
          if (existing === undefined || nextSteps < existing) {
            return { ...prevBest, [levelIdx]: nextSteps };
          }
          return prevBest;
        });
        setLiveStatus(`恭喜通关${level.name}！共移动 ${nextSteps} 步。`);
      } else {
        emitSonar(nx, ny);
      }
    },
    [won, emitSonar, mazeGrid, steps, levelIdx, level.name]
  );

  // Keyboard controls
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
        e.preventDefault();
      }
      if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') move(0, -1);
      else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') move(0, 1);
      else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') move(-1, 0);
      else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') move(1, 0);
      else if (e.key === ' ' && !won) emitSonar(playerRef.current.x, playerRef.current.y);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [move, emitSonar, won]);

  // Animation loop for maze rendering + fading sonar echoes
  useEffect(() => {
    let rafId = 0;
    const render = (now: number) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (canvas && ctx) {
        ctx.fillStyle = '#161512';
        ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

        // Draw maze structure if reveal mode or won
        if (!blindfold || won) {
          for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
              if (mazeGrid[r][c] === 1) {
                ctx.fillStyle = '#2a2823';
                ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
                ctx.strokeStyle = 'rgba(243, 237, 226, 0.1)';
                ctx.lineWidth = 1;
                ctx.strokeRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
              } else if (mazeGrid[r][c] === 2) {
                ctx.fillStyle = 'rgba(159, 201, 176, 0.25)';
                ctx.fillRect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4);
                ctx.fillStyle = '#9fc9b0';
                ctx.font = '600 12px "JetBrains Mono", Consolas, monospace';
                ctx.textAlign = 'center';
                ctx.fillText('出口', (c + 0.5) * CELL, (r + 0.62) * CELL);
              }
            }
          }
        }

        // Expanding sonar ripples
        ripplesRef.current = ripplesRef.current.filter((rp) => now - rp.createdAt < 1100);
        ripplesRef.current.forEach((rp) => {
          const progress = Math.max(0, (now - rp.createdAt) / 1100);
          const radius = Math.max(0.5, progress * 220);
          ctx.strokeStyle = `rgba(255, 138, 61, ${Math.max(0, 0.65 * (1 - progress))})`;
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.arc(rp.x, rp.y, radius, 0, Math.PI * 2);
          ctx.stroke();
        });

        // Fading sonar echo reflection dots (visible even in blindfold mode)
        echoesRef.current = echoesRef.current.filter((dot) => now - dot.createdAt < 2600);
        echoesRef.current.forEach((dot) => {
          const alpha = Math.max(0, 1 - (now - dot.createdAt) / 2600);
          ctx.fillStyle =
            dot.type === 'exit'
              ? `rgba(159, 201, 176, ${alpha})`
              : `rgba(243, 237, 226, ${alpha * 0.9})`;
          ctx.beginPath();
          ctx.arc(dot.x, dot.y, dot.type === 'exit' ? 6.5 : 4, 0, Math.PI * 2);
          ctx.fill();
        });

        // Subtle sonar distance rings around player in blindfold mode
        const cx = (player.x + 0.5) * CELL;
        const cy = (player.y + 0.5) * CELL;
        if (blindfold && !won) {
          [CELL * 1.4, CELL * 2.8, CELL * 4.2].forEach((r) => {
            ctx.strokeStyle = 'rgba(243, 237, 226, 0.06)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();
          });
        }

        // Draw Player Beacon (phosphor core + amber ring, no glow)
        ctx.fillStyle = '#f3ede2';
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ff8a3d';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 11, 0, Math.PI * 2);
        ctx.stroke();
      }
      rafId = requestAnimationFrame(render);
    };
    rafId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(rafId);
  }, [blindfold, player, won, mazeGrid]);

  const selectLevel = (idx: number) => {
    setLevelIdx(idx);
    playerRef.current = { x: 1, y: 1 };
    setPlayer({ x: 1, y: 1 });
    setSteps(0);
    setPings(0);
    setElapsedSec(0);
    setStarted(false);
    setWon(false);
    echoesRef.current = [];
    ripplesRef.current = [];
    setLiveStatus(`已切换至${MAZE_LEVELS[idx].name}。`);
  };

  const resetMaze = () => selectLevel(levelIdx);

  return (
    <div className="grid gap-6 lg:grid-cols-12 items-start">
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {liveStatus}
      </div>

      {/* Left: Tactical Sonar Maze Canvas */}
      <div className="lg:col-span-7 panel p-5 space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="eyebrow">04 · Echolocation</div>
            <h2 className="title-serif text-[20px] mt-1">黑暗迷宫 · 回声定位</h2>
            <p className="text-[13px] text-ink-2 mt-1">
              在全黑模式下，仅凭墙壁回声与出口和弦信标找到通路。
            </p>
          </div>

          <button
            type="button"
            onClick={() => setBlindfold((b) => !b)}
            className={`btn btn-sm shrink-0 ${!blindfold ? 'btn-active' : ''}`}
          >
            {blindfold ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
            {blindfold ? '全黑模式' : '透视模式'}
          </button>
        </div>

        {/* 3-Level Selector */}
        <div className="grid grid-cols-3 gap-2">
          {MAZE_LEVELS.map((lv, idx) => {
            const best = bestSteps[idx];
            return (
              <button
                key={lv.name}
                type="button"
                data-active={idx === levelIdx}
                onClick={() => selectLevel(idx)}
                className="tile"
              >
                <div className="flex items-center justify-between gap-1">
                  <span className="text-[12.5px] font-semibold text-ink truncate">{lv.name}</span>
                  {best !== undefined && (
                    <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-ok" />
                  )}
                </div>
                <div className="text-[11px] text-ink-3 mt-0.5 truncate">
                  {best !== undefined ? `已通关 · 最佳 ${best} 步` : lv.desc}
                </div>
              </button>
            );
          })}
        </div>

        <div className="screen flex justify-center p-2">
          <canvas
            ref={canvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            className="w-full max-w-[484px] h-auto block"
          />

          {blindfold && pings === 0 && !won && (
            <div className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 screen-label">
              按 Space 或点击右侧「发射 3D 声呐」探测四周墙壁
            </div>
          )}

          {won && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-screen/92 p-6 text-center">
              <div className="eyebrow text-amber">Cleared</div>
              <h3 className="title-serif text-[24px] text-phosphor mt-2">
                听声走出黑暗迷宫
              </h3>
              <p className="num text-[13px] text-phosphor/80 mt-2">
                {steps} 步 · {pings} 次声呐 · {elapsedSec} 秒
              </p>
              <div className="mt-5 flex flex-wrap items-center justify-center gap-2.5">
                <button type="button" onClick={resetMaze} className="btn">
                  <RotateCcw className="h-4 w-4" />
                  重玩本关
                </button>
                {levelIdx < MAZE_LEVELS.length - 1 && (
                  <button
                    type="button"
                    onClick={() => selectLevel(levelIdx + 1)}
                    className="btn btn-signal"
                  >
                    进入下一关
                    <ChevronRight className="h-4 w-4" />
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right: Controls & Guide */}
      <div className="lg:col-span-5 space-y-4">
        <div className="panel p-5 space-y-4">
          <div className="grid grid-cols-3 gap-2.5">
            <div className="panel-sunken p-3 text-center">
              <div className="eyebrow">步数</div>
              <div className="num text-xl text-ink mt-1">{steps}</div>
            </div>
            <div className="panel-sunken p-3 text-center">
              <div className="eyebrow">声呐</div>
              <div className="num text-xl text-ink mt-1">{pings}</div>
            </div>
            <div className="panel-sunken p-3 text-center">
              <div className="eyebrow">用时</div>
              <div className="num text-xl text-ink mt-1">{elapsedSec}s</div>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => !won && emitSonar(player.x, player.y)}
              className="btn btn-lg btn-primary flex-1"
            >
              <Radio className="h-4 w-4" />
              发射 3D 声呐
            </button>
            <button
              type="button"
              onClick={() => speakGuidance(player.x, player.y)}
              title="无障碍语音播报当前出口方向与可通行走廊"
              className="btn btn-lg"
            >
              <Volume2 className="h-4 w-4" />
              语音提示
            </button>
          </div>

          {/* Tactile D-Pad */}
          <div className="flex flex-col items-center gap-2 pt-2">
            <button
              type="button"
              onClick={() => move(0, -1)}
              className="btn h-12 w-12 p-0"
              aria-label="向上移动"
            >
              <ArrowUp className="h-5 w-5" />
            </button>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => move(-1, 0)}
                className="btn h-12 w-12 p-0"
                aria-label="向左移动"
              >
                <ArrowLeft className="h-5 w-5" />
              </button>
              <button
                type="button"
                onClick={() => move(0, 1)}
                className="btn h-12 w-12 p-0"
                aria-label="向下移动"
              >
                <ArrowDown className="h-5 w-5" />
              </button>
              <button
                type="button"
                onClick={() => move(1, 0)}
                className="btn h-12 w-12 p-0"
                aria-label="向右移动"
              >
                <ArrowRight className="h-5 w-5" />
              </button>
            </div>
            <div className="flex items-center gap-1.5 text-[12px] text-ink-3 mt-1.5">
              <span className="kbd">W</span>
              <span className="kbd">A</span>
              <span className="kbd">S</span>
              <span className="kbd">D</span>
              <span>或方向键移动 ·</span>
              <span className="kbd">Space</span>
              <span>发射声呐</span>
            </div>
          </div>

          <div className="border-t border-line pt-4 space-y-2 text-[12.5px] text-ink-2 leading-relaxed">
            <div className="eyebrow">听觉线索说明</div>
            <ul className="space-y-1.5">
              <li className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-ink shrink-0" />
                <span>
                  <strong className="text-ink">高频短促回声</strong>：该方向近处有墙壁，距离越近音调越高。
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-signal shrink-0" />
                <span>
                  <strong className="text-ink">双音和弦信标</strong>：指示出口相对方位，左耳响向左，右耳响向右。
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-ink-3 shrink-0" />
                <span>
                  <strong className="text-ink">低沉撞击声</strong>：碰壁提示，随时可切换「透视模式」核对空间感。
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
