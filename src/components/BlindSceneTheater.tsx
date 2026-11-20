import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Eye,
  EyeOff,
  Volume2,
  VolumeX,
  Headphones,
} from 'lucide-react';
import { spatialAudio } from '../engine/spatialAudio';
import { speakCue } from '../engine/aiObjectDetector';

type ScenarioId = 'crosswalk' | 'find_cup' | 'sidewalk' | 'subway';

interface TimelineEvent {
  atSec: number;
  label: string;
  voice: string;
  soundType: 'hazard_right' | 'hazard_left' | 'green_light' | 'finder_right' | 'finder_center' | 'clear_path';
}

interface ScenarioDef {
  id: ScenarioId;
  no: string;
  title: string;
  shortDesc: string;
  durationSec: number;
  hardwareHint: string;
  events: TimelineEvent[];
}

const SCENARIOS: ScenarioDef[] = [
  {
    id: 'crosswalk',
    no: '01',
    title: '十字路口过马路',
    shortDesc: '红绿灯变色提醒 + 右侧来车 3D 避让',
    durationSec: 12,
    hardwareHint: '手机胸前挂绳 / AI 眼镜 + 开放式骨传导耳机（不堵耳朵）',
    events: [
      {
        atSec: 0.4,
        label: '红灯等待',
        voice: '正前方斑马线，交通信号灯当前红灯，请原地等待。',
        soundType: 'hazard_left',
      },
      {
        atSec: 3.4,
        label: '绿灯通行',
        voice: '信号灯已变绿灯，准备通行。',
        soundType: 'green_light',
      },
      {
        atSec: 5.4,
        label: '右侧来车预警',
        voice: '注意！右前方近距离有转弯车辆靠近，请停步避让。',
        soundType: 'hazard_right',
      },
      {
        atSec: 8.8,
        label: '安全通过',
        voice: '右侧车辆已驶离，正前方开阔，可平稳通过斑马线。',
        soundType: 'clear_path',
      },
    ],
  },
  {
    id: 'find_cup',
    no: '02',
    title: '暗光客厅找水杯',
    shortDesc: '测光辨灯 + 避开餐桌 + 3D 寻物信标锁定',
    durationSec: 12,
    hardwareHint: '普通手机摄像头 + 普通双耳耳机',
    events: [
      {
        atSec: 0.4,
        label: '测光识环境',
        voice: '环境较暗，室内灯光未开。已开启 3D 寻物雷达，正在寻找水杯。',
        soundType: 'clear_path',
      },
      {
        atSec: 3.2,
        label: '左侧餐桌避障',
        voice: '左前方近距离有餐桌障碍，请向右前方微调绕行。',
        soundType: 'hazard_left',
      },
      {
        atSec: 6.2,
        label: '右耳寻物信标',
        voice: '右前方中距离发现水杯，请顺着右耳滴答声转向。',
        soundType: 'finder_right',
      },
      {
        atSec: 9.2,
        label: '正前伸手可及',
        voice: '水杯已在正前方近距离，伸手即可拿到。',
        soundType: 'finder_center',
      },
    ],
  },
  {
    id: 'sidewalk',
    no: '03',
    title: '盲道避让消防栓与单车',
    shortDesc: '识别占道障碍 + 左耳警示 + 平滑向右绕行',
    durationSec: 11,
    hardwareHint: 'AI 眼镜 / 手机挂绳 + 骨传导耳机 + 盲杖协同',
    events: [
      {
        atSec: 0.4,
        label: '沿盲道直行',
        voice: '正前方人行道开阔，可沿盲道平稳直行。',
        soundType: 'clear_path',
      },
      {
        atSec: 3.2,
        label: '左侧占道预警',
        voice: '注意，左前方近距离有消防栓与违停自行车，建议向右绕行。',
        soundType: 'hazard_left',
      },
      {
        atSec: 7.4,
        label: '绕过障碍恢复',
        voice: '已安全绕过占道障碍，前方通道畅通。',
        soundType: 'green_light',
      },
    ],
  },
  {
    id: 'subway',
    no: '04',
    title: '地铁大厅寻找出口通道',
    shortDesc: '人群动态避让 + 明亮出口 3D 声学引导',
    durationSec: 11,
    hardwareHint: '手机 / 智能眼镜 + 开放式耳机',
    events: [
      {
        atSec: 0.4,
        label: '定位出口方位',
        voice: '大厅右前方较远处检测到明亮出口通道，左耳与右耳将持续引导方位。',
        soundType: 'finder_right',
      },
      {
        atSec: 3.8,
        label: '避让迎面行人',
        voice: '左前方近距离有迎面行人走来，请略微靠右慢行。',
        soundType: 'hazard_left',
      },
      {
        atSec: 7.6,
        label: '抵达出口闸机',
        voice: '正前方已对准出口通道，声场居中，可直行出站。',
        soundType: 'finder_center',
      },
    ],
  },
];

const W = 560;
const H = 330;

export const BlindSceneTheater: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [scenarioId, setScenarioId] = useState<ScenarioId>('crosswalk');
  const [isPlaying, setIsPlaying] = useState(false);
  const [blindView, setBlindView] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [currentSec, setCurrentSec] = useState(0);
  const [activeEventIdx, setActiveEventIdx] = useState(-1);

  const timeRef = useRef(0);
  const lastFrameRef = useRef<number | null>(null);
  const firedEventsRef = useRef<Set<number>>(new Set());
  const lastAudioPulseRef = useRef(0);

  const scenario = SCENARIOS.find((s) => s.id === scenarioId) || SCENARIOS[0];

  const triggerEventAudio = useCallback(
    (ev: TimelineEvent) => {
      spatialAudio.ensureContext();
      if (ev.soundType === 'hazard_right') {
        spatialAudio.playFinderBeacon(0.88, 0.85, 0.72);
      } else if (ev.soundType === 'hazard_left') {
        spatialAudio.playFinderBeacon(0.14, 0.78, 0.72);
      } else if (ev.soundType === 'green_light') {
        spatialAudio.playVictoryChord(0.55);
      } else if (ev.soundType === 'finder_right') {
        spatialAudio.playFinderBeacon(0.82, 0.55, 0.72);
      } else if (ev.soundType === 'finder_center') {
        spatialAudio.playFinderBeacon(0.5, 0.96, 0.75);
      } else {
        spatialAudio.playBrushTone(0.5, 0.45, {
          scaleMode: 'pentatonic',
          useHRTF: true,
          volume: 0.5,
        });
      }
      if (voiceEnabled) {
        speakCue(ev.voice);
      }
    },
    [voiceEnabled]
  );

  const resetScenario = useCallback((nextId?: ScenarioId) => {
    if (nextId) setScenarioId(nextId);
    timeRef.current = 0;
    lastFrameRef.current = null;
    firedEventsRef.current = new Set();
    setCurrentSec(0);
    setActiveEventIdx(-1);
    setIsPlaying(false);
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const togglePlay = () => {
    spatialAudio.ensureContext();
    if (timeRef.current >= scenario.durationSec - 0.1) {
      timeRef.current = 0;
      firedEventsRef.current = new Set();
      setActiveEventIdx(-1);
    }
    lastFrameRef.current = null;
    setIsPlaying((p) => !p);
  };

  // Main 60fps theater animation + synchronized 3D HRTF audio loop
  useEffect(() => {
    let rafId = 0;

    const drawBlindUser = (
      ctx: CanvasRenderingContext2D,
      ux: number,
      uy: number,
      headingRad: number,
      pulsePhase: number
    ) => {
      // Field of view cone (camera / AI glasses FOV)
      ctx.save();
      ctx.translate(ux, uy);
      ctx.rotate(headingRad);

      ctx.fillStyle = 'rgba(255, 138, 61, 0.10)';
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.arc(0, 0, 145, -Math.PI / 2 - 0.55, -Math.PI / 2 + 0.55);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = 'rgba(255, 138, 61, 0.35)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(Math.sin(-0.55) * 145, -Math.cos(-0.55) * 145);
      ctx.moveTo(0, 0);
      ctx.lineTo(Math.sin(0.55) * 145, -Math.cos(0.55) * 145);
      ctx.stroke();

      // Expanding subtle acoustic sonar wave
      const r = 18 + (pulsePhase % 1) * 95;
      ctx.strokeStyle = `rgba(243, 237, 226, ${Math.max(0, 0.32 * (1 - (pulsePhase % 1)))})`;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.stroke();

      // White cane
      ctx.strokeStyle = '#f3ede2';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(9, -4);
      ctx.lineTo(19, -34);
      ctx.stroke();

      // User head + bone-conduction headphones
      ctx.fillStyle = '#f3ede2';
      ctx.beginPath();
      ctx.arc(0, 0, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ff8a3d';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(0, 0, 12.5, Math.PI * 0.9, Math.PI * 2.1);
      ctx.stroke();
      // Left & right ear transducers
      ctx.fillStyle = '#ff8a3d';
      ctx.beginPath();
      ctx.arc(-12, 0, 3.2, 0, Math.PI * 2);
      ctx.arc(12, 0, 3.2, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
    };

    const drawAcousticWaveFrom = (
      ctx: CanvasRenderingContext2D,
      sx: number,
      sy: number,
      ux: number,
      uy: number,
      color: string,
      phase: number
    ) => {
      const r = 14 + (phase % 1) * 52;
      ctx.strokeStyle = color;
      ctx.globalAlpha = Math.max(0.1, 0.8 * (1 - (phase % 1)));
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.stroke();

      // Dashed directional ray to ear
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ux, uy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    };

    const renderFrame = (now: number) => {
      if (isPlaying) {
        if (lastFrameRef.current !== null) {
          const dt = Math.min(0.1, (now - lastFrameRef.current) / 1000);
          timeRef.current = Math.min(scenario.durationSec, timeRef.current + dt);
          setCurrentSec(timeRef.current);
          if (timeRef.current >= scenario.durationSec) {
            setIsPlaying(false);
          }
        }
        lastFrameRef.current = now;

        // Fire timeline events once when crossed
        scenario.events.forEach((ev, idx) => {
          if (timeRef.current >= ev.atSec && !firedEventsRef.current.has(idx)) {
            firedEventsRef.current.add(idx);
            setActiveEventIdx(idx);
            triggerEventAudio(ev);
          }
        });
      }

      const t = timeRef.current;
      const pulsePhase = (now / 1200) % 1;
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');

      if (canvas && ctx) {
        ctx.fillStyle = '#161512';
        ctx.fillRect(0, 0, W, H);

        // Subtle coordinate grid
        if (!blindView) {
          ctx.strokeStyle = 'rgba(243, 237, 226, 0.045)';
          ctx.lineWidth = 1;
          for (let gx = 40; gx < W; gx += 40) {
            ctx.beginPath();
            ctx.moveTo(gx, 0);
            ctx.lineTo(gx, H);
            ctx.stroke();
          }
          for (let gy = 40; gy < H; gy += 40) {
            ctx.beginPath();
            ctx.moveTo(0, gy);
            ctx.lineTo(W, gy);
            ctx.stroke();
          }
        }

        if (scenario.id === 'crosswalk') {
          const isGreen = t >= 3.4;
          // User walks up, pauses during car pass (5.4s..8.6s), then finishes crossing
          let uy = 285;
          if (t < 3.4) uy = 285 - t * 6;
          else if (t < 5.4) uy = 264 - (t - 3.4) * 30;
          else if (t < 8.6) uy = 204;
          else uy = Math.max(55, 204 - (t - 8.6) * 44);
          const ux = 250;

          // Car enters from right between 4.8s and 8.8s
          const carProgress = Math.max(0, Math.min(1, (t - 4.8) / 4.0));
          const carX = 590 - carProgress * 640;
          const carY = 148;
          const carVisible = t >= 4.8 && t <= 8.8;

          if (!blindView) {
            // Road & crosswalk stripes
            ctx.fillStyle = '#201e1a';
            ctx.fillRect(0, 90, W, 130);
            ctx.fillStyle = 'rgba(243, 237, 226, 0.22)';
            for (let i = 0; i < 6; i++) {
              ctx.fillRect(195 + i * 22, 98, 12, 114);
            }
            // Traffic light housing on top-left of crosswalk
            ctx.fillStyle = '#2b2823';
            ctx.fillRect(156, 36, 26, 46);
            ctx.strokeStyle = 'rgba(243, 237, 226, 0.3)';
            ctx.strokeRect(156, 36, 26, 46);
            ctx.fillStyle = isGreen ? '#3a3730' : '#e4572e';
            ctx.beginPath();
            ctx.arc(169, 49, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = isGreen ? '#7fbf8e' : '#3a3730';
            ctx.beginPath();
            ctx.arc(169, 69, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = isGreen ? '#9fc9b0' : '#f2b84b';
            ctx.font = '600 11px "JetBrains Mono", Consolas, monospace';
            ctx.fillText(isGreen ? '绿灯 PASS' : '红灯 WAIT', 190, 62);

            if (carVisible) {
              ctx.fillStyle = '#f2b84b';
              ctx.fillRect(carX - 34, carY - 18, 68, 36);
              ctx.fillStyle = '#161512';
              ctx.fillRect(carX - 18, carY - 12, 26, 24);
              ctx.fillStyle = '#f2b84b';
              ctx.fillText('来向车辆', carX - 24, carY - 24);
            }
          }

          // Acoustic cues visible even in blindView!
          if (carVisible && carX > ux - 20) {
            drawAcousticWaveFrom(ctx, carX, carY, ux, uy, '#ff8a3d', pulsePhase * 1.8);
            if (isPlaying && now - lastAudioPulseRef.current > 850 && t > 5.6 && t < 8.2) {
              lastAudioPulseRef.current = now;
              const normX = Math.max(0.1, Math.min(0.95, carX / W));
              spatialAudio.playFinderBeacon(normX, 0.85, 0.55);
            }
          } else if (isGreen) {
            drawAcousticWaveFrom(ctx, 169, 69, ux, uy, '#9fc9b0', pulsePhase);
          }

          drawBlindUser(ctx, ux, uy, 0, pulsePhase);
        } else if (scenario.id === 'find_cup') {
          // User walks into dark living room, steers right around left table, turns toward cup on right coffee table
          let ux = 250;
          let uy = 285;
          let heading = 0;
          if (t < 3.2) {
            uy = 285 - t * 18;
          } else if (t < 6.2) {
            const p = (t - 3.2) / 3.0;
            ux = 250 + p * 70;
            uy = 227 - p * 45;
            heading = 0.45;
          } else if (t < 9.2) {
            const p = (t - 6.2) / 3.0;
            ux = 320 + p * 55;
            uy = 182 - p * 45;
            heading = 0.65;
          } else {
            ux = 375;
            uy = 137;
            heading = 0.6;
          }

          const tableX = 165;
          const tableY = 175;
          const cupX = 438;
          const cupY = 104;

          if (!blindView) {
            // Left dining table obstacle
            ctx.fillStyle = '#2c2924';
            ctx.fillRect(105, 140, 120, 70);
            ctx.strokeStyle = 'rgba(243, 237, 226, 0.35)';
            ctx.strokeRect(105, 140, 120, 70);
            ctx.fillStyle = '#f3ede2';
            ctx.font = '600 11px "JetBrains Mono", Consolas, monospace';
            ctx.fillText('餐桌障碍', 142, 179);

            // Right coffee table + cup
            ctx.fillStyle = '#25231f';
            ctx.fillRect(405, 76, 75, 58);
            ctx.strokeStyle = 'rgba(243, 237, 226, 0.25)';
            ctx.strokeRect(405, 76, 75, 58);

            ctx.fillStyle = '#ff8a3d';
            ctx.beginPath();
            ctx.arc(cupX, cupY, 9, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillText('目标：水杯', 402, 66);
          }

          if (t >= 3.0 && t < 6.0) {
            drawAcousticWaveFrom(ctx, tableX, tableY, ux, uy, '#f2b84b', pulsePhase * 1.4);
          }
          if (t >= 6.0) {
            drawAcousticWaveFrom(ctx, cupX, cupY, ux, uy, '#ff8a3d', pulsePhase * 1.9);
            if (isPlaying && now - lastAudioPulseRef.current > 800 && t > 6.6 && t < 10.6) {
              lastAudioPulseRef.current = now;
              const closeness = Math.min(0.95, 0.45 + ((t - 6.0) / 4.5) * 0.5);
              spatialAudio.playFinderBeacon(t > 8.8 ? 0.52 : 0.82, closeness, 0.55);
            }
          }

          drawBlindUser(ctx, ux, uy, heading, pulsePhase);
        } else if (scenario.id === 'sidewalk') {
          // Blind path vertical strip, hydrant + parked bike blocking left side, user arcs smoothly right
          const p = Math.min(1, t / 10.5);
          const uy = 290 - p * 235;
          const dodge = Math.exp(-Math.pow((p - 0.52) / 0.18, 2)) * 78;
          const ux = 240 + dodge;
          const heading = p > 0.3 && p < 0.52 ? 0.4 : p >= 0.52 && p < 0.72 ? -0.35 : 0;

          if (!blindView) {
            // Tactile paving strip (盲道)
            ctx.fillStyle = 'rgba(242, 184, 75, 0.12)';
            ctx.fillRect(222, 0, 36, H);
            ctx.strokeStyle = 'rgba(242, 184, 75, 0.25)';
            ctx.setLineDash([6, 6]);
            ctx.beginPath();
            ctx.moveTo(240, 0);
            ctx.lineTo(240, H);
            ctx.stroke();
            ctx.setLineDash([]);

            // Fire hydrant + parked bike occupying left/center of tactile strip
            ctx.fillStyle = '#e4572e';
            ctx.beginPath();
            ctx.arc(216, 168, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#2d2a24';
            ctx.fillRect(160, 148, 44, 42);
            ctx.strokeStyle = '#f2b84b';
            ctx.strokeRect(160, 148, 44, 42);
            ctx.fillStyle = '#f2b84b';
            ctx.font = '600 11px "JetBrains Mono", Consolas, monospace';
            ctx.fillText('占道单车 + 消防栓', 120, 138);
          }

          if (t >= 2.8 && t <= 7.2) {
            drawAcousticWaveFrom(ctx, 216, 168, ux, uy, '#ff8a3d', pulsePhase * 1.7);
          }
          drawBlindUser(ctx, ux, uy, heading, pulsePhase);
        } else {
          // Subway hall: bright exit gate top-right, oncoming pedestrian on left
          const p = Math.min(1, t / 10.5);
          const ux = 220 + p * 145;
          const uy = 285 - p * 210;
          const pedX = 245 + p * 35;
          const pedY = 65 + p * 220;

          if (!blindView) {
            // Bright exit corridor top-right
            ctx.fillStyle = 'rgba(159, 201, 176, 0.22)';
            ctx.fillRect(340, 24, 96, 36);
            ctx.strokeStyle = '#9fc9b0';
            ctx.strokeRect(340, 24, 96, 36);
            ctx.fillStyle = '#9fc9b0';
            ctx.font = '600 11px "JetBrains Mono", Consolas, monospace';
            ctx.fillText('地铁出口通道', 352, 46);

            // Oncoming pedestrian
            ctx.fillStyle = '#cfc7b8';
            ctx.beginPath();
            ctx.arc(pedX, pedY, 9, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillText('迎面行人', pedX - 24, pedY - 14);
          }

          drawAcousticWaveFrom(ctx, 388, 42, ux, uy, '#9fc9b0', pulsePhase);
          if (t >= 3.4 && t <= 6.8) {
            drawAcousticWaveFrom(ctx, pedX, pedY, ux, uy, '#f2b84b', pulsePhase * 1.6);
          }
          drawBlindUser(ctx, ux, uy, 0.52, pulsePhase);
        }

        // Blindfold indicator inside screen
        if (blindView) {
          ctx.fillStyle = 'rgba(243, 237, 226, 0.55)';
          ctx.font = '600 11px "JetBrains Mono", Consolas, monospace';
          ctx.fillText('盲人视角（隐藏画面，仅保留声波与耳机定位）', 14, H - 14);
        }
      }

      rafId = requestAnimationFrame(renderFrame);
    };

    rafId = requestAnimationFrame(renderFrame);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, scenario, blindView, triggerEventAudio]);

  const activeEvent = activeEventIdx >= 0 ? scenario.events[activeEventIdx] : null;
  const progressPct = Math.min(100, (currentSec / scenario.durationSec) * 100);

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-12 items-start">
        {/* Left 7 Cols: Live Animated Theater Stage */}
        <div className="lg:col-span-7 panel p-5 space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="eyebrow">02 · Living Theater</div>
              <h2 className="title-serif text-[20px] mt-1">盲人实景动画剧场</h2>
              <p className="text-[13px] text-ink-2 mt-1">
                戴上耳机播放动画，亲眼并亲耳体验盲人使用 EchoSight 时的每一步反应。
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setBlindView((v) => !v)}
                className={`btn btn-sm ${blindView ? 'btn-active' : ''}`}
                title="切换明眼人上帝视角与盲人全黑视角"
              >
                {blindView ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                {blindView ? '盲人全黑视角' : '上帝俯视视角'}
              </button>
              <button
                type="button"
                onClick={() => setVoiceEnabled((v) => !v)}
                className="btn btn-sm"
                title="开关剧场同步中文语音播报"
              >
                {voiceEnabled ? <Volume2 className="h-3.5 w-3.5" /> : <VolumeX className="h-3.5 w-3.5" />}
                {voiceEnabled ? '语音开' : '语音关'}
              </button>
            </div>
          </div>

          {/* 4 Scenarios Selector */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {SCENARIOS.map((sc) => (
              <button
                key={sc.id}
                type="button"
                data-active={sc.id === scenarioId}
                onClick={() => resetScenario(sc.id)}
                className="tile"
              >
                <div className="eyebrow">{sc.no}</div>
                <div className="text-[12.5px] font-semibold text-ink mt-0.5 truncate">{sc.title}</div>
                <div className="text-[11px] text-ink-3 mt-0.5 truncate">{sc.shortDesc}</div>
              </button>
            ))}
          </div>

          {/* Animated Canvas Screen */}
          <div className="screen">
            <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
            <div className="pointer-events-none absolute top-3 left-3 right-3 flex items-center justify-between gap-2">
              <span className="screen-label inline-flex items-center gap-1.5">
                <span className={`led ${isPlaying ? 'led-on' : ''}`} />
                {isPlaying ? '实景模拟中' : '就绪'} · {scenario.title}
              </span>
              <span className="screen-label num">
                {currentSec.toFixed(1)}s / {scenario.durationSec}s
              </span>
            </div>
          </div>

          {/* Timeline Bar + Playback Controls */}
          <div className="space-y-3">
            <div className="meter">
              <span style={{ width: `${progressPct}%`, backgroundColor: 'var(--color-signal)' }} />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={togglePlay}
                  className={`btn ${isPlaying ? 'btn-signal' : 'btn-primary'}`}
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  {isPlaying ? '暂停剧场' : currentSec > 0 ? '继续播放' : '播放实景动画（同步 3D 音频）'}
                </button>
                <button type="button" onClick={() => resetScenario()} className="btn">
                  <RotateCcw className="h-4 w-4" />
                  重播
                </button>
              </div>
              <span className="text-[12px] text-ink-3 inline-flex items-center gap-1.5">
                <Headphones className="h-3.5 w-3.5" />
                推荐硬件：{scenario.hardwareHint}
              </span>
            </div>
          </div>
        </div>

        {/* Right 5 Cols: Real-time Earphone Transcript & Hardware Explanation */}
        <div className="lg:col-span-5 space-y-5">
          {/* Live Earphone Cue Feed */}
          <div className="panel p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="eyebrow">Earphone Feed</div>
                <h3 className="text-[15px] font-semibold text-ink mt-0.5">盲人耳机里实时听到了什么</h3>
              </div>
              <span className="chip">{scenario.events.length} 个关键节点</span>
            </div>

            <div className="panel-sunken p-3.5 border-l-2 border-l-signal text-[13px] text-ink leading-relaxed">
              {activeEvent ? (
                <>
                  <div className="eyebrow text-signal-ink mb-1">当前播报 · {activeEvent.label}</div>
                  “{activeEvent.voice}”
                </>
              ) : (
                <span className="text-ink-2">
                  点击左侧「播放实景动画」，系统会按时间轴触发真实的 3D 双耳提示音与中文空间语音播报。
                </span>
              )}
            </div>

            <ol className="space-y-2.5">
              {scenario.events.map((ev, idx) => {
                const passed = currentSec >= ev.atSec;
                const isCurrent = idx === activeEventIdx;
                return (
                  <li
                    key={ev.atSec}
                    className={`rounded-lg border px-3.5 py-2.5 text-[12.5px] transition-colors ${
                      isCurrent
                        ? 'border-signal bg-signal-soft text-ink'
                        : passed
                          ? 'border-line bg-surface text-ink-2'
                          : 'border-line bg-sunken/50 text-ink-3'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <strong className={isCurrent ? 'text-signal-ink' : 'text-ink'}>{ev.label}</strong>
                      <span className="num text-[11px]">{ev.atSec.toFixed(1)}s</span>
                    </div>
                    <p className="mt-1 leading-relaxed">{ev.voice}</p>
                  </li>
                );
              })}
            </ol>
          </div>

          {/* Hardware FAQ Card: Do blind users need special glasses & headphones? */}
          <div className="panel p-5 space-y-3">
            <div className="eyebrow">Hardware Reality</div>
            <h3 className="text-[15px] font-semibold text-ink">盲人需要买专门的眼镜和耳机吗？</h3>
            <div className="space-y-2.5 text-[12.5px] leading-relaxed text-ink-2">
              <div className="panel-sunken p-3">
                <div className="font-semibold text-ink">现在就能用（零专用硬件）</div>
                <p className="mt-0.5">
                  一部普通手机（挂在胸前挂绳、后置摄像头朝前）+ 一副百元级<strong>开放式/骨传导耳机</strong>即可运行。
                  <strong className="text-signal-ink">切忌使用降噪入耳耳机</strong>，必须保留耳朵对汽车鸣笛与环境原声的感知。
                </p>
              </div>
              <div className="panel-sunken p-3">
                <div className="font-semibold text-ink">未来终极形态（解放双手）</div>
                <p className="mt-0.5">
                  移植到轻量级 AI 眼镜（微型广角镜头 + 镜腿骨传导扬声器），像普通墨镜一样轻便，双手可完全留给盲杖或扶手。
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
