import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Power } from 'lucide-react';
import {
  drawPresetScene,
  processFrameToAcousticMatrix,
  SpatialSceneMetrics,
} from '../engine/visionProcessor';
import {
  detectSpatialObjects,
  loadOnDeviceAIModel,
  SpatialDetection,
  speakCue,
  speakSpatialDetections,
} from '../engine/aiObjectDetector';
import { spatialAudio } from '../engine/spatialAudio';
import { analyzeLightAndColor } from '../engine/blindAssistTools';
import {
  buildDetailedSceneReport,
  respondToUserUtterance,
} from '../engine/xiaoGuangAssistant';
import { AcousticThreatMonitor } from '../engine/acousticThreatDetector';

const PROC_W = 640;
const PROC_H = 420;
const COLS = 48;
const ROWS = 16;

export const BlindCompanionView: React.FC = () => {
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const [active, setActive] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [cameraLive, setCameraLive] = useState(false);
  const [scanAudioOn, setScanAudioOn] = useState(true);
  const [volume, setVolume] = useState(0.55);
  const [finderTarget, setFinderTarget] = useState<{ id: string; label: string } | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [assistantMessage, setAssistantMessage] = useState(
    '你好，我是小光。轻触上方任意区域，我就打开眼睛和语音，陪你一起安心走路。'
  );
  const [lastUserSpeech, setLastUserSpeech] = useState<string>('');
  const [detections, setDetections] = useState<SpatialDetection[]>([]);
  const [groundAlert, setGroundAlert] = useState(false);
  const [overheadAlert, setOverheadAlert] = useState(false);
  const [rearAcousticActive, setRearAcousticActive] = useState(false);
  const [rearAlertMsg, setRearAlertMsg] = useState<string>('');
  const acousticMonitorRef = useRef<AcousticThreatMonitor | null>(null);

  const detectionsRef = useRef<SpatialDetection[]>([]);
  const metricsRef = useRef<SpatialSceneMetrics>({
    leftDensity: 0,
    centerDensity: 0,
    rightDensity: 0,
    edgeCoverage: 0,
    navigationCue: '前方视野开阔',
  });
  const matrixRef = useRef<number[][]>(
    Array.from({ length: COLS }, () => new Array(ROWS).fill(0))
  );
  const groundBlockedRef = useRef(false);
  const overheadBlockedRef = useRef(false);
  const finderRef = useRef<{ id: string; label: string } | null>(null);
  finderRef.current = finderTarget;

  const scanColRef = useRef(0);
  const lastScanStepRef = useRef(0);
  const lastFinderPingRef = useRef(0);
  const lastGroundWarnRef = useRef(0);
  const lastReassureRef = useRef(0);

  const say = useCallback((text: string) => {
    setAssistantMessage(text);
    speakCue(text, 1.04);
  }, []);

  // Stop camera & acoustic monitor cleanly on unmount
  useEffect(() => {
    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
      acousticMonitorRef.current?.stop();
    };
  }, []);

  // Start / stop the eyes-free companion
  const toggleCompanion = useCallback(async () => {
    spatialAudio.ensureContext();

    if (active) {
      setActive(false);
      setCameraLive(false);
      setRearAcousticActive(false);
      setRearAlertMsg('');
      acousticMonitorRef.current?.stop();
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      }
      say('好的，我先歇一会儿。随时轻触屏幕，我就继续帮你看路。');
      return;
    }

    setInitializing(true);
    say('你好，我是小光，正在打开高清镜头、360度声学警戒和语音领路，请稍等两秒。');

    // 1. Request high-res rear camera for longer detection range (up to ~10m+)
    let gotCam = false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraLive(true);
      gotCam = true;
    } catch {
      setCameraLive(false);
    }

    // 2. Start 360° rear/side acoustic threat monitor (detects horns & sudden approaching vehicles)
    const monitor = new AcousticThreatMonitor();
    acousticMonitorRef.current = monitor;
    const micOk = await monitor.start((ev) => {
      setRearAlertMsg(ev.message);
      spatialAudio.playCollisionThud(0.85);
      say(ev.message);
      window.setTimeout(() => setRearAlertMsg(''), 6000);
    });
    setRearAcousticActive(micOk);

    // 3. Load high-accuracy on-device AI model
    await loadOnDeviceAIModel();
    setInitializing(false);
    setActive(true);
    lastReassureRef.current = performance.now();

    if (gotCam) {
      say(
        micOk
          ? '10米前向安全圈和360度后方声学警戒都已开启。放心往前走，随时点下半屏跟我说话。'
          : '10米前向安全圈和语音播报已开启。放心往前走，随时点下半屏跟我说话。'
      );
    } else {
      say('当前设备没有可用摄像头，我先用模拟街景陪你演示。语音播报已开启，随时点下半屏跟我对话。');
    }
  }, [active, say]);

  // Continuous vision frame processing + gentle 3D audio scan + ground/overhead unknown-obstacle check
  useEffect(() => {
    if (!active) return;
    let rafId = 0;

    const loop = (now: number) => {
      const src = sourceCanvasRef.current;
      const ctx = src?.getContext('2d', { willReadFrequently: true });
      if (src && ctx) {
        if (cameraLive && videoRef.current && videoRef.current.readyState >= 2) {
          ctx.drawImage(videoRef.current, 0, 0, PROC_W, PROC_H);
        } else {
          drawPresetScene(ctx, PROC_W, PROC_H, 'crosswalk');
        }

        const imgData = ctx.getImageData(0, 0, PROC_W, PROC_H);
        const res = processFrameToAcousticMatrix(imgData, COLS, ROWS, 38);
        matrixRef.current = res.matrix;
        metricsRef.current = res.metrics;

        // Check ground/curb obstacle (center cols 16..32, bottom rows 12..15)
        // and overhead branch/sign obstacle (center cols 16..32, top rows 0..3)
        let groundSum = 0;
        let overheadSum = 0;
        let cells = 0;
        for (let c = 16; c < 32; c++) {
          for (let r = 12; r < 16; r++) {
            groundSum += res.matrix[c][r];
            cells++;
          }
          for (let r = 0; r < 4; r++) {
            overheadSum += res.matrix[c][r];
          }
        }
        const gBlocked = groundSum / Math.max(1, cells) > 0.44;
        const oBlocked = overheadSum / Math.max(1, cells) > 0.46;
        groundBlockedRef.current = gBlocked;
        overheadBlockedRef.current = oBlocked;
        setGroundAlert(gBlocked);
        setOverheadAlert(oBlocked);

        // Gentle 3D spatial soundscape (subtle enough not to drown out speech)
        if (scanAudioOn && volume > 0.05 && now - lastScanStepRef.current >= 75) {
          lastScanStepRef.current = now;
          const col = scanColRef.current;
          spatialAudio.playScanColumn(res.matrix[col], col / (COLS - 1), 0.07, {
            scaleMode: 'pentatonic',
            useHRTF: true,
            volume: volume * 0.45,
          });
          scanColRef.current = (col + 1) % COLS;
        }
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [active, cameraLive, scanAudioOn, volume]);

  // Continuous always-on AI object detection + automatic meter-level voice broadcast
  useEffect(() => {
    if (!active) {
      setDetections([]);
      detectionsRef.current = [];
      return;
    }

    let cancelled = false;
    const tickDetect = async () => {
      const src = sourceCanvasRef.current;
      if (!src) return;
      const res = await detectSpatialObjects(src, !cameraLive, 'crosswalk');
      if (cancelled) return;

      detectionsRef.current = res;
      setDetections(res);

      const now = performance.now();

      // 1. 3D Item-Finder beacon if user asked XiaoGuang to find something
      const target = finderRef.current;
      if (target) {
        const hit = res.find(
          (d) => d.labelEn === target.id || (target.id === 'cup' && d.labelEn === 'bottle')
        );
        if (hit) {
          const [bx, , bw] = hit.bbox;
          const normX = Math.max(0, Math.min(1, (bx + bw / 2) / PROC_W));
          const centeredness = 1 - Math.abs(normX - 0.5) * 2;
          const closeness = Math.min(1, centeredness * 0.65 + (hit.distanceMeters <= 1.5 ? 0.35 : 0.1));
          if (now - lastFinderPingRef.current > 750) {
            lastFinderPingRef.current = now;
            spatialAudio.playFinderBeacon(normX, closeness, Math.max(0.55, volume));
          }
        }
      }

      // 2. Always-on smart voice narration (meters + direction + ground safety)
      let fallbackCue: string | undefined;
      if (res.length === 0 && groundBlockedRef.current && now - lastGroundWarnRef.current > 9000) {
        lastGroundWarnRef.current = now;
        fallbackCue = '注意脚下，正前方大约1到2米路面有未知障碍或台阶，放慢一点脚步。';
      } else if (res.length === 0 && now - lastReassureRef.current > 24000) {
        lastReassureRef.current = now;
        fallbackCue = '前方依然很开阔，走得很稳，我一直陪着你。';
      }

      const spoken = speakSpatialDetections(res, false, fallbackCue);
      if (spoken) {
        setAssistantMessage(spoken);
        lastReassureRef.current = now;
      }
    };

    tickDetect();
    const timer = window.setInterval(tickDetect, cameraLive ? 850 : 2600);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [active, cameraLive, volume]);

  // Process a spoken or tapped utterance with XiaoGuang
  const handleUtterance = useCallback(
    (utterance: string) => {
      setLastUserSpeech(utterance);
      const src = sourceCanvasRef.current;
      const reply = respondToUserUtterance(utterance, {
        detections: detectionsRef.current,
        metrics: metricsRef.current,
        groundBlocked: groundBlockedRef.current,
        overheadBlocked: overheadBlockedRef.current,
        lightReport: null,
      });

      if (reply.action.type === 'set_finder') {
        setFinderTarget({ id: reply.action.targetId, label: reply.action.targetLabel });
      } else if (reply.action.type === 'clear_finder') {
        setFinderTarget(null);
      } else if (reply.action.type === 'check_light_color' && src) {
        const light = analyzeLightAndColor(src);
        say(light.speechText);
        return;
      } else if (reply.action.type === 'set_scan') {
        setScanAudioOn(reply.action.enabled);
      } else if (reply.action.type === 'set_volume') {
        const d = reply.action.delta;
        if (d === 'mute') setVolume(0);
        else if (d === 'unmute') setVolume(0.6);
        else setVolume((v) => Math.max(0.1, Math.min(1, v + d)));
      } else if (reply.action.type === 'pause_all') {
        if (active) toggleCompanion();
        return;
      } else if (reply.action.type === 'resume_all') {
        if (!active) toggleCompanion();
        return;
      }

      say(reply.text);
    },
    [active, say, toggleCompanion]
  );

  // Tap bottom half to talk to XiaoGuang
  const startTalkingToXiaoGuang = useCallback(() => {
    spatialAudio.ensureContext();
    if (typeof window === 'undefined') return;
    type SpeechRecCtor = new () => {
      lang: string;
      interimResults: boolean;
      maxAlternatives: number;
      onstart: (() => void) | null;
      onend: (() => void) | null;
      onerror: (() => void) | null;
      onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
      start: () => void;
    };
    const w = window as unknown as {
      SpeechRecognition?: SpeechRecCtor;
      webkitSpeechRecognition?: SpeechRecCtor;
    };
    const SpeechRec = w.SpeechRecognition || w.webkitSpeechRecognition;

    if (!SpeechRec) {
      // Graceful fallback: immediately read detailed meter-level road report
      const report = buildDetailedSceneReport({
        detections: detectionsRef.current,
        metrics: metricsRef.current,
        groundBlocked: groundBlockedRef.current,
        overheadBlocked: overheadBlockedRef.current,
        lightReport: null,
      });
      say(report.text);
      return;
    }

    // Play a soft chime so blind user knows XiaoGuang is listening
    spatialAudio.playBrushTone(0.5, 0.25, { scaleMode: 'pentatonic', useHRTF: true, volume: 0.55 });

    const rec = new SpeechRec();
    rec.lang = 'zh-CN';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onstart = () => setIsListening(true);
    rec.onend = () => setIsListening(false);
    rec.onerror = () => {
      setIsListening(false);
      say('没太听清，你可以再点一下下半屏跟我说，或者放心继续走。');
    };
    rec.onresult = (ev) => {
      const transcript = ev.results?.[0]?.[0]?.transcript || '';
      handleUtterance(transcript);
    };
    try {
      rec.start();
    } catch {
      setIsListening(false);
    }
  }, [handleUtterance, say]);

  return (
    <div className="mx-auto max-w-xl space-y-5">
      {/* Hidden processing canvas */}
      <canvas ref={sourceCanvasRef} width={PROC_W} height={PROC_H} className="hidden" />

      {/* TOP HALF: Giant single-tap eyes-free activation & live voice status */}
      <button
        type="button"
        onClick={toggleCompanion}
        disabled={initializing}
        aria-pressed={active}
        className="w-full text-left panel p-5 sm:p-6 space-y-4 transition-colors hover:border-ink-2 focus-visible:outline-signal"
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className={`led ${active ? 'led-on' : ''}`} />
            <span className="eyebrow text-ink">
              {initializing ? '正在准备...' : active ? '小光正在帮你看路 · 语音已常开' : '轻触此处 · 一键开始领路'}
            </span>
          </div>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[12px] font-semibold ${
              active ? 'bg-signal text-white' : 'bg-ink text-paper'
            }`}
          >
            <Power className="h-3.5 w-3.5" />
            {active ? '运行中（点此暂停）' : '点此开始'}
          </span>
        </div>

        {/* 360° Three-Zone Safety Ring Radar (2m / 5m / 10m) + Camera Feed */}
        <div className="screen relative h-[220px] sm:h-[240px]">
          <video
            ref={videoRef}
            playsInline
            muted
            className={`absolute inset-0 w-full h-full object-cover opacity-28 ${active && cameraLive ? 'block' : 'hidden'}`}
          />

          <svg
            viewBox="0 0 360 220"
            className="relative z-10 w-full h-full block"
            role="img"
            aria-label="360度个人安全守护圈雷达：内圈2米紧急避碰，中圈5米主动绕行，外圈10米远距预警"
          >
            {/* 360° rear/side acoustic guard ring */}
            <circle
              cx="180"
              cy="126"
              r="86"
              fill="none"
              stroke={rearAlertMsg ? '#e4572e' : rearAcousticActive ? 'rgba(159, 201, 176, 0.32)' : 'rgba(243, 237, 226, 0.12)'}
              strokeWidth={rearAlertMsg ? '2.5' : '1.2'}
              strokeDasharray="4 4"
            />
            {/* 10m outer ring, 5m middle ring, 2m inner danger ring */}
            <circle cx="180" cy="126" r="76" fill="none" stroke="rgba(243, 237, 226, 0.16)" strokeWidth="1" />
            <circle cx="180" cy="126" r="50" fill="none" stroke="rgba(242, 184, 75, 0.28)" strokeWidth="1" />
            <circle cx="180" cy="126" r="26" fill="rgba(228, 87, 46, 0.08)" stroke="rgba(228, 87, 46, 0.5)" strokeWidth="1.2" />

            {/* Forward camera FOV sector */}
            <path
              d="M 180 126 L 116 48 A 88 88 0 0 1 244 48 Z"
              fill="rgba(255, 138, 61, 0.09)"
              stroke="rgba(255, 138, 61, 0.35)"
              strokeWidth="1"
            />

            {/* Ring meter labels */}
            <text x="184" y="104" fill="rgba(228, 87, 46, 0.9)" fontSize="9.5" fontFamily="monospace">2m</text>
            <text x="184" y="80" fill="rgba(242, 184, 75, 0.85)" fontSize="9.5" fontFamily="monospace">5m</text>
            <text x="184" y="54" fill="rgba(243, 237, 226, 0.65)" fontSize="9.5" fontFamily="monospace">10m</text>
            <text x="148" y="206" fill="rgba(159, 201, 176, 0.72)" fontSize="9.5" fontFamily="monospace">
              {rearAcousticActive ? '后方 360° 听觉警戒开' : '后方声学盲区警戒'}
            </text>

            {/* Plotted real-time objects inside the 10m safety circle */}
            {active &&
              detections.slice(0, 6).map((d, idx) => {
                const angleDeg =
                  d.direction === '正左侧'
                    ? -55
                    : d.direction === '左前方'
                      ? -28
                      : d.direction === '正前方'
                        ? 0
                        : d.direction === '右前方'
                          ? 28
                          : 55;
                const rad = (angleDeg * Math.PI) / 180;
                const normDist = Math.min(1, d.distanceMeters / 10);
                const r = 18 + normDist * 58;
                const px = 180 + Math.sin(rad) * r;
                const py = 126 - Math.cos(rad) * r;
                const dotColor =
                  d.distanceMeters <= 2.2 || d.priority === 'danger' ? '#ff8a3d' : '#9fc9b0';
                return (
                  <g key={idx}>
                    <circle cx={px} cy={py} r="4.5" fill={dotColor} />
                    <text
                      x={px + 7}
                      y={py + 3}
                      fill={dotColor}
                      fontSize="10"
                      fontFamily="monospace"
                    >
                      {d.labelZh} {d.distanceText}
                    </text>
                  </g>
                );
              })}

            {/* Center user marker */}
            <circle cx="180" cy="126" r="7" fill="#f3ede2" stroke="#ff8a3d" strokeWidth="2" />
          </svg>

          {!active && (
            <div className="pointer-events-none absolute top-3 left-3 right-3 flex justify-center">
              <span className="screen-label">
                轻触此处即可开启 10 米安全守护圈 + 360° 后方鸣笛声学警戒
              </span>
            </div>
          )}

          {active && (groundAlert || overheadAlert || finderTarget || rearAlertMsg) && (
            <div className="pointer-events-none absolute bottom-2.5 left-2.5 right-2.5 flex flex-wrap gap-1.5">
              {rearAlertMsg && <span className="screen-label text-amber">后方/侧面声学急报！</span>}
              {finderTarget && (
                <span className="screen-label text-amber">3D 寻物锁定：{finderTarget.label}</span>
              )}
              {groundAlert && <span className="screen-label text-amber">注意脚下约1.5米障碍/台阶</span>}
              {overheadAlert && <span className="screen-label text-amber">注意头顶上方遮挡</span>}
            </div>
          )}
        </div>

        {/* Large, clear real-time speech readout */}
        <div className="panel-sunken p-4 border-l-2 border-l-signal space-y-1.5">
          <div className="flex items-center justify-between text-[11.5px] text-ink-3">
            <span className="eyebrow text-signal-ink">小光实时播报</span>
            {lastUserSpeech && <span>你刚才说：「{lastUserSpeech}」</span>}
          </div>
          <p className="title-serif text-[18px] sm:text-[20px] leading-relaxed text-ink">
            {assistantMessage}
          </p>
        </div>

        {/* Exact meter-level obstacle chips (only shown when detected) */}
        {active && detections.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {detections.slice(0, 4).map((d, i) => (
              <span key={i} className="chip">
                <strong className="text-signal-ink">{d.direction}</strong>
                <span className="num font-semibold text-ink">{d.distanceText}</span>
                <span>{d.labelZh}</span>
              </span>
            ))}
          </div>
        )}
      </button>

      {/* BOTTOM HALF: Giant single-tap voice companion button (zero hunting for tiny controls) */}
      <div className="panel p-5 sm:p-6 flex flex-col items-center text-center space-y-4">
        <button
          type="button"
          onClick={startTalkingToXiaoGuang}
          className={`w-full min-h-[118px] rounded-2xl border-2 flex flex-col items-center justify-center gap-2 transition-colors ${
            isListening
              ? 'border-signal bg-signal text-white'
              : 'border-ink bg-ink text-paper hover:bg-ink/90'
          }`}
        >
          {isListening ? <MicOff className="h-8 w-8" /> : <Mic className="h-8 w-8" />}
          <span className="title-serif text-[20px] text-inherit">
            {isListening ? '小光正在听你说…' : '点这里，随时跟小光说话'}
          </span>
          <span className="text-[12.5px] opacity-80">
            问路况几米 · 找椅子/水杯 · 测开灯没 · 语音静音 · 紧张时陪你聊两句
          </span>
        </button>

        {/* 3 gentle quick-talk pills for quick demo / one-tap check */}
        <div className="flex flex-wrap items-center justify-center gap-2">
          {[
            '前面几米有什么？',
            '我有点紧张',
            '帮我找椅子',
            '开灯了吗？什么颜色？',
          ].map((phrase) => (
            <button
              key={phrase}
              type="button"
              onClick={() => handleUtterance(phrase)}
              className="btn btn-sm"
            >
              “{phrase}”
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
