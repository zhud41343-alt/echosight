import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Power, Volume2, VolumeX, AlertTriangle } from 'lucide-react';
import {
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
import { respondToUserUtterance } from '../engine/xiaoGuangAssistant';
import { AcousticThreatMonitor } from '../engine/acousticThreatDetector';
import {
  createInitialSignals,
  EchoSafeSignals,
  EchoSafeSnapshot,
  evaluateEchoSafe,
  PerformanceTelemetry,
  RuntimeMetricsTracker,
  SingleFlightRunner,
} from '../engine/echoSafe';
import { audioScheduler } from '../engine/audioScheduler';

const PROC_W = 640;
const PROC_H = 420;
const COLS = 48;
const ROWS = 16;

type AudioCueStrategy =
  | 'concise_voice'
  | 'low_freq_spatial'
  | 'on_demand'
  | 'continuous_sonification';

const AUDIO_STRATEGIES: { id: AudioCueStrategy; label: string; desc: string }[] = [
  { id: 'concise_voice', label: '简洁语音优先（默认）', desc: '仅播报关键识别变化，减少环境听觉干扰' },
  { id: 'low_freq_spatial', label: '低频空间提示', desc: '检测到近距离目标时播放低频方位提示' },
  { id: 'on_demand', label: '仅按需描述', desc: '平时静音，仅在你主动查询时播报' },
  { id: 'continuous_sonification', label: '实验性连续声场', desc: '持续 3D 声波扫描映射（较耗听觉注意力）' },
];

export const BlindCompanionView: React.FC = () => {
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const acousticMonitorRef = useRef<AcousticThreatMonitor | null>(null);
  const singleFlightRef = useRef(new SingleFlightRunner());
  const metricsTrackerRef = useRef(new RuntimeMetricsTracker());

  const signalsRef = useRef<EchoSafeSignals>(createInitialSignals());
  const [safeSnap, setSafeSnap] = useState<EchoSafeSnapshot>(() =>
    evaluateEchoSafe(createInitialSignals(), performance.now())
  );
  const [perf, setPerf] = useState<PerformanceTelemetry>(() =>
    metricsTrackerRef.current.getSnapshot(performance.now())
  );

  const [audioStrategy, setAudioStrategy] = useState<AudioCueStrategy>('concise_voice');
  const audioStrategyRef = useRef<AudioCueStrategy>('concise_voice');
  audioStrategyRef.current = audioStrategy;
  const [volume, setVolume] = useState(0.5);
  const [finderTarget, setFinderTarget] = useState<{ id: string; label: string } | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [voiceErrorNote, setVoiceErrorNote] = useState<string>('');
  const [assistantMessage, setAssistantMessage] = useState(
    '感知系统未启动。点击「启动实时感知」将申请摄像头权限；本工具为实验性辅助参考，不可替代盲杖与现场判断。'
  );
  const [lastUserSpeech, setLastUserSpeech] = useState<string>('');
  const [detections, setDetections] = useState<SpatialDetection[]>([]);
  const [anomalyNote, setAnomalyNote] = useState<string>('');
  const [lowEdgeDense, setLowEdgeDense] = useState(false);
  const [highEdgeDense, setHighEdgeDense] = useState(false);

  const detectionsRef = useRef<SpatialDetection[]>([]);
  const metricsRef = useRef<SpatialSceneMetrics>({
    leftDensity: 0,
    centerDensity: 0,
    rightDensity: 0,
    edgeCoverage: 0,
    navigationCue: '尚未采集画面',
  });
  const finderRef = useRef<{ id: string; label: string } | null>(null);
  finderRef.current = finderTarget;
  const lowEdgeRef = useRef(false);
  const highEdgeRef = useRef(false);

  const scanColRef = useRef(0);
  const lastScanStepRef = useRef(0);
  const lastFinderPingRef = useRef(0);

  const refreshSafeState = useCallback(() => {
    const now = performance.now();
    const ctx = spatialAudio.ensureContext();
    signalsRef.current.audioState = !ctx
      ? 'unavailable'
      : ctx.state === 'running'
        ? 'running'
        : 'suspended';
    signalsRef.current.pageHidden = typeof document !== 'undefined' ? document.hidden : false;
    const snap = evaluateEchoSafe(signalsRef.current, now);
    setSafeSnap(snap);
    setPerf(metricsTrackerRef.current.getSnapshot(now));
    return snap;
  }, []);

  const say = useCallback(
    (text: string, priority: 'EMERGENCY' | 'USER_QUERY' | 'AMBIENT' = 'USER_QUERY') => {
      setAssistantMessage(text);
      speakCue(text, 1.04, priority);
    },
    []
  );

  const stopAllSensors = useCallback(() => {
    acousticMonitorRef.current?.stop();
    acousticMonitorRef.current = null;
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    signalsRef.current = {
      ...createInitialSignals(),
      modelState: signalsRef.current.modelState === 'ready' ? 'ready' : 'idle',
    };
    detectionsRef.current = [];
    setDetections([]);
    setAnomalyNote('');
    setLowEdgeDense(false);
    setHighEdgeDense(false);
    refreshSafeState();
  }, [refreshSafeState]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      acousticMonitorRef.current?.stop();
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // Page visibility monitoring (browsers suspend camera/audio in background)
  useEffect(() => {
    const onVisibility = () => {
      signalsRef.current.pageHidden = document.hidden;
      if (document.hidden && signalsRef.current.requestedLive) {
        metricsTrackerRef.current.recordMediaInterruption();
        refreshSafeState();
      } else if (!document.hidden && signalsRef.current.requestedLive) {
        // Self-check camera track & audio context upon returning to foreground
        const stream = videoRef.current?.srcObject as MediaStream | null;
        const trackLive = Boolean(stream && stream.getVideoTracks().some((t) => t.readyState === 'live'));
        signalsRef.current.cameraTrackLive = trackLive;
        spatialAudio.ensureContext();
        const snap = refreshSafeState();
        if (!trackLive) {
          say('页面已切回前台，但摄像头视频流已中断，请重新启动实时感知。', 'EMERGENCY');
        } else if (snap.state === 'ACTIVE') {
          say('页面已恢复前台，实时感知自检正常。', 'AMBIENT');
        }
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [refreshSafeState, say]);

  // Periodic health & staleness check (every 1s)
  useEffect(() => {
    const id = window.setInterval(() => {
      if (signalsRef.current.requestedLive || signalsRef.current.initializing) {
        const snap = refreshSafeState();
        if (snap.state === 'UNAVAILABLE' && detectionsRef.current.length > 0) {
          // Immediately clear stale detections when sensor becomes unavailable
          detectionsRef.current = [];
          setDetections([]);
        }
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [refreshSafeState]);

  const toggleCompanion = useCallback(async () => {
    spatialAudio.ensureContext();

    if (signalsRef.current.requestedLive || signalsRef.current.initializing) {
      stopAllSensors();
      say('已停止实时感知。', 'USER_QUERY');
      return;
    }

    signalsRef.current = {
      ...createInitialSignals(),
      requestedLive: true,
      initializing: true,
      modelState: signalsRef.current.modelState === 'ready' ? 'ready' : 'loading',
    };
    refreshSafeState();
    say('正在申请摄像头权限并加载端侧视觉模型，请稍候。', 'USER_QUERY');

    // 1. Request real rear camera (NEVER substitute demo preset on failure)
    if (!navigator.mediaDevices?.getUserMedia) {
      signalsRef.current.initializing = false;
      signalsRef.current.cameraPermission = 'unavailable';
      signalsRef.current.cameraTrackLive = false;
      refreshSafeState();
      say('当前浏览器不支持摄像头采集，实时感知不可用。如需体验模拟场景，请切换至专业实验台。', 'EMERGENCY');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      const track = stream.getVideoTracks()[0];
      if (track) {
        track.onended = () => {
          signalsRef.current.cameraTrackLive = false;
          metricsTrackerRef.current.recordMediaInterruption();
          detectionsRef.current = [];
          setDetections([]);
          refreshSafeState();
          say('摄像头视频流已断开，实时感知不可用，请勿依赖当前结果。', 'EMERGENCY');
        };
      }
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      signalsRef.current.cameraPermission = 'granted';
      signalsRef.current.cameraTrackLive = true;
      signalsRef.current.lastFrameAt = performance.now();
    } catch (err) {
      const isDenied =
        err instanceof DOMException &&
        (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError');
      signalsRef.current.initializing = false;
      signalsRef.current.cameraPermission = isDenied ? 'denied' : 'unavailable';
      signalsRef.current.cameraTrackLive = false;
      refreshSafeState();
      say(
        isDenied
          ? '摄像头权限被拒绝，实时感知不可用。请勿依赖当前结果，如需模拟体验请前往专业实验台。'
          : '未检测到可用摄像头，实时感知不可用。如需模拟体验请前往专业实验台。',
        'EMERGENCY'
      );
      return;
    }

    // 2. Experimental single-channel ambient sound anomaly monitor (optional)
    const monitor = new AcousticThreatMonitor();
    acousticMonitorRef.current = monitor;
    const micOk = await monitor.start((ev) => {
      setAnomalyNote(ev.message);
      say(ev.message, 'EMERGENCY');
      window.setTimeout(() => setAnomalyNote(''), 5500);
    });
    signalsRef.current.microphoneAvailable = micOk;

    // 3. Load on-device AI model (honestly report DEGRADED if load fails)
    const loadRes = await loadOnDeviceAIModel();
    metricsTrackerRef.current.setModelLoadMs(loadRes.loadMs);
    signalsRef.current.modelState = loadRes.modelReady ? 'ready' : 'failed';
    signalsRef.current.initializing = false;

    const snap = refreshSafeState();
    if (snap.state === 'ACTIVE') {
      say(
        '实时感知已启动。本工具仅作辅助参考，未检出物体不代表前方无障碍，请结合盲杖与现场判断。',
        'USER_QUERY'
      );
    } else {
      say(`系统已进入降级状态：${snap.userSummary}`, 'EMERGENCY');
    }
  }, [refreshSafeState, say, stopAllSensors]);

  // Continuous live-camera frame processing loop (strictly live video only)
  useEffect(() => {
    let rafId = 0;

    const loop = (now: number) => {
      if (
        signalsRef.current.requestedLive &&
        !signalsRef.current.initializing &&
        !signalsRef.current.pageHidden &&
        signalsRef.current.cameraTrackLive &&
        videoRef.current &&
        videoRef.current.readyState >= 2
      ) {
        const src = sourceCanvasRef.current;
        const ctx = src?.getContext('2d', { willReadFrequently: true });
        if (src && ctx) {
          ctx.drawImage(videoRef.current, 0, 0, PROC_W, PROC_H);
          signalsRef.current.lastFrameAt = now;
          metricsTrackerRef.current.recordVisionFrame(now);

          const imgData = ctx.getImageData(0, 0, PROC_W, PROC_H);
          const res = processFrameToAcousticMatrix(imgData, COLS, ROWS, 38);
          metricsRef.current = res.metrics;

          // Edge density cues (labeled strictly as visual edge density, not confirmed steps)
          let lowSum = 0;
          let highSum = 0;
          let cells = 0;
          for (let c = 16; c < 32; c++) {
            for (let r = 12; r < 16; r++) {
              lowSum += res.matrix[c][r];
              cells++;
            }
            for (let r = 0; r < 4; r++) {
              highSum += res.matrix[c][r];
            }
          }
          const lDense = lowSum / Math.max(1, cells) > 0.45;
          const hDense = highSum / Math.max(1, cells) > 0.48;
          lowEdgeRef.current = lDense;
          highEdgeRef.current = hDense;
          setLowEdgeDense(lDense);
          setHighEdgeDense(hDense);

          if (
            audioStrategyRef.current === 'continuous_sonification' &&
            volume > 0.05 &&
            now - lastScanStepRef.current >= 80
          ) {
            lastScanStepRef.current = now;
            const col = scanColRef.current;
            spatialAudio.playScanColumn(res.matrix[col], col / (COLS - 1), 0.07, {
              scaleMode: 'pentatonic',
              useHRTF: true,
              volume: volume * 0.4,
            });
            scanColRef.current = (col + 1) % COLS;
          }
        }
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [volume]);

  // Single-flight AI inference loop (only runs when live camera & model are ready)
  useEffect(() => {
    let cancelled = false;

    const timer = window.setInterval(async () => {
      if (cancelled) return;
      if (
        !signalsRef.current.requestedLive ||
        signalsRef.current.initializing ||
        signalsRef.current.pageHidden ||
        !signalsRef.current.cameraTrackLive ||
        signalsRef.current.modelState !== 'ready'
      ) {
        return;
      }

      await singleFlightRef.current.run(async () => {
        const src = sourceCanvasRef.current;
        if (!src || cancelled) return;
        const t0 = performance.now();
        try {
          // CRITICAL: allowDemoFallback is strictly FALSE in live companion mode
          const res = await detectSpatialObjects(src, false, 'crosswalk');
          const duration = performance.now() - t0;
          if (cancelled) return;

          signalsRef.current.lastInferenceAt = performance.now();
          signalsRef.current.lastInferenceDurationMs = duration;
          signalsRef.current.consecutiveInferenceFailures = 0;
          metricsTrackerRef.current.recordInference(performance.now(), duration);

          detectionsRef.current = res;
          setDetections(res);
          refreshSafeState();

          // 3D Item-Finder beacon if user set a target
          const target = finderRef.current;
          if (target) {
            const hit = res.find(
              (d) => d.labelEn === target.id || (target.id === 'cup' && d.labelEn === 'bottle')
            );
            if (hit) {
              const [bx, , bw] = hit.bbox;
              const normX = Math.max(0, Math.min(1, (bx + bw / 2) / PROC_W));
              const centeredness = 1 - Math.abs(normX - 0.5) * 2;
              const closeness = Math.min(1, centeredness * 0.65 + (hit.distance === '近距离' ? 0.35 : 0.1));
              const now = performance.now();
              if (now - lastFinderPingRef.current > 800) {
                lastFinderPingRef.current = now;
                spatialAudio.playFinderBeacon(normX, closeness, Math.max(0.5, volume));
              }
            }
          }

          // Respect selected audio cue strategy (default: concise_voice, never nag)
          if (res.length > 0 && audioStrategyRef.current !== 'on_demand') {
            if (audioStrategyRef.current === 'low_freq_spatial') {
              const first = res[0];
              if (first.distance === '近距离') {
                const [bx, , bw] = first.bbox;
                const normX = Math.max(0, Math.min(1, (bx + bw / 2) / PROC_W));
                spatialAudio.playBrushTone(normX, 0.88, {
                  scaleMode: 'pentatonic',
                  useHRTF: true,
                  volume: volume * 0.65,
                });
              }
            }
            const spoken = speakSpatialDetections(res, false);
            if (spoken) setAssistantMessage(spoken);
          }
        } catch {
          signalsRef.current.consecutiveInferenceFailures += 1;
          refreshSafeState();
        }
      });
    }, 900);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [refreshSafeState, volume]);

  const handleUtterance = useCallback(
    (utterance: string) => {
      setLastUserSpeech(utterance);
      const snap = refreshSafeState();
      const src = sourceCanvasRef.current;
      const reply = respondToUserUtterance(utterance, {
        safeState: snap,
        detections: detectionsRef.current,
        metrics: metricsRef.current,
        lowRegionEdgeDense: lowEdgeRef.current,
        highRegionEdgeDense: highEdgeRef.current,
      });

      if (reply.action.type === 'set_finder') {
        setFinderTarget({ id: reply.action.targetId, label: reply.action.targetLabel });
      } else if (reply.action.type === 'clear_finder') {
        setFinderTarget(null);
      } else if (reply.action.type === 'check_light_color') {
        if (!snap.canTrustLivePerception || !src) {
          say('当前没有实时摄像头画面，无法测量环境光线与颜色。', 'USER_QUERY');
          return;
        }
        const light = analyzeLightAndColor(src);
        say(light.speechText, 'USER_QUERY');
        return;
      } else if (reply.action.type === 'set_scan') {
        setAudioStrategy(reply.action.enabled ? 'continuous_sonification' : 'concise_voice');
      } else if (reply.action.type === 'set_volume') {
        const d = reply.action.delta;
        if (d === 'mute') {
          setVolume(0);
          audioScheduler.setMuted(true);
        } else if (d === 'unmute') {
          audioScheduler.setMuted(false);
          setVolume(0.55);
        } else {
          setVolume((v) => Math.max(0.1, Math.min(1, v + d)));
        }
      } else if (reply.action.type === 'pause_all') {
        stopAllSensors();
      } else if (reply.action.type === 'resume_all') {
        if (!signalsRef.current.requestedLive) toggleCompanion();
        return;
      }

      say(reply.text, reply.mood === 'alert' ? 'EMERGENCY' : 'USER_QUERY');
    },
    [refreshSafeState, say, stopAllSensors, toggleCompanion]
  );

  const startTalking = useCallback(() => {
    spatialAudio.ensureContext();
    setVoiceErrorNote('');
    if (typeof window === 'undefined') return;
    type SpeechRecEvent = { error?: string };
    type SpeechRecCtor = new () => {
      lang: string;
      interimResults: boolean;
      maxAlternatives: number;
      onstart: (() => void) | null;
      onend: (() => void) | null;
      onerror: ((ev: SpeechRecEvent) => void) | null;
      onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
      start: () => void;
      abort?: () => void;
    };
    const w = window as unknown as {
      SpeechRecognition?: SpeechRecCtor;
      webkitSpeechRecognition?: SpeechRecCtor;
    };
    const SpeechRec = w.SpeechRecognition || w.webkitSpeechRecognition;

    // P0-5: NEVER silently return a scene report pretending speech was heard!
    if (!SpeechRec) {
      const msg = '当前浏览器不支持语音识别，无法听取你的提问。请使用下方「查询当前识别状态」等快捷按钮操作。';
      setVoiceErrorNote(msg);
      say(msg, 'USER_QUERY');
      return;
    }

    const rec = new SpeechRec();
    rec.lang = 'zh-CN';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    let gotResult = false;

    rec.onstart = () => setIsListening(true);
    rec.onend = () => {
      setIsListening(false);
      if (!gotResult) {
        const msg = '未检测到有效语音输入，你可以重新点击语音按钮或使用下方快捷查询按钮。';
        setVoiceErrorNote(msg);
      }
    };
    rec.onerror = (ev) => {
      setIsListening(false);
      gotResult = true;
      const code = ev?.error || 'unknown';
      let msg = '语音识别服务异常，请重试或使用下方快捷按钮。';
      if (code === 'not-allowed' || code === 'service-not-allowed') {
        msg = '麦克风或语音识别权限被拒绝，请在浏览器设置中允许麦克风权限，或改用下方快捷按钮。';
      } else if (code === 'no-speech') {
        msg = '没有听到语音输入，请靠近麦克风重试或使用下方快捷按钮。';
      } else if (code === 'network') {
        msg = '语音识别网络连接失败（浏览器语音识别通常依赖云端服务），请改用下方快捷按钮。';
      }
      setVoiceErrorNote(msg);
      say(msg, 'USER_QUERY');
    };
    rec.onresult = (ev) => {
      gotResult = true;
      const transcript = ev.results?.[0]?.[0]?.transcript || '';
      if (!transcript.trim()) {
        const msg = '语音转写结果为空，请重试或使用下方快捷按钮。';
        setVoiceErrorNote(msg);
        say(msg, 'USER_QUERY');
        return;
      }
      handleUtterance(transcript);
    };
    try {
      rec.start();
    } catch {
      setIsListening(false);
      const msg = '无法启动语音识别，请稍后重试或使用下方快捷按钮。';
      setVoiceErrorNote(msg);
      say(msg, 'USER_QUERY');
    }
  }, [handleUtterance, say]);

  const isRunning = safeSnap.state === 'ACTIVE' || safeSnap.state === 'DEGRADED';

  return (
    <div className="mx-auto max-w-xl space-y-5">
      <canvas ref={sourceCanvasRef} width={PROC_W} height={PROC_H} className="hidden" />

      {/* Explicit Safety Boundary & EchoSafe Status Header */}
      <section className="panel p-5 space-y-4" aria-label="随身助行主控面板">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span
              className={`led ${
                safeSnap.state === 'ACTIVE'
                  ? 'led-ok'
                  : safeSnap.state === 'DEGRADED' || safeSnap.state === 'UNAVAILABLE'
                    ? 'led-on'
                    : ''
              }`}
              aria-hidden="true"
            />
            <div>
              <div className="eyebrow">EchoSafe 状态 · {safeSnap.state}</div>
              <div className="text-[13px] font-semibold text-ink">{safeSnap.userSummary}</div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() =>
                setAudioStrategy((prev) =>
                  prev === 'on_demand' ? 'concise_voice' : 'on_demand'
                )
              }
              aria-pressed={audioStrategy === 'on_demand'}
              className="btn btn-sm"
              title="切换主动播报或仅按需播报，避免干扰环境原声"
            >
              {audioStrategy === 'on_demand' ? (
                <VolumeX className="h-3.5 w-3.5" />
              ) : (
                <Volume2 className="h-3.5 w-3.5" />
              )}
              {audioStrategy === 'on_demand' ? '按需静音中' : '主动提示开'}
            </button>
            <button
              type="button"
              onClick={toggleCompanion}
              disabled={safeSnap.state === 'INITIALIZING'}
              className={`btn btn-sm ${isRunning ? 'btn-signal' : 'btn-primary'}`}
            >
              <Power className="h-3.5 w-3.5" />
              {safeSnap.state === 'INITIALIZING'
                ? '初始化中...'
                : isRunning
                  ? '停止实时感知'
                  : '启动实时感知'}
            </button>
          </div>
        </div>

        {/* Camera Viewport (Live only — NEVER draws demo preset) */}
        <div className="screen relative h-[210px] sm:h-[230px]">
          <video
            ref={videoRef}
            playsInline
            muted
            className={`w-full h-full object-cover ${
              safeSnap.dataSource === 'live_camera' ? 'block' : 'hidden'
            }`}
          />
          {safeSnap.dataSource !== 'live_camera' && (
            <div className="w-full h-full flex flex-col items-center justify-center px-6 text-center space-y-2">
              <AlertTriangle className="h-6 w-6 text-amber" aria-hidden="true" />
              <div className="text-[13.5px] font-medium text-phosphor">
                {safeSnap.state === 'UNAVAILABLE'
                  ? safeSnap.userSummary
                  : '当前无实时摄像头输入（实时模式严禁混入模拟画面）'}
              </div>
              <div className="text-[12px] text-phosphor/65">
                如需无摄像头演示，请切换顶部「专业实验台」查看明确标注的模拟场景。
              </div>
            </div>
          )}

          {isRunning && (
            <div className="pointer-events-none absolute top-2.5 left-2.5 right-2.5 flex flex-wrap justify-between gap-1.5">
              <span className="screen-label">
                真实摄像头 · 视觉 {perf.visionFps}fps · AI {perf.aiFps}fps
                {perf.lastInferenceMs !== null ? ` · ${perf.lastInferenceMs}ms` : ''}
              </span>
              {finderTarget && (
                <span className="screen-label text-amber">寻物目标：{finderTarget.label}</span>
              )}
            </div>
          )}

          {(anomalyNote || lowEdgeDense || highEdgeDense) && isRunning && (
            <div className="pointer-events-none absolute bottom-2.5 left-2.5 right-2.5 flex flex-wrap gap-1.5">
              {anomalyNote && <span className="screen-label text-amber">实验性环境声响提示</span>}
              {lowEdgeDense && <span className="screen-label">下方边缘轮廓较密（物体未知）</span>}
              {highEdgeDense && <span className="screen-label">上方边缘轮廓较密（物体未知）</span>}
            </div>
          )}
        </div>

        {/* Live accessible status readout */}
        <div
          className="panel-sunken p-4 border-l-2 border-l-signal space-y-1.5"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <div className="flex items-center justify-between text-[11.5px] text-ink-3">
            <span className="eyebrow text-signal-ink">最近一次有效播报</span>
            {lastUserSpeech && <span>你的询问：「{lastUserSpeech}」</span>}
          </div>
          <p className="title-serif text-[17px] sm:text-[18px] leading-relaxed text-ink">
            {assistantMessage}
          </p>
        </div>

        {/* Confirmed detections list (no uncalibrated decimal claims) */}
        {safeSnap.state === 'ACTIVE' && detections.length > 0 && (
          <div className="flex flex-wrap gap-2" aria-label="当前识别到的物体">
            {detections.slice(0, 4).map((d, i) => (
              <span key={i} className="chip">
                <strong className="text-signal-ink">{d.direction}</strong>
                <span className="num text-ink">{d.distanceText}</span>
                <span>{d.labelZh}</span>
              </span>
            ))}
          </div>
        )}
      </section>

      {/* Voice Query & Quick Controls */}
      <section className="panel p-5 space-y-4" aria-label="语音查询与控制">
        <button
          type="button"
          onClick={startTalking}
          className={`w-full min-h-[84px] rounded-xl border-2 flex flex-col items-center justify-center gap-1.5 transition-colors ${
            isListening
              ? 'border-signal bg-signal text-white'
              : 'border-ink bg-ink text-paper hover:bg-ink/90'
          }`}
        >
          {isListening ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
          <span className="text-[16px] font-semibold text-inherit">
            {isListening ? '正在聆听语音指令…' : '语音查询与操控'}
          </span>
          <span className="text-[12px] opacity-80">
            查询识别结果 · 设定寻物目标 · 测光识色 · 控制音量与暂停
          </span>
        </button>

        {voiceErrorNote && (
          <div
            role="alert"
            className="rounded-lg border border-warn/40 bg-warn-soft px-3.5 py-2 text-[12.5px] text-warn"
          >
            {voiceErrorNote}
          </div>
        )}

        <div className="flex flex-wrap items-center justify-center gap-2">
          {[
            '前面识别到什么？',
            '帮我找椅子',
            '开灯了吗？什么颜色？',
            '关掉声波',
          ].map((phrase) => (
            <button
              key={phrase}
              type="button"
              onClick={() => handleUtterance(phrase)}
              className="btn btn-sm"
            >
              {phrase}
            </button>
          ))}
        </div>

        {/* Selectable Audio Cue Strategy (Task D: respect diverse listening preferences) */}
        <fieldset className="space-y-2 border-t border-line pt-3">
          <legend className="eyebrow">听觉提示策略（默认优先保护环境听觉）</legend>
          <div className="grid grid-cols-2 gap-2">
            {AUDIO_STRATEGIES.map((st) => (
              <button
                key={st.id}
                type="button"
                data-active={audioStrategy === st.id}
                aria-pressed={audioStrategy === st.id}
                onClick={() => {
                  setAudioStrategy(st.id);
                  say(`已切换听觉策略：${st.label}`, 'USER_QUERY');
                }}
                className="tile"
              >
                <div className="text-[12.5px] font-semibold text-ink">{st.label}</div>
                <div className="text-[11px] text-ink-3 mt-0.5">{st.desc}</div>
              </button>
            ))}
          </div>
        </fieldset>

        <p className="text-[11.5px] text-ink-3 leading-relaxed border-t border-line pt-3">
          安全与隐私说明：本系统为浏览器端辅助感知原型，不可替代白手杖、导盲犬或现场交通安全判断；
          视觉与音频在本地计算，但浏览器内置语音识别（Web Speech Recognition）在多数设备上会通过浏览器厂商云服务完成转写。
        </p>
      </section>
    </div>
  );
};
