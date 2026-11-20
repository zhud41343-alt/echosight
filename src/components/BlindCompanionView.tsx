import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Power, Volume2, VolumeX, AlertTriangle } from 'lucide-react';
import {
  processFrameToAcousticMatrix,
  SpatialSceneMetrics,
} from '../engine/visionProcessor';
import {
  DetectionStabilizer,
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
  isPerceptionPacketUsable,
  PerceptionPacket,
  PerformanceTelemetry,
  RuntimeMetricsTracker,
  SessionGuard,
  SingleFlightRunner,
} from '../engine/echoSafe';
import { audioScheduler } from '../engine/audioScheduler';

const MAX_PROC_DIM = 640;
const COLS = 48;
const ROWS = 16;
const PACKET_TTL_MS = 4000;

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

type VideoWithVfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number) => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

export const BlindCompanionView: React.FC = () => {
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const acousticMonitorRef = useRef<AcousticThreatMonitor | null>(null);
  const sessionGuardRef = useRef(new SessionGuard());
  const singleFlightRef = useRef(new SingleFlightRunner());
  const metricsTrackerRef = useRef(new RuntimeMetricsTracker());
  const stabilizerRef = useRef(new DetectionStabilizer());
  const procSizeRef = useRef<{ w: number; h: number }>({ w: 640, h: 480 });

  const signalsRef = useRef<EchoSafeSignals>(createInitialSignals());
  const [safeSnap, setSafeSnap] = useState<EchoSafeSnapshot>(() =>
    evaluateEchoSafe(createInitialSignals(), performance.now())
  );
  const [perf, setPerf] = useState<PerformanceTelemetry>(() =>
    metricsTrackerRef.current.getSnapshot(performance.now())
  );

  // P0-4: Single unified perception packet (never trust stale raw arrays)
  const packetRef = useRef<PerceptionPacket<SpatialDetection> | null>(null);
  const [detections, setDetections] = useState<SpatialDetection[]>([]);

  const [audioStrategy, setAudioStrategy] = useState<AudioCueStrategy>('concise_voice');
  const audioStrategyRef = useRef<AudioCueStrategy>('concise_voice');
  audioStrategyRef.current = audioStrategy;
  const [volume, setVolume] = useState(0.5);
  const [finderTarget, setFinderTarget] = useState<{ id: string; label: string } | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [voiceErrorNote, setVoiceErrorNote] = useState<string>('');
  const [assistantMessage, setAssistantMessage] = useState(
    '感知系统尚未启动。点击「启动实时感知」将申请摄像头权限；本工具为实验性辅助参考，不可替代盲杖与现场判断。'
  );
  const [lastUserSpeech, setLastUserSpeech] = useState<string>('');
  const [anomalyNote, setAnomalyNote] = useState<string>('');
  const [lowEdgeDense, setLowEdgeDense] = useState(false);
  const [highEdgeDense, setHighEdgeDense] = useState(false);

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

  // Real frame advancement tracking (P0-1)
  const lastVideoCurrentTimeRef = useRef<number>(-1);
  const vfcHandleRef = useRef<number | null>(null);
  const scanColRef = useRef(0);

  const getUsableDetections = useCallback((now: number): SpatialDetection[] => {
    const pkt = packetRef.current;
    if (!isPerceptionPacketUsable(pkt, sessionGuardRef.current.currentId, now)) {
      return [];
    }
    return pkt!.detections;
  }, []);

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

    if (!snap.canTrustLivePerception) {
      if (packetRef.current !== null) {
        packetRef.current = null;
        setDetections([]);
      }
    }
    return snap;
  }, []);

  const say = useCallback(
    (text: string, priority: 'EMERGENCY' | 'USER_QUERY' | 'AMBIENT' = 'USER_QUERY') => {
      setAssistantMessage(text);
      speakCue(text, 1.04, priority);
    },
    []
  );

  const stopVideoFrameWatcher = useCallback(() => {
    const vid = videoRef.current as VideoWithVfc | null;
    if (vid && vfcHandleRef.current !== null && typeof vid.cancelVideoFrameCallback === 'function') {
      try {
        vid.cancelVideoFrameCallback(vfcHandleRef.current);
      } catch {
        // Ignore
      }
    }
    vfcHandleRef.current = null;
    lastVideoCurrentTimeRef.current = -1;
  }, []);

  const stopAllSensors = useCallback(() => {
    // P0-2: Invalidate session FIRST so all in-flight async tasks immediately drop their results
    sessionGuardRef.current.invalidate();
    stopVideoFrameWatcher();
    audioScheduler.clearAll();
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
    packetRef.current = null;
    stabilizerRef.current.reset();
    setDetections([]);
    const ov = overlayCanvasRef.current;
    if (ov) {
      const oCtx = ov.getContext('2d');
      oCtx?.clearRect(0, 0, ov.width, ov.height);
    }
    setFinderTarget(null);
    setAnomalyNote('');
    setLowEdgeDense(false);
    setHighEdgeDense(false);
    refreshSafeState();
  }, [refreshSafeState, stopVideoFrameWatcher]);

  // Cleanup on unmount (invalidates session so no late callbacks can touch state)
  useEffect(() => {
    return () => {
      sessionGuardRef.current.invalidate();
      stopVideoFrameWatcher();
      audioScheduler.clearAll();
      acousticMonitorRef.current?.stop();
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
    };
  }, [stopVideoFrameWatcher]);

  // Page visibility monitoring (clear stale packet & require fresh frames upon return)
  useEffect(() => {
    const onVisibility = () => {
      signalsRef.current.pageHidden = document.hidden;
      if (document.hidden && signalsRef.current.requestedLive) {
        metricsTrackerRef.current.recordMediaInterruption();
        packetRef.current = null;
        setDetections([]);
        audioScheduler.clearAll();
        refreshSafeState();
      } else if (!document.hidden && signalsRef.current.requestedLive) {
        const stream = videoRef.current?.srcObject as MediaStream | null;
        const trackLive = Boolean(stream && stream.getVideoTracks().some((t) => t.readyState === 'live'));
        signalsRef.current.cameraTrackLive = trackLive;
        // Require fresh frame & fresh inference after returning from background
        signalsRef.current.lastFrameAt = null;
        signalsRef.current.lastInferenceAt = null;
        signalsRef.current.sessionStartedAt = performance.now();
        packetRef.current = null;
        setDetections([]);
        spatialAudio.ensureContext();
        refreshSafeState();
        if (!trackLive) {
          say('页面已切回前台，但摄像头视频流已中断，请重新启动实时感知。', 'EMERGENCY');
        } else {
          say('页面已恢复前台，正在重新校验摄像头画面。', 'AMBIENT');
        }
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [refreshSafeState, say]);

  // Periodic health & packet TTL check (every 1s)
  useEffect(() => {
    const id = window.setInterval(() => {
      if (signalsRef.current.requestedLive || signalsRef.current.initializing) {
        refreshSafeState();
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [refreshSafeState]);

  // Attach requestVideoFrameCallback watcher when available to detect genuine frame advances
  const attachVideoFrameWatcher = useCallback(
    (mySession: number) => {
      const vid = videoRef.current as VideoWithVfc | null;
      if (!vid || typeof vid.requestVideoFrameCallback !== 'function') return;

      const onFrame = (now: number) => {
        if (!sessionGuardRef.current.isValid(mySession)) return;
        if (signalsRef.current.cameraTrackLive && !signalsRef.current.pageHidden) {
          signalsRef.current.lastFrameAt = now;
          metricsTrackerRef.current.recordVisionFrame(now);
        }
        vfcHandleRef.current = vid.requestVideoFrameCallback!(onFrame);
      };
      vfcHandleRef.current = vid.requestVideoFrameCallback(onFrame);
    },
    []
  );

  const toggleCompanion = useCallback(async () => {
    spatialAudio.ensureContext();

    if (signalsRef.current.requestedLive || signalsRef.current.initializing) {
      stopAllSensors();
      say('已停止实时感知。', 'USER_QUERY');
      return;
    }

    const mySession = sessionGuardRef.current.begin();
    const startTs = performance.now();
    signalsRef.current = {
      ...createInitialSignals(),
      requestedLive: true,
      initializing: true,
      sessionStartedAt: startTs,
      modelState: signalsRef.current.modelState === 'ready' ? 'ready' : 'loading',
    };
    refreshSafeState();
    say('正在申请摄像头权限并加载端侧模型，请稍候。', 'USER_QUERY');

    if (!navigator.mediaDevices?.getUserMedia) {
      if (!sessionGuardRef.current.isValid(mySession)) return;
      signalsRef.current.initializing = false;
      signalsRef.current.cameraPermission = 'unavailable';
      signalsRef.current.cameraTrackLive = false;
      refreshSafeState();
      say('当前浏览器不支持摄像头采集，实时感知不可用。如需模拟演示请前往专业实验台。', 'EMERGENCY');
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

      // P0-2: If user cancelled/stopped while waiting for camera permission, release tracks immediately!
      if (!sessionGuardRef.current.isValid(mySession)) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      const track = stream.getVideoTracks()[0];
      if (track) {
        track.onended = () => {
          if (!sessionGuardRef.current.isValid(mySession)) return;
          signalsRef.current.cameraTrackLive = false;
          metricsTrackerRef.current.recordMediaInterruption();
          packetRef.current = null;
          setDetections([]);
          refreshSafeState();
          say('摄像头视频流已断开，实时感知不可用，请勿依赖当前结果。', 'EMERGENCY');
        };
      }
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      if (!sessionGuardRef.current.isValid(mySession)) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      signalsRef.current.cameraPermission = 'granted';
      signalsRef.current.cameraTrackLive = true;
      // P0-1: Do NOT fake lastFrameAt here! Wait for genuine video frame arrival.
      signalsRef.current.lastFrameAt = null;
      attachVideoFrameWatcher(mySession);
    } catch (err) {
      if (!sessionGuardRef.current.isValid(mySession)) return;
      const isDenied =
        err instanceof DOMException &&
        (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError');
      signalsRef.current.initializing = false;
      signalsRef.current.cameraPermission = isDenied ? 'denied' : 'unavailable';
      signalsRef.current.cameraTrackLive = false;
      refreshSafeState();
      say(
        isDenied
          ? '摄像头权限被拒绝，实时感知不可用。请勿依赖当前结果，如需模拟演示请前往专业实验台。'
          : '未检测到可用摄像头，实时感知不可用。如需模拟演示请前往专业实验台。',
        'EMERGENCY'
      );
      return;
    }

    // Optional experimental single-channel ambient sound anomaly monitor
    const monitor = new AcousticThreatMonitor();
    const micOk = await monitor.start((ev) => {
      if (!sessionGuardRef.current.isValid(mySession)) return;
      setAnomalyNote(ev.message);
      say(ev.message, 'EMERGENCY');
      window.setTimeout(() => {
        if (sessionGuardRef.current.isValid(mySession)) setAnomalyNote('');
      }, 5500);
    });
    if (!sessionGuardRef.current.isValid(mySession)) {
      monitor.stop();
      return;
    }
    acousticMonitorRef.current = monitor;
    signalsRef.current.microphoneAvailable = micOk;

    // Load on-device AI model
    const loadRes = await loadOnDeviceAIModel();
    if (!sessionGuardRef.current.isValid(mySession)) return;

    metricsTrackerRef.current.setModelLoadMs(loadRes.loadMs);
    signalsRef.current.modelState = loadRes.modelReady ? 'ready' : 'failed';
    signalsRef.current.initializing = false;

    const snap = refreshSafeState();
    if (loadRes.modelReady) {
      say(
        '摄像头与视觉模型已就绪，正在完成首次识别。本工具仅作辅助参考，方位相对当前摄像头画面，不可替代盲杖。',
        'USER_QUERY'
      );
    } else {
      say(`系统进入降级状态：${snap.userSummary}`, 'EMERGENCY');
    }
  }, [attachVideoFrameWatcher, refreshSafeState, say, stopAllSensors]);

  // Continuous frame processing loop (only updates lastFrameAt when video.currentTime genuinely advances!)
  useEffect(() => {
    let rafId = 0;

    const loop = (now: number) => {
      const vid = videoRef.current as VideoWithVfc | null;
      if (
        sessionGuardRef.current.isActive &&
        signalsRef.current.requestedLive &&
        !signalsRef.current.initializing &&
        !signalsRef.current.pageHidden &&
        signalsRef.current.cameraTrackLive &&
        vid &&
        vid.readyState >= 2
      ) {
        const hasVfc = typeof vid.requestVideoFrameCallback === 'function';
        const frameAdvanced = hasVfc
          ? signalsRef.current.lastFrameAt !== null && now - signalsRef.current.lastFrameAt < 120
          : vid.currentTime !== lastVideoCurrentTimeRef.current;

        if (!hasVfc && vid.currentTime !== lastVideoCurrentTimeRef.current) {
          lastVideoCurrentTimeRef.current = vid.currentTime;
          signalsRef.current.lastFrameAt = now;
          metricsTrackerRef.current.recordVisionFrame(now);
        }

        const src = sourceCanvasRef.current;
        const ctx = src?.getContext('2d', { willReadFrequently: true });
        if (src && ctx && frameAdvanced) {
          // Adapt canvas to the video's true native aspect ratio (zero portrait squashing!)
          const vw = vid.videoWidth || 640;
          const vh = vid.videoHeight || 480;
          const scale = Math.min(1, MAX_PROC_DIM / Math.max(vw, vh));
          const targetW = Math.max(160, Math.round(vw * scale));
          const targetH = Math.max(160, Math.round(vh * scale));
          if (src.width !== targetW || src.height !== targetH) {
            src.width = targetW;
            src.height = targetH;
            procSizeRef.current = { w: targetW, h: targetH };
          }
          ctx.drawImage(vid, 0, 0, targetW, targetH);
          const imgData = ctx.getImageData(0, 0, targetW, targetH);
          const res = processFrameToAcousticMatrix(imgData, COLS, ROWS, 38);
          metricsRef.current = res.metrics;

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
            audioScheduler.shouldAllowSpatialCue('continuous', now)
          ) {
            const col = scanColRef.current;
            spatialAudio.playScanColumn(res.matrix[col], col / (COLS - 1), 0.07, {
              scaleMode: 'pentatonic',
              useHRTF: true,
              volume: volume * 0.38,
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

  // Single-flight AI inference loop with strict sessionGuard & PerceptionPacket stamping
  useEffect(() => {
    let cancelled = false;

    const timer = window.setInterval(async () => {
      if (cancelled || !sessionGuardRef.current.isActive) return;
      if (
        !signalsRef.current.requestedLive ||
        signalsRef.current.initializing ||
        signalsRef.current.pageHidden ||
        !signalsRef.current.cameraTrackLive ||
        signalsRef.current.modelState !== 'ready' ||
        signalsRef.current.lastFrameAt === null
      ) {
        return;
      }

      const mySession = sessionGuardRef.current.currentId;
      const frameCapturedAt = signalsRef.current.lastFrameAt;

      await singleFlightRef.current.run(async () => {
        const src = sourceCanvasRef.current;
        if (!src || cancelled || !sessionGuardRef.current.isValid(mySession)) return;
        const t0 = performance.now();
        try {
          const rawRes = await detectSpatialObjects(src, false, 'crosswalk');
          const res = stabilizerRef.current.filter(rawRes);
          const tEnd = performance.now();
          const duration = tEnd - t0;

          // P0-2: Drop late results if session was stopped/restarted during async inference
          if (cancelled || !sessionGuardRef.current.isValid(mySession)) return;

          signalsRef.current.lastInferenceAt = tEnd;
          signalsRef.current.lastInferenceDurationMs = duration;
          signalsRef.current.consecutiveInferenceFailures = 0;
          metricsTrackerRef.current.recordInference(tEnd, duration, tEnd - frameCapturedAt);

          const snap = refreshSafeState();
          if (snap.canTrustLivePerception) {
            packetRef.current = {
              dataSource: 'REAL_CAMERA',
              sessionId: mySession,
              capturedAt: frameCapturedAt,
              inferredAt: tEnd,
              expiresAt: tEnd + PACKET_TTL_MS,
              safeState: snap.state,
              detections: res,
            };
            setDetections(res);

            // Render live bounding boxes aligned to the native video aspect ratio
            const ov = overlayCanvasRef.current;
            const { w: pw, h: ph } = procSizeRef.current;
            if (ov) {
              if (ov.width !== pw || ov.height !== ph) {
                ov.width = pw;
                ov.height = ph;
              }
              const oCtx = ov.getContext('2d');
              if (oCtx) {
                oCtx.clearRect(0, 0, pw, ph);
                res.forEach((det) => {
                  const [bx, by, bw, bh] = det.bbox;
                  const color = det.distance === '近距离' ? '#ff8a3d' : '#9fc9b0';
                  oCtx.strokeStyle = color;
                  oCtx.lineWidth = 2.4;
                  oCtx.strokeRect(bx, by, bw, bh);

                  const tag = `${det.direction} · ${det.labelZh} ${Math.round(det.score * 100)}%`;
                  oCtx.font = '600 13px "JetBrains Mono", Consolas, sans-serif';
                  const tw = oCtx.measureText(tag).width + 12;
                  const tx = Math.max(4, Math.min(pw - tw - 4, bx));
                  const ty = Math.max(22, by);
                  oCtx.fillStyle = 'rgba(22, 21, 18, 0.88)';
                  oCtx.fillRect(tx, ty - 22, tw, 22);
                  oCtx.fillStyle = color;
                  oCtx.fillText(tag, tx + 6, ty - 6);
                });
              }
            }

            // Unified validity check before any audio output (P0-4)
            const usable = getUsableDetections(tEnd);

            const target = finderRef.current;
            if (target) {
              const hit = usable.find(
                (d) => d.labelEn === target.id || (target.id === 'cup' && d.labelEn === 'bottle')
              );
              if (hit && audioScheduler.shouldAllowSpatialCue('finder', tEnd)) {
                const [bx, , bw] = hit.bbox;
                const normX = Math.max(0, Math.min(1, (bx + bw / 2) / pw));
                const centeredness = 1 - Math.abs(normX - 0.5) * 2;
                const closeness = Math.min(
                  1,
                  centeredness * 0.65 + (hit.distance === '近距离' ? 0.35 : 0.1)
                );
                spatialAudio.playFinderBeacon(normX, closeness, Math.max(0.5, volume));
              }
            }

            if (usable.length > 0 && audioStrategyRef.current !== 'on_demand') {
              if (
                audioStrategyRef.current === 'low_freq_spatial' &&
                usable[0].distance === '近距离' &&
                audioScheduler.shouldAllowSpatialCue('low_freq', tEnd)
              ) {
                const [bx, , bw] = usable[0].bbox;
                const normX = Math.max(0, Math.min(1, (bx + bw / 2) / pw));
                spatialAudio.playBrushTone(normX, 0.88, {
                  scaleMode: 'pentatonic',
                  useHRTF: true,
                  volume: volume * 0.6,
                });
              }
              const spoken = speakSpatialDetections(usable, false);
              if (spoken) setAssistantMessage(spoken);
            }
          }
        } catch {
          if (!sessionGuardRef.current.isValid(mySession)) return;
          signalsRef.current.consecutiveInferenceFailures += 1;
          refreshSafeState();
        }
      });
    }, 600);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [getUsableDetections, refreshSafeState, volume]);

  const handleUtterance = useCallback(
    (utterance: string) => {
      setLastUserSpeech(utterance);
      const now = performance.now();
      const snap = refreshSafeState();
      const usableDetections = getUsableDetections(now);
      const src = sourceCanvasRef.current;

      const reply = respondToUserUtterance(utterance, {
        safeState: snap,
        detections: usableDetections,
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
          say('当前没有实时有效摄像头画面，无法测量环境光线与颜色。', 'USER_QUERY');
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
    [getUsableDetections, refreshSafeState, say, stopAllSensors, toggleCompanion]
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
    };
    const w = window as unknown as {
      SpeechRecognition?: SpeechRecCtor;
      webkitSpeechRecognition?: SpeechRecCtor;
    };
    const SpeechRec = w.SpeechRecognition || w.webkitSpeechRecognition;

    if (!SpeechRec) {
      const msg =
        '当前浏览器不支持语音识别，无法听取你的提问。请使用下方「前面识别到什么」等快捷按钮操作。';
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
        setVoiceErrorNote('未检测到有效语音输入，你可以重新点击语音按钮或改用下方快捷查询按钮。');
      }
    };
    rec.onerror = (ev) => {
      setIsListening(false);
      gotResult = true;
      const code = ev?.error || 'unknown';
      let msg = '语音识别服务异常，请重试或改用下方快捷按钮。';
      if (code === 'not-allowed' || code === 'service-not-allowed') {
        msg = '麦克风或语音识别权限被拒绝，请在浏览器设置中允许权限，或改用下方快捷按钮。';
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
    <div className="mx-auto max-w-2xl space-y-5">
      <canvas ref={sourceCanvasRef} width={640} height={480} className="hidden" />

      {/* Main Accessible Status & Primary Control */}
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
              <div className="eyebrow">系统状态</div>
              <div className="text-[15px] font-semibold text-ink">{safeSnap.stateLabelZh}</div>
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
              className={`btn btn-sm ${
                isRunning || safeSnap.state === 'INITIALIZING' ? 'btn-signal' : 'btn-primary'
              }`}
            >
              <Power className="h-3.5 w-3.5" />
              {safeSnap.state === 'INITIALIZING'
                ? '取消准备'
                : isRunning
                  ? '停止实时感知'
                  : '启动实时感知'}
            </button>
          </div>
        </div>

        {/* Live accessible speech readout */}
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

        {/* Camera Viewport (Full uncropped aspect ratio + live bounding-box overlay) */}
        <div className="screen relative h-[320px] sm:h-[400px] flex items-center justify-center">
          <video
            ref={videoRef}
            playsInline
            muted
            className={`w-full h-full object-contain ${
              safeSnap.dataSource === 'REAL_CAMERA' ? 'block' : 'hidden'
            }`}
          />
          <canvas
            ref={overlayCanvasRef}
            className={`pointer-events-none absolute inset-0 w-full h-full object-contain ${
              safeSnap.dataSource === 'REAL_CAMERA' ? 'block' : 'hidden'
            }`}
          />
          {safeSnap.dataSource !== 'REAL_CAMERA' && (
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

          {(anomalyNote || lowEdgeDense || highEdgeDense || finderTarget) && isRunning && (
            <div className="pointer-events-none absolute bottom-2.5 left-2.5 right-2.5 flex flex-wrap gap-1.5">
              {finderTarget && <span className="screen-label text-amber">寻物目标：{finderTarget.label}</span>}
              {anomalyNote && <span className="screen-label text-amber">实验性环境声响提示</span>}
              {lowEdgeDense && <span className="screen-label">画面下方边缘较密（物体未知）</span>}
              {highEdgeDense && <span className="screen-label">画面上方边缘较密（物体未知）</span>}
            </div>
          )}
        </div>

        {/* Confirmed detections list (strictly relative to camera frame) */}
        {safeSnap.canTrustLivePerception && detections.length > 0 && (
          <div className="space-y-1.5" aria-label="相对当前摄像头画面识别到的物体">
            <div className="text-[11.5px] text-ink-3">方位均相对当前摄像头画面：</div>
            <div className="flex flex-wrap gap-2">
              {detections.slice(0, 4).map((d, i) => (
                <span key={i} className="chip">
                  <strong className="text-signal-ink">{d.direction}</strong>
                  <span className="num text-ink">{d.distanceText}</span>
                  <span>{d.labelZh}</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Collapsible Technical Telemetry (P1-2: keep main screen clean for blind/low-vision users) */}
        <details className="text-[12px] text-ink-3 border-t border-line pt-2.5">
          <summary className="cursor-pointer font-medium text-ink-2 select-none">
            展开技术运行参数（EchoSafe 状态 · 帧率 · 延迟）
          </summary>
          <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2 num text-[11.5px]">
            <div className="panel-sunken px-2.5 py-2">
              <div className="eyebrow">状态码</div>
              <div className="text-ink mt-0.5">{safeSnap.state}</div>
            </div>
            <div className="panel-sunken px-2.5 py-2">
              <div className="eyebrow">视觉 / AI FPS</div>
              <div className="text-ink mt-0.5">
                {perf.visionFps} / {perf.aiFps}
              </div>
            </div>
            <div className="panel-sunken px-2.5 py-2">
              <div className="eyebrow">推理 P95</div>
              <div className="text-ink mt-0.5">{perf.p95InferenceMs} ms</div>
            </div>
            <div className="panel-sunken px-2.5 py-2">
              <div className="eyebrow">端到端 P95</div>
              <div className="text-ink mt-0.5">{perf.p95E2eMs} ms</div>
            </div>
          </div>
        </details>
      </section>

      {/* Voice Query & Audio Cue Strategy */}
      <section className="panel p-5 space-y-4" aria-label="语音查询与听觉策略">
        <button
          type="button"
          onClick={startTalking}
          className={`w-full min-h-[78px] rounded-xl border-2 flex flex-col items-center justify-center gap-1 transition-colors ${
            isListening
              ? 'border-signal bg-signal text-white'
              : 'border-ink bg-ink text-paper hover:bg-ink/90'
          }`}
        >
          {isListening ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
          <span className="text-[15.5px] font-semibold text-inherit">
            {isListening ? '正在聆听语音指令…' : '语音查询与操控'}
          </span>
          <span className="text-[12px] opacity-80">
            查询识别结果 · 设定寻物目标 · 测光识色 · 语音静音与暂停
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
          安全与隐私说明：本系统为浏览器端辅助感知原型，方位提示均相对当前摄像头画面，不可替代白手杖、导盲犬或现场交通安全判断；
          视觉与音频在本地计算，但浏览器语音识别（Web Speech Recognition）在多数设备上会通过厂商云服务转写。
        </p>
      </section>
    </div>
  );
};
