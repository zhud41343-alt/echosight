import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Power, Volume2, VolumeX, AlertTriangle } from 'lucide-react';
import {
  processFrameToAcousticMatrix,
  SpatialSceneMetrics,
} from '../engine/visionProcessor';
import {
  CameraCalibration,
  clearSavedCameraCalibration,
  computeCameraCalibrationFromDetection,
  detectSpatialObjects,
  loadOnDeviceAIModel,
  loadSavedCameraCalibration,
  saveCameraCalibration,
  SpatialDetection,
  SpatialObjectTracker,
  speakCue,
  speakSpatialDetections,
} from '../engine/aiObjectDetector';
import { spatialAudio } from '../engine/spatialAudio';
import { analyzeLightAndColor } from '../engine/blindAssistTools';
import { respondToUserUtterance } from '../engine/xiaoGuangAssistant';
import { AcousticThreatMonitor } from '../engine/acousticThreatDetector';
import {
  canEmitContinuousSpatialAudio,
  createInitialSignals,
  EchoSafeSignals,
  EchoSafeSnapshot,
  evaluateEchoSafe,
  FrameTracker,
  isPerceptionPacketUsable,
  PerceptionPacket,
  PerformanceTelemetry,
  RuntimeMetricsTracker,
  SessionGuard,
  SingleFlightRunner,
  SystemModeState,
} from '../engine/echoSafe';
import { audioScheduler } from '../engine/audioScheduler';
import { hapticFeedback, ScreenWakeLockController } from '../engine/deviceFeedback';

const MAX_PROC_DIM = 640;
const EDGE_W = 160;
const EDGE_H = 120;
const COLS = 48;
const ROWS = 16;
const PACKET_TTL_MS = 3500;
const MAX_FRAME_AGE_MS = 3000;

type AudioCueStrategy =
  | 'concise_voice'
  | 'low_freq_spatial'
  | 'on_demand'
  | 'continuous_sonification';

const AUDIO_STRATEGIES: { id: AudioCueStrategy; label: string; desc: string }[] = [
  { id: 'concise_voice', label: '简洁语音优先（默认）', desc: '仅播报稳定识别变化，减少环境听觉干扰' },
  { id: 'low_freq_spatial', label: '低频空间提示', desc: '近距离稳定目标播放低频方位提示' },
  { id: 'on_demand', label: '仅按需描述', desc: '平时安静，仅在你主动查询时播报' },
  { id: 'continuous_sonification', label: '实验性连续声场', desc: '持续 3D 声波扫描映射（较耗听觉注意力）' },
];

type VideoWithVfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number) => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

interface SpeechRecEvent {
  error?: string;
}
interface SpeechRecInstance {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((ev: SpeechRecEvent) => void) | null;
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  start: () => void;
  abort: () => void;
}
type SpeechRecCtor = new () => SpeechRecInstance;

export const BlindCompanionView: React.FC = () => {
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const edgeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const acousticMonitorRef = useRef<AcousticThreatMonitor | null>(null);
  const sessionGuardRef = useRef(new SessionGuard());
  const singleFlightRef = useRef(new SingleFlightRunner());
  const metricsTrackerRef = useRef(new RuntimeMetricsTracker());
  const trackerRef = useRef(new SpatialObjectTracker());
  const frameTrackerRef = useRef(new FrameTracker());
  const wakeLockRef = useRef(new ScreenWakeLockController());
  const lastEdgeFrameIdRef = useRef<number>(0);
  const prevSafeStateRef = useRef<SystemModeState>('IDLE');
  const procSizeRef = useRef<{ w: number; h: number }>({ w: 640, h: 480 });

  // P0-4: Explicit SpeechRecognition lifecycle refs
  const speechRecRef = useRef<SpeechRecInstance | null>(null);
  const voiceSessionIdRef = useRef<number>(0);

  const signalsRef = useRef<EchoSafeSignals>(createInitialSignals());
  const [safeSnap, setSafeSnap] = useState<EchoSafeSnapshot>(() =>
    evaluateEchoSafe(createInitialSignals(), performance.now())
  );
  const [perf, setPerf] = useState<PerformanceTelemetry>(() =>
    metricsTrackerRef.current.getSnapshot(performance.now())
  );

  const packetRef = useRef<PerceptionPacket<SpatialDetection> | null>(null);
  const [detections, setDetections] = useState<SpatialDetection[]>([]);

  const [audioStrategy, setAudioStrategy] = useState<AudioCueStrategy>('concise_voice');
  const audioStrategyRef = useRef<AudioCueStrategy>('concise_voice');
  audioStrategyRef.current = audioStrategy;
  const [experimentalMicAlert, setExperimentalMicAlert] = useState(false);
  const [screenReaderSafe, setScreenReaderSafe] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem('echosight:sr_safe') === '1';
  });
  const [hapticsEnabled, setHapticsEnabled] = useState(true);
  const [wakeLockActive, setWakeLockActive] = useState(false);
  const [calibration, setCalibration] = useState<CameraCalibration | null>(() =>
    loadSavedCameraCalibration()
  );
  const calibrationRef = useRef<CameraCalibration | null>(calibration);
  calibrationRef.current = calibration;
  const [calibrationNote, setCalibrationNote] = useState<string>('');
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

  const lastVideoCurrentTimeRef = useRef<number>(-1);
  const vfcHandleRef = useRef<number | null>(null);
  const lastEdgeEvalRef = useRef<number>(0);
  const scanColRef = useRef(0);

  const clearOverlayCanvas = useCallback(() => {
    const ov = overlayCanvasRef.current;
    if (ov) {
      const oCtx = ov.getContext('2d');
      oCtx?.clearRect(0, 0, ov.width, ov.height);
    }
  }, []);

  // P0-4: Abort any active SpeechRecognition instance and detach callbacks cleanly
  const abortSpeechRecognition = useCallback(() => {
    voiceSessionIdRef.current += 1;
    const rec = speechRecRef.current;
    speechRecRef.current = null;
    if (rec) {
      rec.onstart = null;
      rec.onend = null;
      rec.onerror = null;
      rec.onresult = null;
      try {
        rec.abort();
      } catch {
        // Ignore abort errors
      }
    }
    setIsListening(false);
  }, []);

  // P0-2: Unified strict PerceptionPacket validity gate
  const getUsableDetections = useCallback(
    (now: number, currentSafeState: SystemModeState): SpatialDetection[] => {
      const pkt = packetRef.current;
      if (
        !isPerceptionPacketUsable(
          pkt,
          sessionGuardRef.current.currentId,
          currentSafeState,
          now,
          MAX_FRAME_AGE_MS
        )
      ) {
        return [];
      }
      return pkt!.detections;
    },
    []
  );

  const refreshSafeState = useCallback(() => {
    const now = performance.now();
    const ctx = spatialAudio.ensureContext();
    const audioRunning = Boolean(ctx && ctx.state === 'running');
    signalsRef.current.audioState = !ctx
      ? 'unavailable'
      : audioRunning
        ? 'running'
        : 'suspended';
    audioScheduler.setAudioOutputReady(audioRunning);
    signalsRef.current.pageHidden = typeof document !== 'undefined' ? document.hidden : false;

    const snap = evaluateEchoSafe(signalsRef.current, now);
    const prevState = prevSafeStateRef.current;
    prevSafeStateRef.current = snap.state;
    setSafeSnap(snap);
    setPerf(metricsTrackerRef.current.getSnapshot(now));

    if (snap.state === 'UNAVAILABLE' || snap.state === 'IDLE') {
      spatialAudio.stopAllImmediate();
      lastEdgeFrameIdRef.current = 0;
      if (lowEdgeRef.current) {
        lowEdgeRef.current = false;
        setLowEdgeDense(false);
      }
      if (highEdgeRef.current) {
        highEdgeRef.current = false;
        setHighEdgeDense(false);
      }
      // Explicit fail-safe feedback on transition into UNAVAILABLE (works even when audio is muted/unavailable)
      if (
        snap.state === 'UNAVAILABLE' &&
        (prevState === 'ACTIVE' || prevState === 'DEGRADED' || prevState === 'INITIALIZING')
      ) {
        hapticFeedback.trigger('unavailable');
        setAssistantMessage(`实时感知已失效：${snap.userSummary}`);
      }
    }

    if (!snap.canTrustLivePerception) {
      if (packetRef.current !== null) {
        packetRef.current = null;
        setDetections([]);
        clearOverlayCanvas();
      }
    }
    return snap;
  }, [clearOverlayCanvas]);

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

  // Sync screen-reader anti-collision mode & haptic preference
  useEffect(() => {
    audioScheduler.setScreenReaderSafeMode(screenReaderSafe);
    if (typeof window !== 'undefined') {
      if (screenReaderSafe) window.localStorage.setItem('echosight:sr_safe', '1');
      else window.localStorage.removeItem('echosight:sr_safe');
    }
  }, [screenReaderSafe]);

  useEffect(() => {
    hapticFeedback.setEnabled(hapticsEnabled);
  }, [hapticsEnabled]);

  const stopAllSensors = useCallback(() => {
    sessionGuardRef.current.invalidate();
    abortSpeechRecognition();
    stopVideoFrameWatcher();
    audioScheduler.clearAll();
    spatialAudio.stopAllImmediate();
    void wakeLockRef.current.release().then(() => setWakeLockActive(false));
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
    trackerRef.current.reset();
    frameTrackerRef.current.reset();
    lastEdgeFrameIdRef.current = 0;
    setDetections([]);
    clearOverlayCanvas();
    setFinderTarget(null);
    setAnomalyNote('');
    setLowEdgeDense(false);
    setHighEdgeDense(false);
    refreshSafeState();
  }, [abortSpeechRecognition, clearOverlayCanvas, refreshSafeState, stopVideoFrameWatcher]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      sessionGuardRef.current.invalidate();
      abortSpeechRecognition();
      stopVideoFrameWatcher();
      audioScheduler.clearAll();
      spatialAudio.stopAllImmediate();
      void wakeLockRef.current.release();
      acousticMonitorRef.current?.stop();
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
    };
  }, [abortSpeechRecognition, stopVideoFrameWatcher]);

  // Page visibility lifecycle (background immediately invalidates packet, stops spatial audio & releases Wake Lock)
  useEffect(() => {
    const onVisibility = () => {
      signalsRef.current.pageHidden = document.hidden;
      if (document.hidden && signalsRef.current.requestedLive) {
        metricsTrackerRef.current.recordMediaInterruption();
        abortSpeechRecognition();
        packetRef.current = null;
        trackerRef.current.reset();
        frameTrackerRef.current.reset();
        lastEdgeFrameIdRef.current = 0;
        setDetections([]);
        clearOverlayCanvas();
        audioScheduler.clearAll();
        spatialAudio.stopAllImmediate();
        void wakeLockRef.current.handleVisibilityChange(false, false).then(setWakeLockActive);
        refreshSafeState();
      } else if (!document.hidden && signalsRef.current.requestedLive) {
        const stream = videoRef.current?.srcObject as MediaStream | null;
        const trackLive = Boolean(stream && stream.getVideoTracks().some((t) => t.readyState === 'live'));
        signalsRef.current.cameraTrackLive = trackLive;
        signalsRef.current.lastFrameAt = null;
        signalsRef.current.lastInferenceAt = null;
        signalsRef.current.sessionStartedAt = performance.now();
        packetRef.current = null;
        trackerRef.current.reset();
        frameTrackerRef.current.reset();
        lastEdgeFrameIdRef.current = 0;
        setDetections([]);
        clearOverlayCanvas();
        spatialAudio.ensureContext();
        void wakeLockRef.current.handleVisibilityChange(true, trackLive).then(setWakeLockActive);
        refreshSafeState();
        if (!trackLive) {
          say('页面已切回前台，但摄像头视频流已中断，请重新启动实时感知。', 'EMERGENCY');
        }
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [abortSpeechRecognition, clearOverlayCanvas, refreshSafeState, say]);

  // Periodic health & packet TTL check (1s)
  useEffect(() => {
    const id = window.setInterval(() => {
      if (signalsRef.current.requestedLive || signalsRef.current.initializing) {
        refreshSafeState();
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [refreshSafeState]);

  const attachVideoFrameWatcher = useCallback((mySession: number) => {
    const vid = videoRef.current as VideoWithVfc | null;
    if (!vid || typeof vid.requestVideoFrameCallback !== 'function') return;

    const onFrame = (now: number) => {
      if (!sessionGuardRef.current.isValid(mySession)) return;
      if (signalsRef.current.cameraTrackLive && !signalsRef.current.pageHidden) {
        signalsRef.current.lastFrameAt = now;
        frameTrackerRef.current.recordNewFrame(now);
        metricsTrackerRef.current.recordVisionFrame(now);
      }
      vfcHandleRef.current = vid.requestVideoFrameCallback!(onFrame);
    };
    vfcHandleRef.current = vid.requestVideoFrameCallback(onFrame);
  }, []);

  const toggleCompanion = useCallback(async () => {
    spatialAudio.ensureContext();
    hapticFeedback.trigger('confirm');

    if (signalsRef.current.requestedLive || signalsRef.current.initializing) {
      stopAllSensors();
      say('已停止实时感知。', 'USER_QUERY');
      return;
    }

    const mySession = sessionGuardRef.current.begin();
    trackerRef.current.reset();
    frameTrackerRef.current.reset();
    lastEdgeFrameIdRef.current = 0;
    void wakeLockRef.current.acquire().then((held) => {
      if (sessionGuardRef.current.isValid(mySession)) setWakeLockActive(held);
    });
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
      say('当前浏览器不支持摄像头采集，实时感知不可用。', 'EMERGENCY');
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
          trackerRef.current.reset();
          frameTrackerRef.current.reset();
          lastEdgeFrameIdRef.current = 0;
          spatialAudio.stopAllImmediate();
          setDetections([]);
          clearOverlayCanvas();
          refreshSafeState();
          say('摄像头视频流已断开，实时感知不可用。', 'EMERGENCY');
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
        isDenied ? '摄像头权限被拒绝，实时感知不可用。' : '未检测到可用摄像头，实时感知不可用。',
        'EMERGENCY'
      );
      return;
    }

    // Optional experimental single-channel ambient sound anomaly monitor (OFF by default to prevent false chatter)
    if (experimentalMicAlert) {
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
    }

    const loadRes = await loadOnDeviceAIModel(undefined, true);
    if (!sessionGuardRef.current.isValid(mySession)) return;

    metricsTrackerRef.current.setModelLoadMs(loadRes.loadMs);
    signalsRef.current.modelState = loadRes.modelReady ? 'ready' : 'failed';
    signalsRef.current.initializing = false;

    const snap = refreshSafeState();
    if (loadRes.modelReady) {
      say('实时感知已启动。方位相对当前摄像头画面，仅作辅助参考。', 'USER_QUERY');
    } else {
      say(`系统进入降级状态：${snap.userSummary}`, 'EMERGENCY');
    }
  }, [attachVideoFrameWatcher, clearOverlayCanvas, experimentalMicAlert, refreshSafeState, say, stopAllSensors]);

  // Lightweight throttled edge analysis (~8fps on 160x120 canvas -> 96% less mobile CPU!)
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
        if (!hasVfc && vid.currentTime !== lastVideoCurrentTimeRef.current) {
          lastVideoCurrentTimeRef.current = vid.currentTime;
          signalsRef.current.lastFrameAt = now;
          frameTrackerRef.current.recordNewFrame(now);
          metricsTrackerRef.current.recordVisionFrame(now);
        }

        // P0-2 (V5): Gate Sobel edge analysis & continuous spatial audio strictly on fresh video frame + live EchoSafe
        const audioGate = canEmitContinuousSpatialAudio(signalsRef.current, now);
        const frameSnap = frameTrackerRef.current.getSnapshot();
        if (!audioGate.allowed || frameSnap.latestFrameId === 0) {
          if (lastEdgeFrameIdRef.current !== 0) {
            lastEdgeFrameIdRef.current = 0;
            spatialAudio.stopAllImmediate();
            if (lowEdgeRef.current) {
              lowEdgeRef.current = false;
              setLowEdgeDense(false);
            }
            if (highEdgeRef.current) {
              highEdgeRef.current = false;
              setHighEdgeDense(false);
            }
          }
        } else if (now - lastEdgeEvalRef.current >= 120) {
          lastEdgeEvalRef.current = now;
          lastEdgeFrameIdRef.current = frameSnap.latestFrameId;
          const edgeCanvas = edgeCanvasRef.current;
          const eCtx = edgeCanvas?.getContext('2d', { willReadFrequently: true });
          if (edgeCanvas && eCtx) {
            eCtx.drawImage(vid, 0, 0, EDGE_W, EDGE_H);
            const imgData = eCtx.getImageData(0, 0, EDGE_W, EDGE_H);
            const res = processFrameToAcousticMatrix(imgData, COLS, ROWS, 40);
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
            const lDense = lowSum / Math.max(1, cells) > 0.48;
            const hDense = highSum / Math.max(1, cells) > 0.5;
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
              spatialAudio.playScanColumn(res.matrix[col], col / (COLS - 1), 0.08, {
                scaleMode: 'pentatonic',
                useHRTF: true,
                volume: volume * 0.35,
              });
              scanColRef.current = (col + 1) % COLS;
            }
          }
        }
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [volume]);

  // Responsive single-flight AI inference loop (320ms cadence + IoU SpatialObjectTracker)
  useEffect(() => {
    let cancelled = false;

    const timer = window.setInterval(async () => {
      if (cancelled || !sessionGuardRef.current.isActive) return;
      const vid = videoRef.current;
      if (
        !vid ||
        vid.readyState < 2 ||
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

      await singleFlightRef.current.run(async () => {
        const src = sourceCanvasRef.current;
        const sCtx = src?.getContext('2d', { willReadFrequently: true });
        if (!src || !sCtx || cancelled || !sessionGuardRef.current.isValid(mySession)) return;

        // P0-1 (V5): Require a genuinely NEW, fresh video frame before running AI inference
        const claimedFrame = frameTrackerRef.current.claimNextFrameForInference(
          performance.now(),
          MAX_FRAME_AGE_MS
        );
        if (!claimedFrame) return;

        // Draw single clean frame at native aspect ratio right before inference (zero wasted 60fps copies)
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
        sCtx.drawImage(vid, 0, 0, targetW, targetH);

        const t0 = performance.now();
        try {
          const rawRes = await detectSpatialObjects(
            src,
            false,
            'crosswalk',
            calibrationRef.current
          );
          const tEnd = performance.now();
          const duration = tEnd - t0;

          // P0-1 & P0-2: Verify SessionGuard BEFORE mutating tracker state or writing results!
          if (cancelled || !sessionGuardRef.current.isValid(mySession)) return;

          const res = trackerRef.current.update(rawRes, targetW, targetH, claimedFrame.frameId);

          signalsRef.current.lastInferenceAt = tEnd;
          signalsRef.current.lastInferenceDurationMs = duration;
          signalsRef.current.consecutiveInferenceFailures = 0;
          metricsTrackerRef.current.recordInference(tEnd, duration, tEnd - claimedFrame.capturedAt);

          const snap = refreshSafeState();
          if (snap.canTrustLivePerception) {
            packetRef.current = {
              dataSource: 'REAL_CAMERA',
              sessionId: mySession,
              frameId: claimedFrame.frameId,
              capturedAt: claimedFrame.capturedAt,
              inferredAt: tEnd,
              expiresAt: tEnd + PACKET_TTL_MS,
              safeState: snap.state,
              detections: res,
            };

            const usable = getUsableDetections(tEnd, snap.state);
            setDetections(usable);

            // Render live bounding boxes aligned to native aspect ratio
            const ov = overlayCanvasRef.current;
            if (ov) {
              if (ov.width !== targetW || ov.height !== targetH) {
                ov.width = targetW;
                ov.height = targetH;
              }
              const oCtx = ov.getContext('2d');
              if (oCtx) {
                oCtx.clearRect(0, 0, targetW, targetH);
                usable.forEach((det) => {
                  const [bx, by, bw, bh] = det.bbox;
                  const color = det.distance === '近距离' ? '#ff8a3d' : '#9fc9b0';
                  oCtx.strokeStyle = color;
                  oCtx.lineWidth = 2.4;
                  oCtx.strokeRect(bx, by, bw, bh);

                  const tag = `${det.direction} · ${det.labelZh} ${Math.round(det.score * 100)}%`;
                  oCtx.font = '600 13px "JetBrains Mono", Consolas, sans-serif';
                  const tw = oCtx.measureText(tag).width + 12;
                  const tx = Math.max(4, Math.min(targetW - tw - 4, bx));
                  const ty = Math.max(22, by);
                  oCtx.fillStyle = 'rgba(22, 21, 18, 0.88)';
                  oCtx.fillRect(tx, ty - 22, tw, 22);
                  oCtx.fillStyle = color;
                  oCtx.fillText(tag, tx + 6, ty - 6);
                });
              }
            }

            const target = finderRef.current;
            if (target) {
              const hit = usable.find(
                (d) => d.labelEn === target.id || (target.id === 'cup' && d.labelEn === 'bottle')
              );
              if (hit && audioScheduler.shouldAllowSpatialCue('finder', tEnd)) {
                const [bx, , bw] = hit.bbox;
                const normX = Math.max(0, Math.min(1, (bx + bw / 2) / targetW));
                const centeredness = 1 - Math.abs(normX - 0.5) * 2;
                const closeness = Math.min(
                  1,
                  centeredness * 0.65 + (hit.distance === '近距离' ? 0.35 : 0.1)
                );
                spatialAudio.playFinderBeacon(normX, closeness, Math.max(0.5, volume));
              }
            }

            if (usable.length > 0 && audioStrategyRef.current !== 'on_demand') {
              if (usable[0].distance === '近距离' && usable[0].priority === 'danger') {
                hapticFeedback.trigger('warning', tEnd);
              }
              if (
                audioStrategyRef.current === 'low_freq_spatial' &&
                usable[0].distance === '近距离' &&
                audioScheduler.shouldAllowSpatialCue('low_freq', tEnd)
              ) {
                const [bx, , bw] = usable[0].bbox;
                const normX = Math.max(0, Math.min(1, (bx + bw / 2) / targetW));
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
    }, 320);

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
      const usableDetections = getUsableDetections(now, snap.state);
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
          // Only silence continuous/ambient cues — NEVER silence user-query replies!
          setVolume(0);
          setAudioStrategy('on_demand');
        } else if (d === 'unmute') {
          audioScheduler.setMuted(false);
          setVolume(0.55);
          setAudioStrategy('concise_voice');
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

  // P0-4: Full lifecycle-managed SpeechRecognition (prevents overlapping instances & late callbacks)
  const startTalking = useCallback(() => {
    spatialAudio.ensureContext();
    setVoiceErrorNote('');

    // Clicking while already listening cleanly aborts the current recognition instance
    if (speechRecRef.current) {
      abortSpeechRecognition();
      return;
    }

    if (typeof window === 'undefined') return;
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

    const myVoiceSession = ++voiceSessionIdRef.current;
    const rec = new SpeechRec();
    rec.lang = 'zh-CN';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    let gotResult = false;

    rec.onstart = () => {
      if (voiceSessionIdRef.current !== myVoiceSession) return;
      setIsListening(true);
    };
    rec.onend = () => {
      if (voiceSessionIdRef.current !== myVoiceSession) return;
      speechRecRef.current = null;
      setIsListening(false);
      if (!gotResult) {
        setVoiceErrorNote('未检测到有效语音输入，你可以重新点击语音按钮或改用下方快捷查询按钮。');
      }
    };
    rec.onerror = (ev) => {
      if (voiceSessionIdRef.current !== myVoiceSession) return;
      speechRecRef.current = null;
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
      if (voiceSessionIdRef.current !== myVoiceSession) return;
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

    speechRecRef.current = rec;
    try {
      rec.start();
    } catch {
      abortSpeechRecognition();
      const msg = '无法启动语音识别，请稍后重试或使用下方快捷按钮。';
      setVoiceErrorNote(msg);
      say(msg, 'USER_QUERY');
    }
  }, [abortSpeechRecognition, handleUtterance, say]);

  const isRunning = safeSnap.state === 'ACTIVE' || safeSnap.state === 'DEGRADED';

  return (
    <div className="mx-auto max-w-2xl space-y-5">
      <canvas ref={sourceCanvasRef} width={640} height={480} className="hidden" />
      <canvas ref={edgeCanvasRef} width={EDGE_W} height={EDGE_H} className="hidden" />

      {/* Layer 1 & 2: Clear Status + Primary Controls */}
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
              {audioStrategy === 'on_demand' ? '按需安静' : '简洁播报'}
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

        {/* Layer 3: Verified Speech Readout */}
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

        {/* Full Uncropped Camera Viewport + Live Aligned BBox Overlay */}
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

        {/* Confirmed Spatially-Tracked Detections */}
        {safeSnap.canTrustLivePerception && detections.length > 0 && (
          <div className="space-y-1.5" aria-label="相对当前摄像头画面识别到的物体">
            <div className="text-[11.5px] text-ink-3">
              方位均相对当前摄像头画面（经 IoU 多目标稳定追踪）：
            </div>
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

        {/* Layer 4: Collapsible Technical Telemetry */}
        <details className="text-[12px] text-ink-3 border-t border-line pt-2.5">
          <summary className="cursor-pointer font-medium text-ink-2 select-none">
            展开技术运行参数与实验开关（帧率 · 延迟 · 环境声响提示）
          </summary>
          <div className="mt-2.5 space-y-2.5">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 num text-[11.5px]">
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
            <div className="space-y-1.5">
              <label className="flex items-center gap-2 text-[12px] text-ink-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={screenReaderSafe}
                  onChange={(e) => setScreenReaderSafe(e.target.checked)}
                />
                读屏器防冲突模式（静音网页内置朗读，交由 VoiceOver / TalkBack 通过 ARIA 朗读）
              </label>
              <label className="flex items-center gap-2 text-[12px] text-ink-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hapticsEnabled}
                  onChange={(e) => setHapticsEnabled(e.target.checked)}
                />
                事件触觉振动提示（启动确认 / 近距危险 / 失效报警，支持振动设备生效）
              </label>
              <label className="flex items-center gap-2 text-[12px] text-ink-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={experimentalMicAlert}
                  onChange={(e) => setExperimentalMicAlert(e.target.checked)}
                />
                下次启动时开启「实验性单路环境异常声响提示」（默认关闭以减少误报干扰）
              </label>
              <div className="text-[11px] text-ink-3">
                屏幕常亮保护（Screen Wake Lock）：{wakeLockActive ? '已激活（防行进中自动息屏）' : '待机 / 设备不支持'}
              </div>
            </div>

            <div className="panel-sunken p-2.5 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="eyebrow">单目相机焦距标定（可选）</div>
                  <div className="text-[11.5px] text-ink-2 mt-0.5">
                    {calibration?.calibrated
                      ? `已标定（f = ${calibration.focalLengthPx}px @ ${calibration.referenceHeightPx}px，输出 ±15% 保守区间）`
                      : '未标定（默认仅输出近/中/远保守粗估，不输出小数米精度）'}
                  </div>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    disabled={!safeSnap.canTrustLivePerception || detections.length === 0}
                    onClick={() => {
                      const primary = detections[0];
                      if (!primary) return;
                      const nextCal = computeCameraCalibrationFromDetection(
                        primary.labelEn,
                        primary.bbox,
                        2.0,
                        procSizeRef.current.h,
                        'User 2.0m Reference'
                      );
                      if (!nextCal) {
                        setCalibrationNote(
                          '标定未通过：请将已知高度物体（如行人/椅子）完整置于 2.0 米处且勿贴边裁切。'
                        );
                        return;
                      }
                      saveCameraCalibration(nextCal);
                      setCalibration(nextCal);
                      setCalibrationNote(
                        `已按 2.0 米处「${primary.labelZh}」完成标定（f = ${nextCal.focalLengthPx}px）。`
                      );
                    }}
                    className="btn btn-sm disabled:opacity-45"
                  >
                    按 2.0m 参考物标定
                  </button>
                  {calibration?.calibrated && (
                    <button
                      type="button"
                      onClick={() => {
                        clearSavedCameraCalibration();
                        setCalibration(null);
                        setCalibrationNote('已清除标定，恢复未标定保守粗估。');
                      }}
                      className="btn btn-sm"
                    >
                      清除标定
                    </button>
                  )}
                </div>
              </div>
              {calibrationNote && <div className="text-[11px] text-signal-ink">{calibrationNote}</div>}
            </div>
          </div>
        </details>
      </section>

      {/* Voice Query & Audio Strategy */}
      <section className="panel p-5 space-y-4" aria-label="语音查询与听觉策略">
        <button
          type="button"
          onClick={startTalking}
          className={`w-full min-h-[76px] rounded-xl border-2 flex flex-col items-center justify-center gap-1 transition-colors ${
            isListening
              ? 'border-signal bg-signal text-white'
              : 'border-ink bg-ink text-paper hover:bg-ink/90'
          }`}
        >
          {isListening ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
          <span className="text-[15.5px] font-semibold text-inherit">
            {isListening ? '正在聆听（再次点击取消）' : '语音查询与操控'}
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
                  if (st.id !== 'continuous_sonification') {
                    spatialAudio.stopAllImmediate();
                  }
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
          安全与隐私说明：本系统为浏览器端辅助感知原型，方位提示均相对当前摄像头画面（非人体朝向），不可替代白手杖、导盲犬或现场交通安全判断；
          视觉与音频在本地计算，但浏览器语音识别（Web Speech Recognition）在多数设备上会通过厂商云服务转写。
        </p>
      </section>
    </div>
  );
};
