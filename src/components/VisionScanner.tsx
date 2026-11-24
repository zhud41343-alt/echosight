import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Play,
  Pause,
  Camera,
  CameraOff,
  Upload,
  Brain,
  Volume2,
  Crosshair,
  Accessibility,
  Mic,
  MicOff,
  Sun,
  Target,
} from 'lucide-react';
import {
  drawMediaContain,
  drawPresetScene,
  PresetSceneId,
  processFrameToAcousticMatrix,
  SpatialSceneMetrics,
} from '../engine/visionProcessor';
import { ScaleMode, spatialAudio } from '../engine/spatialAudio';
import {
  detectSpatialObjects,
  loadOnDeviceAIModel,
  SpatialDetection,
  speakCue,
  speakSpatialDetections,
} from '../engine/aiObjectDetector';
import {
  analyzeLightAndColor,
  FINDER_TARGETS,
  LightAndColorReport,
  parseVoiceCommand,
} from '../engine/blindAssistTools';
import {
  ECHO_DEPTH_MODEL_VERSION,
  loadEchoDepthOnnxModel,
  runEchoDepthInference,
} from '../engine/echoDepthOnnx';
import { AcousticMatrixRouter, ActiveAcousticSource } from '../engine/acousticRouter';
import { FrameTracker } from '../engine/echoSafe';

type VideoWithVfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number) => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

const CANVAS_W = 480;
const CANVAS_H = 300;
const MATRIX_COLS = 48;
const MATRIX_ROWS = 16;

type ViewMode = 'source' | 'sobel' | 'matrix' | 'depth';

const PRESETS: { id: PresetSceneId; label: string; desc: string }[] = [
  { id: 'crosswalk', label: '斑马线与路口', desc: '信号灯 + 来车轮廓' },
  { id: 'room', label: '室内客厅避障', desc: '桌灯障碍 + 明亮门框' },
  { id: 'mountain', label: '星月与山峦', desc: '起伏山脊立体旋律' },
  { id: 'calibration', label: '声学空间校准', desc: '斜线升调 + 同心圆' },
];

export const VisionScanner: React.FC = () => {
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const spectrumCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [preset, setPreset] = useState<PresetSceneId>('mountain');
  const [inputSource, setInputSource] = useState<'preset' | 'camera' | 'upload'>('preset');
  const [uploadVersion, setUploadVersion] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>('sobel');
  const [isScanning, setIsScanning] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraDisconnected, setCameraDisconnected] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const cameraStartingRef = useRef(false);
  const lastAiCameraFrameIdRef = useRef<number>(0);

  // On-Device AI Object Detection state
  const [aiEnabled, setAiEnabled] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiStatus, setAiStatus] = useState<string>('');
  const [depthStatus, setDepthStatus] = useState<string>('');
  const [depthMs, setDepthMs] = useState<number | null>(null);
  const depthHeatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const depthBusyRef = useRef(false);
  const [autoSpeakAI, setAutoSpeakAI] = useState(false);
  const [detections, setDetections] = useState<SpatialDetection[]>([]);
  const detectionsRef = useRef<SpatialDetection[]>([]);

  // Live telemetry & interactive point-probe state
  const [scanTelemetry, setScanTelemetry] = useState<{ azimuthDeg: number; freqHz: number | null }>({
    azimuthDeg: 0,
    freqHz: null,
  });
  const [probeHud, setProbeHud] = useState<{
    azimuthDeg: number;
    freqHz: number;
    noteName: string;
    energyPct: number;
  } | null>(null);
  const isProbingRef = useRef(false);
  const lastProbeSoundRef = useRef(0);
  const probePointRef = useRef<{ x: number; y: number; at: number } | null>(null);

  // Algorithm parameters
  const [scaleMode, setScaleMode] = useState<ScaleMode>('pentatonic');
  const [useHRTF, setUseHRTF] = useState(true);
  const [edgeThreshold, setEdgeThreshold] = useState(36);
  const [scanSpeedMs, setScanSpeedMs] = useState(55); // ms per column
  const [volume, setVolume] = useState(0.7);

  const [metrics, setMetrics] = useState<SpatialSceneMetrics>({
    leftDensity: 24,
    centerDensity: 35,
    rightDensity: 42,
    edgeCoverage: 30,
    navigationCue: '点击「启动声波扫描」，聆听画面在 3D 空间中的轮廓分布',
  });
  const metricsRef = useRef<SpatialSceneMetrics>(metrics);
  metricsRef.current = metrics;

  const [assistMode, setAssistMode] = useState(false);
  const savedVolumeRef = useRef(0.7);

  // Real-world blind assist tools: 3D Item Finder, Light & Color, Voice Command
  const [finderTargetId, setFinderTargetId] = useState<string>('');
  const finderTargetRef = useRef<string>('');
  finderTargetRef.current = finderTargetId;
  const [lightReport, setLightReport] = useState<LightAndColorReport | null>(null);
  const [isListeningVoice, setIsListeningVoice] = useState(false);
  const [voiceFeedback, setVoiceFeedback] = useState<string>('');
  const lastFinderPingRef = useRef(0);

  // Refs for real-time animation loop (strictly separated Sobel vs Neural EchoDepthNet router)
  const routerRef = useRef(new AcousticMatrixRouter(MATRIX_COLS, MATRIX_ROWS));
  const cameraFrameTrackerRef = useRef(new FrameTracker());
  const lastSobelCameraFrameIdRef = useRef<number>(0);
  const vfcHandleRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);
  const cameraSessionRef = useRef<number>(0);
  const assistSessionRef = useRef<number>(0);
  const aiInFlightRef = useRef<boolean>(false);
  const lastCommittedAiFrameIdRef = useRef<number>(0);
  const aiPacketRef = useRef<{
    sessionId: number;
    inputSource: 'preset' | 'upload' | 'camera';
    frameId: number;
    capturedAt: number;
    startedAt: number;
    completedAt: number;
    expiresAt: number;
    detections: SpatialDetection[];
  } | null>(null);
  const [activeAudioSource, setActiveAudioSource] = useState<ActiveAcousticSource>('sobel');
  const edgeImageRef = useRef<ImageData | null>(null);
  const scanColRef = useRef<number>(0);
  const lastStepTimeRef = useRef<number>(0);
  const uploadedImgRef = useRef<HTMLImageElement | null>(null);

  const isLiveCameraFrameFresh = useCallback(
    (now: number): boolean => {
      if (!cameraActive || cameraDisconnected) return false;
      if (typeof document !== 'undefined' && document.hidden) return false;
      const snap = cameraFrameTrackerRef.current.getSnapshot();
      if (snap.latestFrameId <= 0 || snap.latestFrameCapturedAt === null) return false;
      const age = now - snap.latestFrameCapturedAt;
      return age >= -50 && age <= 3000;
    },
    [cameraActive, cameraDisconnected]
  );

  // P0-01 & P0-02 (V8.9): Single unified PerceptionPacket validity gate for ALL AI speech & HUD readers
  const getUsableAiDetections = useCallback(
    (now: number): SpatialDetection[] | null => {
      const pkt = aiPacketRef.current;
      if (inputSource === 'camera') {
        if (!isLiveCameraFrameFresh(now)) return null;
        if (!pkt || pkt.inputSource !== 'camera') return null;
        if (pkt.sessionId !== cameraSessionRef.current) return null;
        if (now > pkt.expiresAt || now - pkt.capturedAt > 3000) return null;
        return pkt.detections;
      }
      return pkt && pkt.inputSource === inputSource ? pkt.detections : detectionsRef.current;
    },
    [inputSource, isLiveCameraFrameFresh]
  );

  const recomputeCurrentFrame = useCallback(() => {
    const srcCanvas = sourceCanvasRef.current;
    if (!srcCanvas) return;
    const ctx = srcCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // P0-01 (V8.7): When camera disconnected (inputSource === 'camera' && !cameraActive),
    // NEVER silently fall back to drawing a virtual preset scene!
    if (inputSource === 'camera' && !cameraActive) {
      routerRef.current.invalidateSobel();
      routerRef.current.invalidateNeural();
      detectionsRef.current = [];
      setDetections([]);
      return;
    }

    if (cameraActive && videoRef.current && videoRef.current.readyState >= 2) {
      drawMediaContain(ctx, videoRef.current, CANVAS_W, CANVAS_H);
    } else if (uploadedImgRef.current) {
      ctx.fillStyle = '#161512';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      const img = uploadedImgRef.current;
      const scale = Math.min(CANVAS_W / img.width, CANVAS_H / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      ctx.drawImage(img, (CANVAS_W - dw) / 2, (CANVAS_H - dh) / 2, dw, dh);
    } else {
      drawPresetScene(ctx, CANVAS_W, CANVAS_H, preset);
    }

    const imgData = ctx.getImageData(0, 0, CANVAS_W, CANVAS_H);
    const res = processFrameToAcousticMatrix(imgData, MATRIX_COLS, MATRIX_ROWS, edgeThreshold);
    if (!cameraActive) {
      routerRef.current.updateSobel(res.matrix, {
        inputSource: uploadedImgRef.current ? 'upload' : 'preset',
        frameId: 0,
        capturedAt: performance.now(),
      });
    }
    edgeImageRef.current = res.edgeImageData;
    setMetrics(res.metrics);
  }, [cameraActive, inputSource, preset, edgeThreshold]);

  // Recompute whenever preset or threshold changes
  useEffect(() => {
    recomputeCurrentFrame();
  }, [recomputeCurrentFrame]);

  // Cleanup camera & video frame watcher on unmount
  const stopCameraFrameWatcher = useCallback(() => {
    const vid = videoRef.current as VideoWithVfc | null;
    if (vid && vfcHandleRef.current !== null && typeof vid.cancelVideoFrameCallback === 'function') {
      try {
        vid.cancelVideoFrameCallback(vfcHandleRef.current);
      } catch {
        // Ignore
      }
    }
    vfcHandleRef.current = null;
    lastVideoTimeRef.current = -1;
    lastSobelCameraFrameIdRef.current = 0;
    lastAiCameraFrameIdRef.current = 0;
    cameraFrameTrackerRef.current.reset();
    routerRef.current.invalidateSobel();
  }, []);

  // P1-03 (V8.7): Unified all-mode page visibility listener (stops Sobel & Neural camera audio immediately on hide)
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) {
        routerRef.current.invalidateNeural();
        if (cameraActive) {
          routerRef.current.invalidateSobel();
          detectionsRef.current = [];
          setDetections([]);
        }
        spatialAudio.stopAllImmediate();
        setActiveAudioSource('none');
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [cameraActive]);

  const attachCameraFrameWatcher = useCallback((myCamSession: number) => {
    const vid = videoRef.current as VideoWithVfc | null;
    if (!vid || typeof vid.requestVideoFrameCallback !== 'function') return;
    const onFrame = (now: number) => {
      if (cameraSessionRef.current !== myCamSession) return;
      if (typeof document === 'undefined' || !document.hidden) {
        cameraFrameTrackerRef.current.recordNewFrame(now);
      }
      vfcHandleRef.current = vid.requestVideoFrameCallback!(onFrame);
    };
    vfcHandleRef.current = vid.requestVideoFrameCallback(onFrame);
  }, []);

  useEffect(() => {
    return () => {
      cameraSessionRef.current += 1;
      stopCameraFrameWatcher();
      routerRef.current.invalidateNeural();
      spatialAudio.stopAllImmediate();
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };
  }, [stopCameraFrameWatcher]);

  // Main render + scan + FFT spectrum animation loop
  useEffect(() => {
    let rafId = 0;

    const renderLoop = (timestamp: number) => {
      // P0 (V8.6): On live camera, ONLY update Sobel when a genuinely NEW video frame arrived (never redraw frozen frames!)
      if (cameraActive && videoRef.current && videoRef.current.readyState >= 2) {
        const vid = videoRef.current as VideoWithVfc;
        if (typeof vid.requestVideoFrameCallback !== 'function' && vid.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = vid.currentTime;
          cameraFrameTrackerRef.current.recordNewFrame(timestamp);
        }
        const frameSnap = cameraFrameTrackerRef.current.getSnapshot();
        if (
          frameSnap.latestFrameId > lastSobelCameraFrameIdRef.current &&
          frameSnap.latestFrameCapturedAt !== null
        ) {
          lastSobelCameraFrameIdRef.current = frameSnap.latestFrameId;
          const srcCanvas = sourceCanvasRef.current;
          const sCtx = srcCanvas?.getContext('2d', { willReadFrequently: true });
          if (srcCanvas && sCtx) {
            drawMediaContain(sCtx, videoRef.current, CANVAS_W, CANVAS_H);
            const imgData = sCtx.getImageData(0, 0, CANVAS_W, CANVAS_H);
            const res = processFrameToAcousticMatrix(imgData, MATRIX_COLS, MATRIX_ROWS, edgeThreshold);
            routerRef.current.updateSobel(res.matrix, {
              inputSource: 'camera',
              frameId: frameSnap.latestFrameId,
              capturedAt: frameSnap.latestFrameCapturedAt,
            });
            edgeImageRef.current = res.edgeImageData;
            if (Math.floor(timestamp / 250) !== Math.floor(lastStepTimeRef.current / 250)) {
              setMetrics(res.metrics);
            }
          }
        }
      }

      const pageHidden = typeof document !== 'undefined' ? document.hidden : false;
      const activeFeed = routerRef.current.resolveActive(viewMode, timestamp, pageHidden);

      // P1-02 (V8.7): Stop lingering spatial audio IMMEDIATELY on the very next frame (outside scanSpeedMs throttle)
      if (!activeFeed.matrix && (cameraActive || viewMode === 'depth' || inputSource === 'camera')) {
        spatialAudio.stopAllImmediate();
        setActiveAudioSource((prev) => (prev === 'none' ? prev : 'none'));
      }

      // Advance acoustic scanline ONLY when active acoustic source (Sobel OR Neural) is valid & non-expired
      if (isScanning && activeFeed.matrix && timestamp - lastStepTimeRef.current >= scanSpeedMs) {
        lastStepTimeRef.current = timestamp;
        const col = scanColRef.current;
        const normX = col / (MATRIX_COLS - 1);
        const telemetry = spatialAudio.playScanColumn(activeFeed.matrix[col], normX, scanSpeedMs / 1000, {
          scaleMode,
          useHRTF,
          volume,
        });
        if (col % 2 === 0) {
          setScanTelemetry({
            azimuthDeg: telemetry.azimuthDeg,
            freqHz: telemetry.dominantFreqHz,
          });
        }
        scanColRef.current = (col + 1) % MATRIX_COLS;
      }

      // Draw display canvas
      const disp = displayCanvasRef.current;
      const dCtx = disp?.getContext('2d');
      if (disp && dCtx) {
        if (viewMode === 'source' && sourceCanvasRef.current) {
          dCtx.drawImage(sourceCanvasRef.current, 0, 0, CANVAS_W, CANVAS_H);
        } else if (viewMode === 'sobel' && edgeImageRef.current) {
          dCtx.putImageData(edgeImageRef.current, 0, 0);
        } else if (viewMode === 'depth' && depthHeatmapCanvasRef.current) {
          dCtx.drawImage(depthHeatmapCanvasRef.current, 0, 0, CANVAS_W, CANVAS_H);
        } else {
          // Acoustic matrix view (warm phosphor/amber instrument grid)
          dCtx.fillStyle = '#161512';
          dCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);
          const gridMat = activeFeed.matrix;
          const cw = CANVAS_W / MATRIX_COLS;
          const ch = CANVAS_H / MATRIX_ROWS;
          for (let c = 0; c < MATRIX_COLS; c++) {
            for (let r = 0; r < MATRIX_ROWS; r++) {
              const v = gridMat ? gridMat[c][r] : 0;
              if (v > 0.08) {
                dCtx.fillStyle = `rgba(255, 138, 61, ${Math.min(1, v * 1.15)})`;
                dCtx.fillRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
              } else {
                dCtx.fillStyle = 'rgba(243, 237, 226, 0.03)';
                dCtx.fillRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
              }
            }
          }
        }

        // Draw crisp amber scanline overlay
        if (isScanning) {
          const sx = (scanColRef.current / MATRIX_COLS) * CANVAS_W;
          dCtx.fillStyle = 'rgba(255, 138, 61, 0.14)';
          dCtx.fillRect(sx - 18, 0, 18, CANVAS_H);

          dCtx.strokeStyle = '#ff8a3d';
          dCtx.lineWidth = 2;
          dCtx.beginPath();
          dCtx.moveTo(sx, 0);
          dCtx.lineTo(sx, CANVAS_H);
          dCtx.stroke();
        }

        // Draw On-Device AI Spatial Bounding Boxes
        if (aiEnabled && detectionsRef.current.length > 0) {
          detectionsRef.current.forEach((det) => {
            const [bx, by, bw, bh] = det.bbox;
            const isLockedTarget =
              finderTargetRef.current !== '' &&
              (det.labelEn === finderTargetRef.current ||
                (finderTargetRef.current === 'cup' && det.labelEn === 'bottle'));
            const color = isLockedTarget
              ? '#ff8a3d'
              : det.distance === '近距离'
                ? '#f2b84b'
                : '#9fc9b0';
            dCtx.strokeStyle = color;
            dCtx.lineWidth = isLockedTarget ? 2.6 : 1.8;
            dCtx.strokeRect(bx, by, bw, bh);

            const prefix = isLockedTarget ? '[锁定] ' : '';
            const tag = `${prefix}${det.direction} · ${det.distanceText} · ${det.labelZh}`;
            dCtx.font = '600 11px "JetBrains Mono", Consolas, monospace';
            const tw = dCtx.measureText(tag).width + 10;
            const tx = Math.max(4, Math.min(CANVAS_W - tw - 4, bx));
            // Avoid overlapping the top-left status overlay pill
            const inTopLeftOverlay = by < 46 && bx < 260;
            const ty = inTopLeftOverlay
              ? Math.min(CANVAS_H - 28, by + bh + 18)
              : Math.max(18, by);
            dCtx.fillStyle = 'rgba(22, 21, 18, 0.88)';
            dCtx.fillRect(tx, ty - 18, tw, 18);
            dCtx.fillStyle = color;
            dCtx.fillText(tag, tx + 5, ty - 5);
          });
        }

        // Draw Interactive Acoustic Point-Probe Crosshair
        const pt = probePointRef.current;
        if (pt && timestamp - pt.at < 1200) {
          const alpha = Math.max(0, 1 - (timestamp - pt.at) / 1200);
          dCtx.strokeStyle = `rgba(255, 138, 61, ${alpha})`;
          dCtx.lineWidth = 1.8;
          dCtx.beginPath();
          dCtx.arc(pt.x, pt.y, 12 + (1 - alpha) * 12, 0, Math.PI * 2);
          dCtx.stroke();
          dCtx.fillStyle = `rgba(243, 237, 226, ${alpha})`;
          dCtx.beginPath();
          dCtx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
          dCtx.fill();
        }
      }

      // Draw real-time FFT Spectrum (solid phosphor bars + 2px amber cap)
      const specCanvas = spectrumCanvasRef.current;
      const specCtx = specCanvas?.getContext('2d');
      if (specCanvas && specCtx) {
        const data = spatialAudio.getSpectrumData();
        const w = specCanvas.width;
        const h = specCanvas.height;
        specCtx.fillStyle = '#161512';
        specCtx.fillRect(0, 0, w, h);

        // Subtle baseline rule
        specCtx.strokeStyle = 'rgba(243, 237, 226, 0.08)';
        specCtx.lineWidth = 1;
        specCtx.beginPath();
        specCtx.moveTo(0, h - 1);
        specCtx.lineTo(w, h - 1);
        specCtx.stroke();

        const bars = 40;
        const barW = w / bars;
        for (let i = 0; i < bars; i++) {
          const val = data[i] || 0;
          const barH = Math.max(2, (val / 255) * (h - 8));
          const x = i * barW + 1.5;
          const bw = Math.max(2, barW - 3);
          const y = h - barH;
          specCtx.fillStyle = 'rgba(243, 237, 226, 0.82)';
          specCtx.fillRect(x, y, bw, barH);
          if (val > 18) {
            specCtx.fillStyle = '#ff8a3d';
            specCtx.fillRect(x, y, bw, 2);
          }
        }
      }

      rafId = requestAnimationFrame(renderLoop);
    };

    rafId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(rafId);
  }, [isScanning, scanSpeedMs, scaleMode, useHRTF, volume, viewMode, cameraActive, edgeThreshold, aiEnabled]);

  // Run EchoDepthNet ONNX real-time neural depth & 48x16 acoustic hazard matrix when 'depth' view is active
  useEffect(() => {
    const myGen = routerRef.current.beginGeneration();

    if (viewMode !== 'depth') {
      setActiveAudioSource('sobel');
      setDepthMs(null);
      return;
    }
    setActiveAudioSource('none');
    let cancelled = false;

    const runOnce = async () => {
      if (cancelled || depthBusyRef.current) return;
      if (typeof document !== 'undefined' && document.hidden) {
        routerRef.current.invalidateNeural();
        spatialAudio.stopAllImmediate();
        setActiveAudioSource('none');
        return;
      }
      const src = sourceCanvasRef.current;
      if (!src) return;

      // P0-1 (V8.5): On live camera, MUST claim a genuinely NEW, fresh video frame before running ONNX!
      let frameId = 1;
      let capturedAt = performance.now();
      if (cameraActive) {
        const claimed = cameraFrameTrackerRef.current.claimNextFrameForInference(capturedAt, 2500);
        if (!claimed) {
          // Camera has not delivered a new frame -> NEVER re-infer old frame or refresh TTL!
          const cur = routerRef.current.resolveActive('depth', capturedAt, false);
          if (!cur.matrix) {
            spatialAudio.stopAllImmediate();
            setActiveAudioSource('none');
            setDepthStatus('摄像头画面未推进或已冻结，已停止过期神经声场');
          }
          return;
        }
        frameId = claimed.frameId;
        capturedAt = claimed.capturedAt;
      }

      depthBusyRef.current = true;
      const startedAt = performance.now();
      try {
        const out = await runEchoDepthInference(src);
        const completedAt = performance.now();
        if (cancelled) return;
        if (!out) {
          routerRef.current.invalidateNeural();
          spatialAudio.stopAllImmediate();
          setActiveAudioSource('none');
          return;
        }
        const committed = routerRef.current.commitNeuralPacket(
          {
            matrix: out.acousticMatrix,
            inputSource,
            frameId,
            generation: myGen,
            capturedAt,
            startedAt,
            completedAt,
            // Live camera TTL anchored strictly to raw frame capture time T0 (never extended by re-inference)
            expiresAt: cameraActive ? capturedAt + 3000 : Number.POSITIVE_INFINITY,
            modelVersion: ECHO_DEPTH_MODEL_VERSION,
          },
          completedAt
        );
        if (!committed.accepted) {
          if (cameraActive) {
            spatialAudio.stopAllImmediate();
            setActiveAudioSource('none');
          }
          return;
        }

        if (!depthHeatmapCanvasRef.current) {
          const hc = document.createElement('canvas');
          hc.width = 256;
          hc.height = 256;
          depthHeatmapCanvasRef.current = hc;
        }
        const hCtx = depthHeatmapCanvasRef.current.getContext('2d');
        hCtx?.putImageData(out.depthHeatmap, 0, 0);
        setDepthMs(out.inferenceMs);
        setActiveAudioSource('neural_echodepth');
      } finally {
        depthBusyRef.current = false;
      }
    };

    void loadEchoDepthOnnxModel((msg) => {
      if (!cancelled) setDepthStatus(msg);
    }).then((ok) => {
      if (ok && !cancelled) void runOnce();
      else if (!ok && !cancelled) {
        routerRef.current.invalidateNeural();
        setActiveAudioSource('none');
      }
    });

    const onVisibility = () => {
      if (document.hidden) {
        routerRef.current.invalidateNeural();
        spatialAudio.stopAllImmediate();
        setActiveAudioSource('none');
      }
    };
    document.addEventListener('visibilitychange', onVisibility);

    const timer = window.setInterval(() => {
      if (cameraActive) void runOnce();
    }, 450);

    return () => {
      cancelled = true;
      document.removeEventListener('visibilitychange', onVisibility);
      window.clearInterval(timer);
    };
  }, [viewMode, inputSource, preset, uploadVersion, cameraActive]);

  // Run On-Device AI detection when enabled (P0-01 V8.9: PerceptionPacket-bound single-flight inference)
  useEffect(() => {
    if (!aiEnabled) {
      aiPacketRef.current = null;
      detectionsRef.current = [];
      setDetections([]);
      return;
    }

    let cancelled = false;
    const runDetect = async () => {
      if (aiInFlightRef.current) return;
      const srcCanvas = sourceCanvasRef.current;
      if (!srcCanvas) return;
      const startedAt = performance.now();
      const mySession = cameraSessionRef.current;
      let frameId = 0;
      let capturedAt = startedAt;

      // P0-01 (V8.9): Snapshot the EXACT frame identity (sessionId, frameId, capturedAt) BEFORE inference
      if (inputSource === 'camera') {
        if (!isLiveCameraFrameFresh(startedAt)) {
          aiPacketRef.current = null;
          detectionsRef.current = [];
          setDetections([]);
          return;
        }
        const snap = cameraFrameTrackerRef.current.getSnapshot();
        if (
          snap.latestFrameCapturedAt === null ||
          snap.latestFrameId <= lastAiCameraFrameIdRef.current
        ) {
          return;
        }
        frameId = snap.latestFrameId;
        capturedAt = snap.latestFrameCapturedAt;
        lastAiCameraFrameIdRef.current = frameId;
      }

      aiInFlightRef.current = true;
      try {
        const isProcedural = inputSource === 'preset';
        const res = await detectSpatialObjects(srcCanvas, isProcedural, preset);
        const completedAt = performance.now();
        if (cancelled) return;

        if (inputSource === 'camera') {
          const pageHidden = typeof document !== 'undefined' && document.hidden;
          const ownFrameAge = completedAt - capturedAt;
          // Reject stale session, hidden page, own-frame TTL expiry, or out-of-order late commit
          if (
            pageHidden ||
            mySession !== cameraSessionRef.current ||
            !cameraActive ||
            cameraDisconnected ||
            ownFrameAge > 3000 ||
            frameId <= lastCommittedAiFrameIdRef.current
          ) {
            if (ownFrameAge > 3000 || !isLiveCameraFrameFresh(completedAt)) {
              aiPacketRef.current = null;
              detectionsRef.current = [];
              setDetections([]);
            }
            return;
          }
          lastCommittedAiFrameIdRef.current = frameId;
        }

        aiPacketRef.current = {
          sessionId: mySession,
          inputSource,
          frameId,
          capturedAt,
          startedAt,
          completedAt,
          expiresAt: inputSource === 'camera' ? capturedAt + 3000 : Number.POSITIVE_INFINITY,
          detections: res,
        };
        detectionsRef.current = res;
        setDetections(res);

        // 3D Item-Finder Beacon: ping in 3D space when the target item is spotted
        const target = finderTargetRef.current;
        if (target && volume > 0.001) {
          const hit = res.find(
            (d) => d.labelEn === target || (target === 'cup' && d.labelEn === 'bottle')
          );
          if (hit) {
            const [bx, , bw] = hit.bbox;
            const normX = Math.max(0, Math.min(1, (bx + bw / 2) / CANVAS_W));
            const centeredness = 1 - Math.abs(normX - 0.5) * 2; // 0..1
            const distBoost = hit.distance === '近距离' ? 0.35 : hit.distance === '中距离' ? 0.15 : 0;
            const closeness = Math.min(1, centeredness * 0.65 + distBoost);
            if (completedAt - lastFinderPingRef.current > 750) {
              lastFinderPingRef.current = completedAt;
              spatialAudio.playFinderBeacon(normX, closeness, volume);
            }
          }
        }

        if (autoSpeakAI) {
          const m = metricsRef.current;
          const hasDenseObstacle = m.centerDensity > 42 || m.leftDensity > 48 || m.rightDensity > 48;
          speakSpatialDetections(res, false, hasDenseObstacle ? m.navigationCue : undefined);
        }
      } finally {
        aiInFlightRef.current = false;
      }
    };

    runDetect();
    const timer = window.setInterval(runDetect, cameraActive ? 900 : 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [aiEnabled, cameraActive, cameraDisconnected, preset, inputSource, uploadVersion, autoSpeakAI, isLiveCameraFrameFresh, volume]);

  const announceSceneNow = useCallback(() => {
    const now = performance.now();
    const usable = getUsableAiDetections(now);
    if (inputSource === 'camera' && usable === null) {
      speakCue('摄像头画面已冻结、过期或断开，实时感知不可用，请重新连接摄像头后再查询。');
      return;
    }
    const dets = usable ?? [];
    const m = metricsRef.current;
    if (dets.length > 0) {
      speakSpatialDetections(dets, true, m.navigationCue);
    } else {
      speakCue(
        `${m.navigationCue}。左侧障碍密度${m.leftDensity}%，正前方${m.centerDensity}%，右侧${m.rightDensity}%。`
      );
    }
  }, [getUsableAiDetections, inputSource]);

  const runLightAndColorCheck = useCallback(() => {
    if (inputSource === 'camera' && !isLiveCameraFrameFresh(performance.now())) {
      speakCue('摄像头画面已冻结或断开，无法测量当前环境光线与颜色。');
      return;
    }
    const srcCanvas = sourceCanvasRef.current;
    if (!srcCanvas) return;
    const report = analyzeLightAndColor(srcCanvas);
    setLightReport(report);
    speakCue(report.speechText);
  }, [inputSource, isLiveCameraFrameFresh]);

  const selectFinderTarget = useCallback(
    async (targetId: string, label?: string) => {
      spatialAudio.ensureContext();
      if (!targetId || finderTargetRef.current === targetId) {
        setFinderTargetId('');
        speakCue('已关闭 3D 寻物雷达');
        return;
      }
      setFinderTargetId(targetId);
      const targetLabel =
        label || FINDER_TARGETS.find((t) => t.id === targetId)?.label || targetId;
      if (!aiEnabled) {
        setAiLoading(true);
        const { ready } = await loadOnDeviceAIModel((msg) => setAiStatus(msg));
        setAiLoading(false);
        if (ready) setAiEnabled(true);
      }
      speakCue(`已开启 3D 寻物雷达，正在寻找${targetLabel}。发现后会从对应方向发出滴答信标。`);
    },
    [aiEnabled]
  );

  const startVoiceCommand = useCallback(() => {
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
      const msg = '当前浏览器暂不支持语音识别，可使用快捷键 R 播报画面或直接点击按钮。';
      setVoiceFeedback(msg);
      speakCue(msg);
      return;
    }

    const rec = new SpeechRec();
    rec.lang = 'zh-CN';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onstart = () => {
      setIsListeningVoice(true);
      setVoiceFeedback('正在聆听指令… 可说“前面有什么 / 帮我找水杯 / 开灯了吗 / 什么颜色”');
    };
    rec.onend = () => {
      setIsListeningVoice(false);
    };
    rec.onerror = () => {
      setIsListeningVoice(false);
      setVoiceFeedback('未听清语音指令，请重试或按 R 键播报画面。');
    };
    rec.onresult = (ev) => {
      const transcript = ev.results?.[0]?.[0]?.transcript || '';
      const intent = parseVoiceCommand(transcript);
      setVoiceFeedback(`识别到指令：「${transcript}」`);

      if (intent.type === 'read_scene') {
        announceSceneNow();
      } else if (intent.type === 'light_color') {
        runLightAndColorCheck();
      } else if (intent.type === 'find_item') {
        selectFinderTarget(intent.targetId, intent.targetLabel);
      } else if (intent.type === 'clear_finder') {
        setFinderTargetId('');
        speakCue('已取消寻物追踪');
      } else if (intent.type === 'start_scan') {
        spatialAudio.ensureContext();
        setIsScanning(true);
        speakCue('已启动 3D 声波扫描');
      } else if (intent.type === 'stop_scan') {
        setIsScanning(false);
        speakCue('已暂停 3D 声波扫描');
      } else {
        speakCue(`收到${transcript}，你可以说“前面有什么”“帮我找水杯”或“开灯了吗”`);
      }
    };
    try {
      rec.start();
    } catch {
      setIsListeningVoice(false);
    }
  }, [announceSceneNow, runLightAndColorCheck, selectFinderTarget]);

  // Eyes-free keyboard shortcuts: Space (scan), R (read scene), M (mute/unmute), V (voice command)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLButtonElement ||
        e.metaKey ||
        e.ctrlKey ||
        e.altKey
      ) {
        return;
      }
      if (e.key === ' ') {
        e.preventDefault();
        spatialAudio.ensureContext();
        setIsScanning((prev) => {
          const next = !prev;
          if (assistMode) speakCue(next ? '3D 声波扫描已开启' : '3D 声波扫描已暂停');
          return next;
        });
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        announceSceneNow();
      } else if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        startVoiceCommand();
      } else if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        setVolume((prev) => {
          if (prev > 0.05) {
            savedVolumeRef.current = prev;
            speakCue('声场已静音');
            return 0;
          }
          speakCue('声场已恢复');
          return savedVolumeRef.current || 0.7;
        });
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [announceSceneNow, assistMode, startVoiceCommand]);

  const probeAtEvent = (e: React.PointerEvent<HTMLCanvasElement>, forceSound = false) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const normX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const normY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    const col = Math.min(MATRIX_COLS - 1, Math.floor(normX * MATRIX_COLS));
    const row = Math.min(MATRIX_ROWS - 1, Math.floor(normY * MATRIX_ROWS));
    const activeMat = routerRef.current.resolveActive(viewMode, performance.now(), false).matrix;
    const energyPct = Math.round((activeMat?.[col]?.[row] ?? 0) * 100);

    probePointRef.current = { x: normX * CANVAS_W, y: normY * CANVAS_H, at: performance.now() };

    // P1-04 (V8.9): Respect mute (volume === 0) and block perception probe tones when live matrix is invalid/expired
    if (volume <= 0.001) return;
    if (!activeMat && (inputSource === 'camera' || viewMode === 'depth')) return;

    const now = performance.now();
    if (forceSound || now - lastProbeSoundRef.current > 65) {
      lastProbeSoundRef.current = now;
      const info = spatialAudio.playBrushTone(normX, normY, {
        scaleMode,
        useHRTF,
        volume: Math.min(1, volume * (0.4 + (energyPct / 100) * 0.7)),
      });
      setProbeHud({ ...info, energyPct });
    }
  };

  const toggleBlindAssistMode = async () => {
    spatialAudio.ensureContext();
    if (assistMode) {
      // P1-01 (V8.9): Synchronously cancel any pending AI model load or getUserMedia request on exit
      assistSessionRef.current += 1;
      cameraSessionRef.current += 1;
      cameraStartingRef.current = false;
      aiPacketRef.current = null;
      lastCommittedAiFrameIdRef.current = 0;
      setAssistMode(false);
      setAutoSpeakAI(false);
      setIsScanning(false);
      stopCameraFrameWatcher();
      spatialAudio.stopAllImmediate();
      if (cameraActive && videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
        videoRef.current.srcObject = null;
        setCameraActive(false);
        setInputSource(uploadedImgRef.current ? 'upload' : 'preset');
      }
      speakCue('已退出盲人助行模式');
      return;
    }

    const myAssistSession = ++assistSessionRef.current;
    setAssistMode(true);
    setAutoSpeakAI(true);
    setVolume((v) => (v < 0.2 ? 0.55 : v));
    speakCue('正在启动盲人助行模式，将开启识别与语音播报。按 R 键随时听取面前物体，按空格键开关 3D 声场。');

    // 1. Load on-device AI if not yet enabled
    if (!aiEnabled) {
      setAiLoading(true);
      const { ready } = await loadOnDeviceAIModel((msg) => setAiStatus(msg));
      setAiLoading(false);
      if (assistSessionRef.current !== myAssistSession) return;
      if (ready) setAiEnabled(true);
    }

    if (assistSessionRef.current !== myAssistSession) return;

    // 2. Try enabling live camera; if unavailable, seamlessly keep current scene
    if (!cameraActive && !cameraStartingRef.current) {
      cameraStartingRef.current = true;
      const myCamSession = ++cameraSessionRef.current;
      let acquiredStream: MediaStream | null = null;
      try {
        acquiredStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: 480, height: 300 },
          audio: false,
        });
        if (cameraSessionRef.current !== myCamSession) {
          acquiredStream.getTracks().forEach((t) => t.stop());
          return;
        }
        stopCameraFrameWatcher();
        const track = acquiredStream.getVideoTracks()[0];
        if (track) {
          track.onended = () => {
            if (cameraSessionRef.current !== myCamSession) return;
            // P0-01 (V8.7): Enter explicit UNAVAILABLE — NEVER fall back to virtual preset!
            stopCameraFrameWatcher();
            routerRef.current.invalidateSobel();
            routerRef.current.invalidateNeural();
            spatialAudio.stopAllImmediate();
            aiPacketRef.current = null;
            lastCommittedAiFrameIdRef.current = 0;
            detectionsRef.current = [];
            setDetections([]);
            setIsScanning(false);
            setActiveAudioSource('none');
            setCameraActive(false);
            setCameraDisconnected(true);
            setCameraError('摄像头视频流已断开，实时感知不可用（已停止声场，严禁自动切回模拟场景）。');
            speakCue('摄像头视频流已断开，实时感知不可用，已停止声场。', 1.04, 'EMERGENCY');
          };
        }
        uploadedImgRef.current = null;
        if (videoRef.current) {
          videoRef.current.srcObject = acquiredStream;
          await videoRef.current.play();
        }
        if (cameraSessionRef.current !== myCamSession) {
          acquiredStream.getTracks().forEach((t) => t.stop());
          return;
        }
        attachCameraFrameWatcher(myCamSession);
        setCameraDisconnected(false);
        setCameraActive(true);
        setInputSource('camera');
      } catch {
        if (acquiredStream) {
          acquiredStream.getTracks().forEach((t) => t.stop());
        }
        setCameraError('未检测到可用摄像头，盲人助行模式已自动切换为当前场景演示。');
        speakCue('未检测到摄像头，已切换为当前演示场景进行语音播报。');
      } finally {
        cameraStartingRef.current = false;
      }
    }

    if (assistSessionRef.current !== myAssistSession) return;
    setIsScanning(true);
  };

  const toggleAI = async () => {
    if (aiEnabled) {
      setAiEnabled(false);
      setAutoSpeakAI(false);
      setAiStatus('');
      return;
    }
    setAiLoading(true);
    const { ready } = await loadOnDeviceAIModel((msg) => setAiStatus(msg));
    setAiLoading(false);
    if (ready) {
      setAiEnabled(true);
    }
  };

  const toggleScan = () => {
    spatialAudio.ensureContext();
    setIsScanning((prev) => !prev);
  };

  const toggleCamera = async () => {
    setCameraError(null);
    if (cameraActive) {
      cameraSessionRef.current += 1;
      cameraStartingRef.current = false;
      stopCameraFrameWatcher();
      routerRef.current.invalidateNeural();
      spatialAudio.stopAllImmediate();
      detectionsRef.current = [];
      setDetections([]);
      setCameraDisconnected(false);
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      }
      setCameraActive(false);
      setInputSource(uploadedImgRef.current ? 'upload' : 'preset');
      return;
    }

    // P1-01 (V8.9): Clicking while STARTING explicitly cancels the pending getUserMedia request!
    if (cameraStartingRef.current) {
      cameraSessionRef.current += 1;
      cameraStartingRef.current = false;
      setCameraError('已取消摄像头启动请求。');
      return;
    }
    cameraStartingRef.current = true;
    const myCamSession = ++cameraSessionRef.current;
    aiPacketRef.current = null;
    lastCommittedAiFrameIdRef.current = 0;
    let acquiredStream: MediaStream | null = null;
    try {
      acquiredStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 480, height: 300 },
        audio: false,
      });
      if (cameraSessionRef.current !== myCamSession) {
        acquiredStream.getTracks().forEach((t) => t.stop());
        return;
      }
      stopCameraFrameWatcher();
      const track = acquiredStream.getVideoTracks()[0];
      if (track) {
        track.onended = () => {
          if (cameraSessionRef.current !== myCamSession) return;
          // P0-01 (V8.7): Enter explicit UNAVAILABLE — NEVER fall back to virtual preset!
          stopCameraFrameWatcher();
          routerRef.current.invalidateSobel();
          routerRef.current.invalidateNeural();
          spatialAudio.stopAllImmediate();
          detectionsRef.current = [];
          setDetections([]);
          setIsScanning(false);
          setActiveAudioSource('none');
          setCameraActive(false);
          setCameraDisconnected(true);
          setCameraError('摄像头视频流已断开，实时感知不可用（已停止声场，严禁自动切回模拟场景）。');
          speakCue('摄像头视频流已断开，实时感知不可用，已停止声场。', 1.04, 'EMERGENCY');
        };
      }
      uploadedImgRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = acquiredStream;
        await videoRef.current.play();
      }
      if (cameraSessionRef.current !== myCamSession) {
        acquiredStream.getTracks().forEach((t) => t.stop());
        return;
      }
      attachCameraFrameWatcher(myCamSession);
      setCameraDisconnected(false);
      setCameraActive(true);
      setInputSource('camera');
      spatialAudio.ensureContext();
      setIsScanning(true);
    } catch {
      if (acquiredStream) {
        acquiredStream.getTracks().forEach((t) => t.stop());
      }
      setCameraError('无法访问摄像头（可能未授权或无设备），您可以继续使用内置场景或上传图片体验。');
    } finally {
      cameraStartingRef.current = false;
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        if (cameraActive) toggleCamera();
        setCameraDisconnected(false);
        setCameraError(null);
        uploadedImgRef.current = img;
        setInputSource('upload');
        setUploadVersion((v) => v + 1);
        recomputeCurrentFrame();
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  };

  const activeRadarAngle = isScanning
    ? scanTelemetry.azimuthDeg
    : probeHud
      ? probeHud.azimuthDeg
      : null;
  const beamRad = activeRadarAngle !== null ? (activeRadarAngle * Math.PI) / 180 : 0;
  const radarBx = 140 + Math.sin(beamRad) * 96;
  const radarBy = 112 - Math.cos(beamRad) * 96;

  return (
    <div className="grid gap-6 lg:grid-cols-12 items-start">
      {/* Hidden source canvas & video */}
      <canvas ref={sourceCanvasRef} width={CANVAS_W} height={CANVAS_H} className="hidden" />
      <video ref={videoRef} playsInline muted className="hidden" />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="hidden"
      />

      {/* Left 7 Cols: Visual Viewport + Presets + Spectrum */}
      <div className="lg:col-span-7 space-y-5">
        <div className="panel p-5 space-y-4">
          {/* Header & View Mode Tabs */}
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="eyebrow">01 · Scanner</div>
              <h2 className="title-serif text-[20px] mt-1">实时声画扫描</h2>
              <p className="text-[13px] text-ink-2 mt-1">
                水平位置决定左右声像，垂直高度决定音高。
              </p>
            </div>

            <div className="seg">
              {[
                { id: 'source' as ViewMode, label: '原始画面' },
                { id: 'sobel' as ViewMode, label: 'Sobel 边缘' },
                { id: 'matrix' as ViewMode, label: '声学矩阵' },
                { id: 'depth' as ViewMode, label: '神经深度' },
              ].map((m) => (
                <button
                  key={m.id}
                  type="button"
                  data-active={viewMode === m.id}
                  onClick={() => setViewMode(m.id)}
                  className="seg-item"
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Main Canvas inside dark instrument screen */}
          <div className="screen">
            <canvas
              ref={displayCanvasRef}
              width={CANVAS_W}
              height={CANVAS_H}
              onPointerDown={(e) => {
                e.currentTarget.setPointerCapture(e.pointerId);
                isProbingRef.current = true;
                probeAtEvent(e, true);
              }}
              onPointerMove={(e) => {
                if (isProbingRef.current) probeAtEvent(e, false);
              }}
              onPointerUp={() => {
                isProbingRef.current = false;
              }}
              onPointerLeave={() => {
                isProbingRef.current = false;
              }}
              className="w-full h-auto block touch-none cursor-crosshair"
            />
            <div className="pointer-events-none absolute top-3 left-3 flex flex-wrap items-center gap-2">
              <span className="screen-label">
                {inputSource === 'camera'
                  ? '真实摄像头'
                  : inputSource === 'upload'
                    ? '本地图片'
                    : '模拟演示场景'}
              </span>
              <span className="screen-label inline-flex items-center gap-1.5">
                <span className={`led ${isScanning ? 'led-on' : ''}`} />
                {isScanning ? 'HRTF 扫描中' : '待机'}
              </span>
              {isScanning && (
                <span className="screen-label">
                  {scanTelemetry.azimuthDeg > 0 ? `+${scanTelemetry.azimuthDeg}` : scanTelemetry.azimuthDeg}°
                  {scanTelemetry.freqHz ? ` · ${scanTelemetry.freqHz} Hz` : ''}
                </span>
              )}
              {viewMode === 'depth' && (
                <span className="screen-label text-amber">
                  {activeAudioSource === 'neural_echodepth' && depthMs !== null
                    ? `声场源：EchoDepthNet ONNX · ${depthMs}ms`
                    : depthStatus || 'EchoDepthNet 神经声场加载中…'}
                </span>
              )}
              {viewMode !== 'depth' && (
                <span className="screen-label">声场源：Sobel 边缘矩阵</span>
              )}
            </div>

            <div className="pointer-events-none absolute bottom-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2">
              <span className="screen-label inline-flex items-center gap-1.5">
                <Crosshair className="h-3 w-3 text-amber" />
                点击或拖动画面任意处点探 3D 音高
              </span>
              {probeHud && (
                <span className="screen-label text-amber">
                  {probeHud.azimuthDeg > 0 ? `+${probeHud.azimuthDeg}` : probeHud.azimuthDeg}° ·{' '}
                  {probeHud.freqHz} Hz ({probeHud.noteName}) · {probeHud.energyPct}%
                </span>
              )}
            </div>
          </div>

          {/* Eyes-Free Blind Assist Quick Bar */}
          <div className="panel-sunken p-3.5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2.5">
              <div className="flex items-center gap-2 text-[12.5px] text-ink-2">
                <Accessibility className="h-4 w-4 text-signal-ink shrink-0" />
                <span>
                  <strong className="text-ink">盲人日常助行</strong>：免视全语音 · 3D 寻物 · 测光识色 · 红绿灯识别
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={announceSceneNow}
                  className="btn btn-sm"
                  title="快捷键 R：立即语音播报当前画面内容与障碍方位"
                >
                  <Volume2 className="h-3.5 w-3.5" />
                  听画面 <span className="kbd ml-0.5">R</span>
                </button>
                <button
                  type="button"
                  onClick={startVoiceCommand}
                  className={`btn btn-sm ${isListeningVoice ? 'btn-signal' : ''}`}
                  title="快捷键 V：语音说“前面有什么 / 帮我找水杯 / 开灯了吗”"
                >
                  {isListeningVoice ? <MicOff className="h-3.5 w-3.5" /> : <Mic className="h-3.5 w-3.5" />}
                  {isListeningVoice ? '聆听中...' : '语音指令'}
                  {!isListeningVoice && <span className="kbd ml-0.5">V</span>}
                </button>
                <button
                  type="button"
                  onClick={runLightAndColorCheck}
                  className="btn btn-sm"
                  title="判断室内是否开灯，并朗读画面正中的物体/衣物颜色"
                >
                  <Sun className="h-3.5 w-3.5" />
                  测光识色
                </button>
                <button
                  type="button"
                  onClick={toggleBlindAssistMode}
                  className={`btn btn-sm ${assistMode ? 'btn-signal' : 'btn-primary'}`}
                >
                  {assistMode ? '退出助行' : '一键助行'}
                </button>
              </div>
            </div>

            {/* 3D Item-Finder Beacon Target Selector */}
            <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-line">
              <span className="eyebrow inline-flex items-center gap-1 mr-1 text-ink-2">
                <Target className="h-3 w-3 text-signal-ink" />
                3D 寻物信标
              </span>
              {FINDER_TARGETS.map((t) => {
                const active = finderTargetId === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => selectFinderTarget(t.id, t.label)}
                    className={`btn btn-sm !h-7 !px-2.5 !text-[12px] ${active ? 'btn-active' : ''}`}
                  >
                    找{t.label}
                  </button>
                );
              })}
              {finderTargetId && (
                <button
                  type="button"
                  onClick={() => selectFinderTarget('')}
                  className="text-[11.5px] text-ink-3 hover:text-ink underline ml-1"
                >
                  取消锁定
                </button>
              )}
            </div>

            {(lightReport || voiceFeedback) && (
              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-line text-[12px] text-ink-2">
                {lightReport && (
                  <span className="inline-flex items-center gap-2">
                    <span
                      className="inline-block h-3.5 w-3.5 rounded-full border border-line-strong shrink-0"
                      style={{ backgroundColor: lightReport.centerHex }}
                    />
                    <span>
                      亮度 <strong className="num text-ink">{lightReport.brightnessPct}%</strong>（{lightReport.lampHint}）·
                      中心颜色 <strong className="text-ink">{lightReport.colorName}</strong>
                    </span>
                  </span>
                )}
                {voiceFeedback && <span className="text-signal-ink">{voiceFeedback}</span>}
              </div>
            )}
          </div>

          {/* Primary Action Bar */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={toggleScan}
              className={`btn btn-lg flex-1 min-w-[150px] ${isScanning ? 'btn-signal' : 'btn-primary'}`}
            >
              {isScanning ? <Pause className="h-4 w-4 shrink-0" /> : <Play className="h-4 w-4 shrink-0" />}
              {isScanning ? '暂停声波扫描' : '启动声波扫描'}
            </button>

            <button
              type="button"
              onClick={toggleCamera}
              className={`btn btn-lg ${cameraActive ? 'btn-active' : ''}`}
            >
              {cameraActive ? <CameraOff className="h-4 w-4 shrink-0" /> : <Camera className="h-4 w-4 shrink-0" />}
              {cameraActive ? '关闭摄像头' : '实时摄像头'}
            </button>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="btn btn-lg"
            >
              <Upload className="h-4 w-4 shrink-0" />
              上传图片
            </button>

            <button
              type="button"
              disabled={aiLoading}
              onClick={toggleAI}
              className={`btn btn-lg ${aiEnabled ? 'btn-active' : ''}`}
            >
              <Brain className="h-4 w-4 shrink-0" />
              {aiLoading ? '加载端侧 AI...' : aiEnabled ? '端侧 AI：开' : '端侧 AI 识别'}
            </button>
          </div>

          {aiStatus && (
            <div className="rounded-lg border border-line bg-sunken px-3.5 py-2.5 text-[12.5px] text-ink-2 flex flex-wrap items-center justify-between gap-2">
              <span className="inline-flex items-center gap-2">
                <span className={`led ${aiEnabled ? 'led-ok' : 'led-on'}`} />
                {aiStatus}
              </span>
              <div className="flex items-center gap-2">
                {aiEnabled && (
                  <button
                    type="button"
                    onClick={() => setAutoSpeakAI((v) => !v)}
                    className={`btn btn-sm ${autoSpeakAI ? 'btn-active' : ''}`}
                  >
                    {autoSpeakAI ? '自动播报：开' : '自动播报：关'}
                  </button>
                )}
                {detections.length > 0 && (
                  <button
                    type="button"
                    onClick={announceSceneNow}
                    className="btn btn-sm"
                  >
                    <Volume2 className="h-3.5 w-3.5" />
                    语音播报
                  </button>
                )}
              </div>
            </div>
          )}

          {aiEnabled && detections.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {detections.map((d, idx) => (
                <span key={idx} className="chip">
                  <span className="font-semibold text-signal-ink">{d.direction}</span>
                  <span>·</span>
                  <span className="num text-ink">{d.distanceText}</span>
                  <span>·</span>
                  <strong className="text-ink">{d.labelZh}</strong>
                  <span className="num text-[11px] text-ink-3">{Math.round(d.score * 100)}%</span>
                </span>
              ))}
            </div>
          )}

          {cameraError && (
            <div className="rounded-lg border border-warn/40 bg-warn-soft px-3.5 py-2 text-[12.5px] text-warn">
              {cameraError}
            </div>
          )}

          {/* Scene Presets */}
          <div className="pt-1 space-y-2">
            <div className="flex items-center justify-between">
              <span className="eyebrow">内置场景</span>
              <span className="text-[11.5px] text-ink-3">无需摄像头即可测试</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {PRESETS.map((p) => {
                const active = inputSource === 'preset' && preset === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    data-active={active}
                    onClick={() => {
                      if (cameraActive) toggleCamera();
                      setCameraDisconnected(false);
                      setCameraError(null);
                      uploadedImgRef.current = null;
                      setInputSource('preset');
                      setPreset(p.id);
                    }}
                    className="tile"
                  >
                    <div className="text-[12.5px] font-semibold text-ink">{p.label}</div>
                    <div className="text-[11px] text-ink-3 mt-0.5 truncate">{p.desc}</div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Live FFT Spectrum */}
        <div className="panel p-4 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-[13px] font-semibold text-ink">实时合成音频频谱</span>
            <span className="eyebrow">
              {useHRTF ? 'HRTF 3D' : 'Stereo'} · {MATRIX_COLS}×{MATRIX_ROWS}
            </span>
          </div>
          <div className="screen">
            <canvas
              ref={spectrumCanvasRef}
              width={480}
              height={72}
              className="w-full h-[72px] block"
            />
          </div>
        </div>
      </div>

      {/* Right 5 Cols: Spatial Obstacle HUD + Algorithm Parameter Lab */}
      <div className="lg:col-span-5 space-y-5">
        {/* Spatial Obstacle Navigation HUD */}
        <div className="panel p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="eyebrow">Binaural Radar</div>
              <h3 className="text-[15px] font-semibold text-ink mt-0.5">双耳方位雷达与拓扑</h3>
            </div>
            <span className="chip num">边缘 {metrics.edgeCoverage}%</span>
          </div>

          {/* 180-deg Live Binaural Azimuth Radar */}
          <div className="screen px-3 py-2">
            <svg
              viewBox="0 0 280 128"
              className="w-full h-[116px] block"
              role="img"
              aria-label="3D 双耳声场方位雷达"
            >
              {/* Concentric range arcs */}
              {[34, 66, 98].map((r) => (
                <path
                  key={r}
                  d={`M ${140 - r} 112 A ${r} ${r} 0 0 1 ${140 + r} 112`}
                  fill="none"
                  stroke="rgba(243, 237, 226, 0.12)"
                  strokeWidth="1"
                  strokeDasharray={r === 66 ? '3 3' : undefined}
                />
              ))}
              {/* Sector guide lines (-68°, -22°, +22°, +68°) */}
              {[-68, -22, 22, 68].map((deg) => {
                const rad = (deg * Math.PI) / 180;
                return (
                  <line
                    key={deg}
                    x1="140"
                    y1="112"
                    x2={140 + Math.sin(rad) * 100}
                    y2={112 - Math.cos(rad) * 100}
                    stroke="rgba(243, 237, 226, 0.12)"
                    strokeWidth="1"
                  />
                );
              })}
              {/* Sector density indicators */}
              {[
                { deg: -45, val: metrics.leftDensity },
                { deg: 0, val: metrics.centerDensity },
                { deg: 45, val: metrics.rightDensity },
              ].map((s) => {
                const rad = (s.deg * Math.PI) / 180;
                const r = Math.max(24, 96 - s.val * 0.55);
                const sx = 140 + Math.sin(rad) * r;
                const sy = 112 - Math.cos(rad) * r;
                return (
                  <circle
                    key={s.deg}
                    cx={sx}
                    cy={sy}
                    r={4 + (s.val / 100) * 5}
                    fill={s.val > 45 ? 'rgba(242, 184, 75, 0.45)' : 'rgba(243, 237, 226, 0.25)'}
                  />
                );
              })}
              {/* Active HRTF acoustic beam */}
              {activeRadarAngle !== null && (
                <g>
                  <line
                    x1="140"
                    y1="112"
                    x2={radarBx}
                    y2={radarBy}
                    stroke="#ff8a3d"
                    strokeWidth="2"
                  />
                  <circle cx={radarBx} cy={radarBy} r="4" fill="#ff8a3d" />
                </g>
              )}
              {/* AI Semantic Object Blips */}
              {aiEnabled &&
                detections.map((d, i) => {
                  const deg =
                    d.direction === '正左侧'
                      ? -58
                      : d.direction === '左前方'
                        ? -34
                        : d.direction === '正前方'
                          ? 0
                          : d.direction === '右前方'
                            ? 34
                            : 58;
                  const rad = (deg * Math.PI) / 180;
                  const distR = d.distance === '近距离' ? 42 : d.distance === '中距离' ? 68 : 90;
                  const dx = 140 + Math.sin(rad) * distR;
                  const dy = 112 - Math.cos(rad) * distR;
                  const c = d.distance === '近距离' ? '#f2b84b' : '#9fc9b0';
                  return (
                    <g key={i}>
                      <circle cx={dx} cy={dy} r="4.5" fill={c} />
                      <circle cx={dx} cy={dy} r="8.5" fill="none" stroke={c} strokeOpacity="0.5" />
                    </g>
                  );
                })}
              {/* Listener Head Icon */}
              <circle cx="140" cy="112" r="9.5" fill="#161512" stroke="#f3ede2" strokeWidth="1.6" />
              <circle cx="128.5" cy="112" r="2.6" fill="#ff8a3d" />
              <circle cx="151.5" cy="112" r="2.6" fill="#ff8a3d" />
              <text x="16" y="116" fill="rgba(243, 237, 226, 0.5)" fontSize="10" fontFamily="monospace">
                L -68°
              </text>
              <text x="224" y="116" fill="rgba(243, 237, 226, 0.5)" fontSize="10" fontFamily="monospace">
                +68° R
              </text>
              <text x="132" y="10" fill="rgba(243, 237, 226, 0.5)" fontSize="10" fontFamily="monospace">
                0°
              </text>
            </svg>
          </div>

          <div className="grid grid-cols-3 gap-2.5">
            {[
              { label: '左视野', val: metrics.leftDensity },
              { label: '正前方', val: metrics.centerDensity },
              { label: '右视野', val: metrics.rightDensity },
            ].map((zone) => {
              const high = zone.val > 45;
              return (
                <div key={zone.label} className="panel-sunken p-3 text-center">
                  <div className="eyebrow">{zone.label}</div>
                  <div className={`num mt-1 text-xl ${high ? 'text-warn' : 'text-ink'}`}>
                    {zone.val}%
                  </div>
                  <div className="meter mt-2">
                    <span
                      style={{
                        width: `${zone.val}%`,
                        backgroundColor: high ? 'var(--color-warn)' : 'var(--color-ink)',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="border-l-2 border-signal pl-3 py-0.5 text-[13px] text-ink-2 leading-relaxed">
            <span className="font-semibold text-ink">导航建议：</span>
            {metrics.navigationCue}
          </div>
        </div>

        {/* Algorithm Parameter Lab */}
        <div className="panel p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="eyebrow">Parameters</div>
              <h3 className="text-[15px] font-semibold text-ink mt-0.5">算法参数台</h3>
            </div>
            <span className="text-[12px] text-ink-3">实时生效</span>
          </div>

          {/* Spatial Model */}
          <div className="space-y-1.5">
            <label className="block text-[12.5px] font-medium text-ink-2">空间定位模型</label>
            <div className="seg w-full">
              <button
                type="button"
                data-active={useHRTF}
                onClick={() => setUseHRTF(true)}
                className="seg-item flex-1"
              >
                3D HRTF 双耳头模
              </button>
              <button
                type="button"
                data-active={!useHRTF}
                onClick={() => setUseHRTF(false)}
                className="seg-item flex-1"
              >
                传统双声道 Stereo
              </button>
            </div>
          </div>

          {/* Frequency Scale Mapping */}
          <div className="space-y-1.5">
            <label className="block text-[12.5px] font-medium text-ink-2">垂直坐标 → 频率音阶</label>
            <div className="seg w-full">
              {[
                { id: 'pentatonic' as ScaleMode, label: '五声音阶' },
                { id: 'chromatic' as ScaleMode, label: '半音阶' },
                { id: 'voice_log' as ScaleMode, label: 'vOICe 对数谱' },
              ].map((s) => (
                <button
                  key={s.id}
                  type="button"
                  data-active={scaleMode === s.id}
                  onClick={() => setScaleMode(s.id)}
                  className="seg-item flex-1"
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* Sliders */}
          <div className="space-y-3.5 pt-1">
            <div>
              <div className="flex justify-between text-[12.5px] mb-1">
                <span className="text-ink-2">Sobel 边缘阈值</span>
                <span className="num text-ink">{edgeThreshold}</span>
              </div>
              <input
                type="range"
                min={10}
                max={120}
                value={edgeThreshold}
                onChange={(e) => setEdgeThreshold(Number(e.target.value))}
                className="slider"
              />
            </div>

            <div>
              <div className="flex justify-between text-[12.5px] mb-1">
                <span className="text-ink-2">扫描步进（单列时长）</span>
                <span className="num text-ink">{scanSpeedMs} ms</span>
              </div>
              <input
                type="range"
                min={25}
                max={120}
                value={scanSpeedMs}
                onChange={(e) => setScanSpeedMs(Number(e.target.value))}
                className="slider"
              />
            </div>

            <div>
              <div className="flex justify-between text-[12.5px] mb-1">
                <span className="text-ink-2">主输出音量</span>
                <span className="num text-ink">{Math.round(volume * 100)}%</span>
              </div>
              <input
                type="range"
                min={10}
                max={100}
                value={Math.round(volume * 100)}
                onChange={(e) => setVolume(Number(e.target.value) / 100)}
                className="slider"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
