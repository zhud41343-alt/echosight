// Experimental Single-Channel Ambient Sound Anomaly Detector (Web Audio API)
// IMPORTANT SCIENTIFIC LIMITATION:
// A single smartphone microphone cannot perform true 360° sound source localization
// and cannot reliably confirm an approaching vehicle from behind.
// This module only detects sudden high-energy spectral anomalies (sharp mid-high spikes
// or strong low-frequency surges) as an experimental auxiliary cue.

export interface AcousticAnomalyEvent {
  kind: 'sharp_spike' | 'low_freq_surge';
  peakEnergy: number;
  message: string;
}

export interface AcousticFrameMetrics {
  avgTotal: number;
  avgLow: number; // 60-350Hz average magnitude (0..255)
  midHighPeak: number; // 700-2800Hz peak magnitude (0..255)
  baseline: number;
}

export interface AcousticEvaluationSample {
  id: string;
  expectedAnomaly: boolean;
  frame: AcousticFrameMetrics;
}

export interface AcousticEvaluationSummary {
  total: number;
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
}

/**
 * Pure frame classifier for deterministic offline false-positive / false-negative evaluation.
 */
export function classifyAcousticFrame(frame: AcousticFrameMetrics): AcousticAnomalyEvent | null {
  if (frame.midHighPeak > 195 && frame.midHighPeak > frame.baseline * 3.2) {
    return {
      kind: 'sharp_spike',
      peakEnergy: Math.round(frame.midHighPeak),
      message: '检测到突发尖锐环境声响（实验性提示，无法判断方向，请现场确认）。',
    };
  }
  if (frame.avgLow > 140 && frame.avgLow > frame.baseline * 2.9) {
    return {
      kind: 'low_freq_surge',
      peakEnergy: Math.round(frame.avgLow),
      message: '检测到突发低频环境声响增强（实验性提示，无法判断方向，请现场确认）。',
    };
  }
  return null;
}

/**
 * Offline benchmark helper to measure false-positive and false-negative rates
 * on recorded or synthetic acoustic frame datasets.
 */
export function evaluateAcousticDataset(
  samples: AcousticEvaluationSample[]
): AcousticEvaluationSummary {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (const s of samples) {
    const predicted = classifyAcousticFrame(s.frame) !== null;
    if (s.expectedAnomaly && predicted) tp++;
    else if (!s.expectedAnomaly && predicted) fp++;
    else if (!s.expectedAnomaly && !predicted) tn++;
    else fn++;
  }
  const negatives = fp + tn;
  const positives = tp + fn;
  return {
    total: samples.length,
    truePositives: tp,
    falsePositives: fp,
    trueNegatives: tn,
    falseNegatives: fn,
    falsePositiveRate: negatives > 0 ? Math.round((fp / negatives) * 1000) / 1000 : 0,
    falseNegativeRate: positives > 0 ? Math.round((fn / positives) * 1000) / 1000 : 0,
  };
}

export class AcousticThreatMonitor {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private analyser: AnalyserNode | null = null;
  private data: Uint8Array<ArrayBuffer> | null = null;
  private timer: number | null = null;
  private baselineEnergy = 24;
  private lastAlertAt = 0;

  public async start(onAnomaly: (ev: AcousticAnomalyEvent) => void): Promise<boolean> {
    if (typeof window === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      return false;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new AudioCtx();
      const src = this.ctx.createMediaStreamSource(this.stream);
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 512;
      this.analyser.smoothingTimeConstant = 0.45;
      src.connect(this.analyser);
      this.data = new Uint8Array(new ArrayBuffer(this.analyser.frequencyBinCount));

      const sampleRate = this.ctx.sampleRate;
      const binHz = sampleRate / this.analyser.fftSize;

      this.timer = window.setInterval(() => {
        if (!this.analyser || !this.data) return;
        this.analyser.getByteFrequencyData(this.data);

        let lowSum = 0;
        let lowCount = 0;
        let midHighPeak = 0;
        let totalSum = 0;

        for (let i = 1; i < this.data.length; i++) {
          const hz = i * binHz;
          const v = this.data[i];
          totalSum += v;
          if (hz >= 60 && hz <= 350) {
            lowSum += v;
            lowCount++;
          } else if (hz >= 700 && hz <= 2800) {
            if (v > midHighPeak) midHighPeak = v;
          }
        }

        const avgTotal = totalSum / this.data.length;
        const avgLow = lowCount > 0 ? lowSum / lowCount : 0;
        this.baselineEnergy = this.baselineEnergy * 0.94 + avgTotal * 0.06;

        const now = performance.now();
        if (now - this.lastAlertAt < 6000) return;

        const ev = classifyAcousticFrame({
          avgTotal,
          avgLow,
          midHighPeak,
          baseline: this.baselineEnergy,
        });
        if (ev) {
          this.lastAlertAt = now;
          onAnomaly(ev);
        }
      }, 220);

      return true;
    } catch {
      return false;
    }
  }

  public stop() {
    if (this.timer !== null) {
      window.clearInterval(this.timer);
      this.timer = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.ctx) {
      this.ctx.close().catch(() => {});
      this.ctx = null;
    }
    this.analyser = null;
    this.data = null;
  }
}
