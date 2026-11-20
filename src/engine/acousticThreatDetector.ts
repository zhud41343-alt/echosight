// 360° On-Device Acoustic Rear/Side Threat Detector (Web Audio API Microphone Analyser)
// Compensates for camera rear blind-spot by detecting sudden horn blasts (700-2800Hz)
// or rapidly approaching vehicle rumble spikes (60-350Hz) from behind/sides.

export interface AcousticThreatEvent {
  kind: 'horn' | 'approaching_noise';
  levelDb: number;
  message: string;
}

export class AcousticThreatMonitor {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private analyser: AnalyserNode | null = null;
  private data: Uint8Array<ArrayBuffer> | null = null;
  private timer: number | null = null;
  private baselineEnergy = 22;
  private lastAlertAt = 0;

  public async start(onThreat: (ev: AcousticThreatEvent) => void): Promise<boolean> {
    if (typeof window === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      return false;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: false, // Keep raw ambient horn/engine spikes detectable
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

        // Compute low-band (60-350Hz: vehicle engine/rumble) & mid-high horn band (700-2800Hz)
        let lowSum = 0;
        let lowCount = 0;
        let hornPeak = 0;
        let totalSum = 0;

        for (let i = 1; i < this.data.length; i++) {
          const hz = i * binHz;
          const v = this.data[i];
          totalSum += v;
          if (hz >= 60 && hz <= 350) {
            lowSum += v;
            lowCount++;
          } else if (hz >= 700 && hz <= 2800) {
            if (v > hornPeak) hornPeak = v;
          }
        }

        const avgTotal = totalSum / this.data.length;
        const avgLow = lowCount > 0 ? lowSum / lowCount : 0;

        // Smoothly adapt ambient baseline
        this.baselineEnergy = this.baselineEnergy * 0.94 + avgTotal * 0.06;

        const now = performance.now();
        if (now - this.lastAlertAt < 5500) return;

        // Sharp car/e-bike horn spike
        if (hornPeak > 195 && hornPeak > this.baselineEnergy * 3.2) {
          this.lastAlertAt = now;
          onThreat({
            kind: 'horn',
            levelDb: Math.round(hornPeak / 2.5),
            message: '注意四周和后方！检测到急促鸣笛声，请立刻靠边停步确认安全。',
          });
          return;
        }

        // Sudden loud approaching vehicle rumble from blind spot
        if (avgLow > 135 && avgLow > this.baselineEnergy * 2.8) {
          this.lastAlertAt = now;
          onThreat({
            kind: 'approaching_noise',
            levelDb: Math.round(avgLow / 2.5),
            message: '小心后方或侧面！检测到快速逼近的车辆轰鸣声，请先停一下。',
          });
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
