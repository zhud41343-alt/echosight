// EchoSight 3D HRTF Binaural Spatial Audio & Sensory Substitution Engine
// Implements real-time Web Audio API PannerNode (HRTF) + multi-band additive visual-to-audio sonification

export type ScaleMode = 'pentatonic' | 'chromatic' | 'voice_log';

export interface SpatialAudioConfig {
  scaleMode: ScaleMode;
  useHRTF: boolean;
  volume: number; // 0 to 1
}

// 16-band pleasant Pentatonic frequencies (bottom -> top)
const PENTATONIC_16 = [
  196.0, 220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25,
  587.33, 659.25, 783.99, 880.0, 1046.5, 1174.66, 1318.51, 1567.98,
];

function getRowFrequency(rowIndexFromTop: number, totalRows: number, mode: ScaleMode): number {
  // Invert so top of image = high pitch, bottom of image = low pitch (neuroscience natural cross-modal mapping)
  const normY = 1 - rowIndexFromTop / Math.max(1, totalRows - 1);

  if (mode === 'pentatonic') {
    const idx = Math.min(
      PENTATONIC_16.length - 1,
      Math.max(0, Math.round(normY * (PENTATONIC_16.length - 1)))
    );
    return PENTATONIC_16[idx];
  }

  if (mode === 'chromatic') {
    // 2.5 octaves from C3 (130.81Hz) to G5 (784Hz) + upper extension
    const semitones = Math.round(normY * 32);
    return 164.81 * Math.pow(2, semitones / 12);
  }

  // 'voice_log': Classic vOICe Sensory Substitution logarithmic mapping (250Hz -> 2800Hz)
  return 250 * Math.pow(2800 / 250, normY);
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export function frequencyToNoteName(freq: number): string {
  if (!Number.isFinite(freq) || freq <= 0) return '--';
  const midi = Math.round(69 + 12 * Math.log2(freq / 440));
  const note = NOTE_NAMES[((midi % 12) + 12) % 12];
  const octave = Math.floor(midi / 12) - 1;
  return `${note}${octave}`;
}

export function getPitchInfo(
  normY: number,
  mode: ScaleMode,
  totalRows = 16
): { freqHz: number; noteName: string } {
  const clamped = Math.max(0, Math.min(1, normY));
  const rowIdx = Math.round(clamped * (totalRows - 1));
  const freqHz = getRowFrequency(rowIdx, totalRows, mode);
  return { freqHz: Math.round(freqHz), noteName: frequencyToNoteName(freqHz) };
}

class SpatialAudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private freqData: Uint8Array<ArrayBuffer> | null = null;
  private activeOscillators = new Set<OscillatorNode>();

  private trackOscillator(osc: OscillatorNode, onCleanup?: () => void) {
    this.activeOscillators.add(osc);
    osc.onended = () => {
      this.activeOscillators.delete(osc);
      onCleanup?.();
    };
  }

  /**
   * P0-2 (V5): Immediately silence and release all active real-time spatial oscillators.
   * Called whenever EchoSafe transitions to UNAVAILABLE / IDLE, page hides, or session stops.
   */
  public stopAllImmediate() {
    if (this.masterGain && this.ctx) {
      try {
        this.masterGain.gain.cancelScheduledValues(this.ctx.currentTime);
        this.masterGain.gain.setValueAtTime(0.0001, this.ctx.currentTime);
      } catch {
        // Ignore audio context state errors
      }
    }
    this.activeOscillators.forEach((osc) => {
      try {
        osc.stop();
        osc.disconnect();
      } catch {
        // Ignore already-stopped oscillator
      }
    });
    this.activeOscillators.clear();
  }

  public ensureContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    if (!this.ctx) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return null;
      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.65;

      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.78;
      this.freqData = new Uint8Array(new ArrayBuffer(this.analyser.frequencyBinCount));

      this.masterGain.connect(this.analyser);
      this.analyser.connect(this.ctx.destination);
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  public getSpectrumData(): Uint8Array {
    if (!this.analyser || !this.freqData) {
      return new Uint8Array(64);
    }
    this.analyser.getByteFrequencyData(this.freqData);
    return this.freqData;
  }

  public setMasterVolume(vol: number) {
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(Math.max(0, Math.min(1, vol)), this.ctx.currentTime, 0.03);
    }
  }

  /**
   * Sonify a single vertical column of the visual acoustic matrix at normalized X in [0, 1].
   * Uses true 3D HRTF PannerNode for azimuth & elevation positioning.
   * Returns dominant telemetry for HUD display.
   */
  public playScanColumn(
    columnValues: number[], // array of length rows, each in [0, 1]
    normX: number, // 0 (far left) -> 1 (far right)
    durationSec: number,
    config: SpatialAudioConfig
  ): { azimuthDeg: number; dominantFreqHz: number | null } {
    const azimuthDeg = Math.round((normX * 2 - 1) * 68);
    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) return { azimuthDeg, dominantFreqHz: null };

    this.setMasterVolume(config.volume);

    const now = ctx.currentTime;
    const rows = columnValues.length;

    // Select top active rows to avoid muddy distortion while preserving crisp vertical contour
    const active: { row: number; amp: number }[] = [];
    for (let r = 0; r < rows; r++) {
      if (columnValues[r] > 0.14) {
        active.push({ row: r, amp: columnValues[r] });
      }
    }
    if (active.length === 0) return { azimuthDeg, dominantFreqHz: null };

    // Cap polyphony per slice to 7 strongest bands for crystal-clear binaural localization
    active.sort((a, b) => b.amp - a.amp);
    const selected = active.slice(0, 7);
    const dominantFreqHz = Math.round(getRowFrequency(selected[0].row, rows, config.scaleMode));

    // Create 3D spatial panner for this column's azimuth (-68° to +68°)
    const azimuthRad = (normX * 2 - 1) * (Math.PI * 0.38);
    const radius = 2.2;
    const posX = Math.sin(azimuthRad) * radius;
    const posZ = -Math.cos(azimuthRad) * radius;

    let spatialNode: AudioNode;
    if (config.useHRTF && typeof ctx.createPanner === 'function') {
      const panner = ctx.createPanner();
      panner.panningModel = 'HRTF';
      panner.distanceModel = 'inverse';
      panner.positionX.setValueAtTime(posX, now);
      panner.positionY.setValueAtTime(0, now);
      panner.positionZ.setValueAtTime(posZ, now);
      panner.connect(this.masterGain);
      spatialNode = panner;
    } else {
      const stereo = ctx.createStereoPanner();
      stereo.pan.setValueAtTime(normX * 2 - 1, now);
      stereo.connect(this.masterGain);
      spatialNode = stereo;
    }

    const sliceGain = ctx.createGain();
    const normFactor = 0.24 / Math.sqrt(selected.length);
    sliceGain.gain.setValueAtTime(0.0001, now);
    sliceGain.gain.linearRampToValueAtTime(normFactor, now + Math.min(0.015, durationSec * 0.25));
    sliceGain.gain.exponentialRampToValueAtTime(0.0001, now + durationSec * 1.15);
    sliceGain.connect(spatialNode);

    const stopAt = now + durationSec * 1.2;
    selected.forEach((item, idx) => {
      const osc = ctx.createOscillator();
      const rowGain = ctx.createGain();
      osc.type = config.scaleMode === 'voice_log' ? 'sine' : 'triangle';
      osc.frequency.setValueAtTime(getRowFrequency(item.row, rows, config.scaleMode), now);
      rowGain.gain.setValueAtTime(item.amp, now);
      osc.connect(rowGain);
      rowGain.connect(sliceGain);
      this.trackOscillator(osc, () => {
        rowGain.disconnect();
        if (idx === 0) {
          sliceGain.disconnect();
          spatialNode.disconnect();
        }
      });
      osc.start(now);
      osc.stop(stopAt);
    });

    return { azimuthDeg, dominantFreqHz };
  }

  /**
   * Instant spatial brush tone while user draws on the Acoustic Canvas or probes VisionScanner.
   */
  public playBrushTone(
    normX: number,
    normY: number,
    config: SpatialAudioConfig
  ): { azimuthDeg: number; freqHz: number; noteName: string } {
    const clampedX = Math.max(0, Math.min(1, normX));
    const clampedY = Math.max(0, Math.min(1, normY));
    const azimuthDeg = Math.round((clampedX * 2 - 1) * 68);
    const { freqHz, noteName } = getPitchInfo(clampedY, config.scaleMode, 16);

    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) return { azimuthDeg, freqHz, noteName };
    this.setMasterVolume(config.volume);

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = config.scaleMode === 'voice_log' ? 'sine' : 'triangle';
    osc.frequency.setValueAtTime(freqHz, now);

    gain.gain.setValueAtTime(0.001, now);
    gain.gain.linearRampToValueAtTime(0.18, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.16);

    const panner = ctx.createPanner();
    panner.panningModel = config.useHRTF ? 'HRTF' : 'equalpower';
    const angle = (clampedX * 2 - 1) * (Math.PI * 0.4);
    panner.positionX.setValueAtTime(Math.sin(angle) * 2, now);
    panner.positionY.setValueAtTime((0.5 - clampedY) * 1.5, now);
    panner.positionZ.setValueAtTime(-Math.cos(angle) * 2, now);

    osc.connect(gain);
    gain.connect(panner);
    panner.connect(this.masterGain);

    this.trackOscillator(osc, () => {
      gain.disconnect();
      panner.disconnect();
    });

    osc.start(now);
    osc.stop(now + 0.17);
    return { azimuthDeg, freqHz, noteName };
  }

  /**
   * Quick 3-point Left -> Center -> Right 3D HRTF headphone calibration sweep.
   */
  public playStereoCalibration(volume = 0.72) {
    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) return;
    this.setMasterVolume(volume);

    const now = ctx.currentTime;
    const cues = [
      { offset: 0, angle: -Math.PI * 0.4, freq: 392.0 }, // Left G4
      { offset: 0.42, angle: 0, freq: 523.25 },          // Center C5
      { offset: 0.84, angle: Math.PI * 0.4, freq: 659.25 }, // Right E5
    ];

    cues.forEach(({ offset, angle, freq }) => {
      const t = now + offset;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const panner = ctx.createPanner();
      panner.panningModel = 'HRTF';
      panner.positionX.setValueAtTime(Math.sin(angle) * 2.2, t);
      panner.positionY.setValueAtTime(0, t);
      panner.positionZ.setValueAtTime(-Math.cos(angle) * 2.2, t);

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(freq, t);
      gain.gain.setValueAtTime(0.001, t);
      gain.gain.linearRampToValueAtTime(0.22, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.34);

      osc.connect(gain);
      gain.connect(panner);
      panner.connect(this.masterGain!);

      osc.onended = () => {
        gain.disconnect();
        panner.disconnect();
      };

      osc.start(t);
      osc.stop(t + 0.36);
    });
  }

  /**
   * 2.2-second interactive "bottom-left -> top-right" 3D diagonal sweep demo
   * so first-time users immediately grasp how X->azimuth and Y->pitch work.
   */
  public playDiagonalDemo(
    onProgress?: (t: number) => void,
    onDone?: () => void,
    volume = 0.72
  ): () => void {
    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) {
      onDone?.();
      return () => {};
    }
    this.setMasterVolume(volume);

    const steps = 20;
    const stepMs = 95;
    let current = 0;
    let cancelled = false;

    const tick = () => {
      if (cancelled) return;
      const p = current / (steps - 1); // 0 -> 1
      onProgress?.(p);
      // Bottom-left (x=0.08, y=0.88) -> Top-right (x=0.92, y=0.12)
      this.playBrushTone(0.08 + p * 0.84, 0.88 - p * 0.76, {
        scaleMode: 'pentatonic',
        useHRTF: true,
        volume,
      });
      current += 1;
      if (current < steps) {
        timer = window.setTimeout(tick, stepMs);
      } else {
        timer = window.setTimeout(() => {
          if (!cancelled) onDone?.();
        }, 220);
      }
    };

    let timer = window.setTimeout(tick, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }

  /**
   * 3D Echolocation Ping for Dark Maze:
   * Emits a sonar click, followed by distance-delayed 3D HRTF echoes from walls and a warm beacon from the exit.
   */
  public playEcholocationPulse(
    rays: { angleOffsetRad: number; distance: number; hitType: 'wall' | 'exit' }[],
    exitRelative: { dx: number; dy: number; dist: number },
    volume = 0.75
  ) {
    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) return;
    this.setMasterVolume(volume);

    const now = ctx.currentTime;

    // 1. Central sonar emission click
    const clickOsc = ctx.createOscillator();
    const clickGain = ctx.createGain();
    clickOsc.type = 'sine';
    clickOsc.frequency.setValueAtTime(1400, now);
    clickOsc.frequency.exponentialRampToValueAtTime(520, now + 0.045);
    clickGain.gain.setValueAtTime(0.14, now);
    clickGain.gain.exponentialRampToValueAtTime(0.001, now + 0.05);
    clickOsc.connect(clickGain);
    clickGain.connect(this.masterGain);
    clickOsc.start(now);
    clickOsc.stop(now + 0.055);

    // 2. Spatial echoes from each raycast wall hit
    rays.forEach((ray) => {
      const delay = 0.05 + Math.min(0.42, ray.distance * 0.055);
      const t = now + delay;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const panner = ctx.createPanner();
      panner.panningModel = 'HRTF';
      panner.distanceModel = 'inverse';

      const px = Math.sin(ray.angleOffsetRad) * Math.max(0.5, ray.distance);
      const pz = -Math.cos(ray.angleOffsetRad) * Math.max(0.5, ray.distance);
      panner.positionX.setValueAtTime(px, t);
      panner.positionY.setValueAtTime(0, t);
      panner.positionZ.setValueAtTime(pz, t);

      if (ray.hitType === 'exit') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(880, t);
      } else {
        // Closer wall = higher urgency pitch & louder reflection
        osc.type = 'sine';
        const proximity = Math.max(0, 1 - ray.distance / 7);
        osc.frequency.setValueAtTime(240 + proximity * 420, t);
      }

      const amp = Math.max(0.03, 0.22 / (1 + ray.distance * 0.55));
      gain.gain.setValueAtTime(0.001, t);
      gain.gain.linearRampToValueAtTime(amp, t + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.14);

      osc.connect(gain);
      gain.connect(panner);
      panner.connect(this.masterGain!);

      osc.onended = () => {
        gain.disconnect();
        panner.disconnect();
      };

      osc.start(t);
      osc.stop(t + 0.15);
    });

    // 3. Harmonic exit beacon positioned in 3D relative to player
    const beaconTime = now + 0.12;
    const beaconOsc = ctx.createOscillator();
    const beaconGain = ctx.createGain();
    const beaconPanner = ctx.createPanner();
    beaconPanner.panningModel = 'HRTF';
    beaconPanner.positionX.setValueAtTime(exitRelative.dx * 0.7, beaconTime);
    beaconPanner.positionY.setValueAtTime(0.3, beaconTime);
    beaconPanner.positionZ.setValueAtTime(exitRelative.dy * 0.7, beaconTime);

    beaconOsc.type = 'triangle';
    beaconOsc.frequency.setValueAtTime(523.25, beaconTime);
    beaconOsc.frequency.setValueAtTime(659.25, beaconTime + 0.09);

    const beaconAmp = Math.max(0.06, 0.24 / (1 + exitRelative.dist * 0.25));
    beaconGain.gain.setValueAtTime(0.001, beaconTime);
    beaconGain.gain.linearRampToValueAtTime(beaconAmp, beaconTime + 0.02);
    beaconGain.gain.exponentialRampToValueAtTime(0.001, beaconTime + 0.3);

    beaconOsc.connect(beaconGain);
    beaconGain.connect(beaconPanner);
    beaconPanner.connect(this.masterGain);
    beaconOsc.onended = () => {
      beaconGain.disconnect();
      beaconPanner.disconnect();
    };
    beaconOsc.start(beaconTime);
    beaconOsc.stop(beaconTime + 0.32);
  }

  public playCollisionThud(volume = 0.7) {
    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) return;
    this.setMasterVolume(volume);
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(125, now);
    osc.frequency.exponentialRampToValueAtTime(55, now + 0.14);
    gain.gain.setValueAtTime(0.22, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.16);
    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start(now);
    osc.stop(now + 0.17);
  }

  public playVictoryChord(volume = 0.7) {
    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) return;
    this.setMasterVolume(volume);
    const now = ctx.currentTime;
    [523.25, 659.25, 783.99, 1046.5].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      const t = now + i * 0.08;
      osc.frequency.setValueAtTime(freq, t);
      gain.gain.setValueAtTime(0.16, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.45);
      osc.connect(gain);
      gain.connect(this.masterGain!);
      osc.start(t);
      osc.stop(t + 0.47);
    });
  }

  /**
   * 3D HRTF Item-Finder Beacon:
   * Emits a crisp double-ping positioned at the target item's exact 3D azimuth.
   * Higher pitch & double-pulse when the target is centered and close!
   */
  public playFinderBeacon(
    normX: number, // 0..1
    centeredCloseness: number, // 0..1 (1 = centered & close)
    volume = 0.75
  ) {
    const ctx = this.ensureContext();
    if (!ctx || !this.masterGain) return;
    this.setMasterVolume(volume);

    const now = ctx.currentTime;
    const angle = (Math.max(0, Math.min(1, normX)) * 2 - 1) * (Math.PI * 0.42);
    const baseFreq = 660 + centeredCloseness * 440; // 660Hz -> 1100Hz
    const pulses = centeredCloseness > 0.65 ? [0, 0.11] : [0];

    pulses.forEach((offset) => {
      const t = now + offset;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const panner = ctx.createPanner();
      panner.panningModel = 'HRTF';
      panner.positionX.setValueAtTime(Math.sin(angle) * 2.0, t);
      panner.positionY.setValueAtTime(0.15, t);
      panner.positionZ.setValueAtTime(-Math.cos(angle) * 2.0, t);

      osc.type = 'sine';
      osc.frequency.setValueAtTime(baseFreq, t);
      gain.gain.setValueAtTime(0.001, t);
      gain.gain.linearRampToValueAtTime(0.24, t + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.09);

      osc.connect(gain);
      gain.connect(panner);
      panner.connect(this.masterGain!);
      this.trackOscillator(osc, () => {
        gain.disconnect();
        panner.disconnect();
      });
      osc.start(t);
      osc.stop(t + 0.1);
    });
  }
}

export const spatialAudio = new SpatialAudioEngine();
