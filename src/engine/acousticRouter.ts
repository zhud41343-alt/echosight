// Acoustic Matrix Router (P0-1 V8.5)
// Strictly isolates Sobel edge acoustic matrices from EchoDepthNet ONNX neural matrices
// and binds live camera neural packets to real raw video frame identity (`frameId` & `capturedAt`),
// preventing frozen cameras or duplicate ONNX runs from refreshing TTL or feeding stale audio to HRTF.

export type ScannerViewMode = 'source' | 'sobel' | 'matrix' | 'depth';
export type ScannerInputSource = 'preset' | 'camera' | 'upload';
export type ActiveAcousticSource = 'sobel' | 'neural_echodepth' | 'none';

export interface NeuralAcousticPacket {
  matrix: number[][];
  inputSource: ScannerInputSource;
  /** Strictly monotonic real video frame ID from FrameTracker (or 1 for static preset/upload) */
  frameId: number;
  generation: number;
  /** Real raw frame capture timestamp T0 (for camera) */
  capturedAt: number;
  /** ONNX inference start timestamp T1 */
  startedAt: number;
  /** ONNX inference completion timestamp T2 */
  completedAt: number;
  /** Expiry deadline anchored to T0 (never refreshed by re-inferring the same frame) */
  expiresAt: number;
  modelVersion: string;
}

export interface SobelAcousticPacket {
  matrix: number[][];
  inputSource: ScannerInputSource;
  /** Real video frame ID from FrameTracker (or 0 for static preset/upload) */
  frameId: number;
  /** Browser frame receipt timestamp from rVFC / currentTime advance */
  capturedAt: number;
}

export interface ResolvedAcousticFeed {
  source: ActiveAcousticSource;
  matrix: number[][] | null;
  reason?:
    | 'page_hidden'
    | 'neural_not_ready'
    | 'neural_expired'
    | 'neural_frame_stale'
    | 'neural_invalid'
    | 'sobel_not_ready'
    | 'sobel_frame_stale';
}

export function validateAcousticMatrix(
  matrix: unknown,
  expectedCols = 48,
  expectedRows = 16
): matrix is number[][] {
  if (!Array.isArray(matrix) || matrix.length !== expectedCols) return false;
  for (let c = 0; c < expectedCols; c++) {
    const col = matrix[c];
    if (!Array.isArray(col) || col.length !== expectedRows) return false;
    for (let r = 0; r < expectedRows; r++) {
      const v = col[r];
      if (typeof v !== 'number' || !Number.isFinite(v)) return false;
    }
  }
  return true;
}

export class AcousticMatrixRouter {
  private sobelPacket: SobelAcousticPacket | null = null;
  private neuralPacket: NeuralAcousticPacket | null = null;
  private generation = 0;
  private lastAcceptedNeuralFrameId = 0;
  private cols: number;
  private rows: number;

  constructor(cols = 48, rows = 16) {
    this.cols = cols;
    this.rows = rows;
  }

  /**
   * Advances generation token whenever viewMode, inputSource, preset, or camera changes.
   * Immediately invalidates any previous neural matrix and rejects late-resolving ONNX tasks.
   */
  public beginGeneration(): number {
    this.generation += 1;
    this.neuralPacket = null;
    this.lastAcceptedNeuralFrameId = 0;
    return this.generation;
  }

  public get currentGeneration(): number {
    return this.generation;
  }

  public updateSobel(
    matrix: number[][],
    meta: { inputSource: ScannerInputSource; frameId: number; capturedAt: number } = {
      inputSource: 'preset',
      frameId: 0,
      capturedAt: 0,
    }
  ): boolean {
    if (!validateAcousticMatrix(matrix, this.cols, this.rows)) return false;
    if (meta.inputSource === 'camera') {
      if (!Number.isInteger(meta.frameId) || meta.frameId <= 0 || !Number.isFinite(meta.capturedAt)) {
        return false;
      }
      // P1-01 (V8.7): Strictly monotonic camera Sobel frameId — reject equal & older frame IDs!
      if (
        this.sobelPacket?.inputSource === 'camera' &&
        meta.frameId <= this.sobelPacket.frameId
      ) {
        return false;
      }
    }
    this.sobelPacket = {
      matrix,
      inputSource: meta.inputSource,
      frameId: meta.frameId,
      capturedAt: meta.capturedAt,
    };
    return true;
  }

  public invalidateSobel(): void {
    this.sobelPacket = null;
  }

  public invalidateNeural(): void {
    this.neuralPacket = null;
  }

  public commitNeuralPacket(
    packet: NeuralAcousticPacket,
    now?: number,
    maxRawFrameAgeMs = 3000
  ): { accepted: boolean; reason?: string } {
    if (packet.generation !== this.generation) {
      return { accepted: false, reason: 'stale_generation' };
    }
    if (!Number.isInteger(packet.frameId) || packet.frameId <= 0) {
      return { accepted: false, reason: 'invalid_frame_id' };
    }
    // P0-1 (V8.5): Strictly monotonic frameId — equal or older frameId MUST be rejected!
    if (packet.frameId <= this.lastAcceptedNeuralFrameId) {
      return { accepted: false, reason: 'duplicate_or_out_of_order_frame' };
    }
    if (
      !Number.isFinite(packet.capturedAt) ||
      !Number.isFinite(packet.startedAt) ||
      !Number.isFinite(packet.completedAt) ||
      packet.capturedAt > packet.startedAt ||
      packet.startedAt > packet.completedAt ||
      packet.completedAt > packet.expiresAt
    ) {
      return { accepted: false, reason: 'invalid_timestamps' };
    }
    const checkNow = now ?? packet.completedAt;
    if (packet.inputSource === 'camera') {
      if (packet.capturedAt > checkNow + 50 || checkNow - packet.capturedAt > maxRawFrameAgeMs) {
        return { accepted: false, reason: 'stale_raw_frame' };
      }
      if (checkNow > packet.expiresAt) {
        return { accepted: false, reason: 'already_expired' };
      }
    }
    if (!validateAcousticMatrix(packet.matrix, this.cols, this.rows)) {
      return { accepted: false, reason: 'invalid_matrix' };
    }
    this.lastAcceptedNeuralFrameId = packet.frameId;
    this.neuralPacket = packet;
    return { accepted: true };
  }

  public resolveActive(
    viewMode: ScannerViewMode,
    now: number,
    pageHidden = false,
    maxRawFrameAgeMs = 3000
  ): ResolvedAcousticFeed {
    if (pageHidden) {
      return { source: 'none', matrix: null, reason: 'page_hidden' };
    }

    if (viewMode === 'depth') {
      if (!this.neuralPacket) {
        return { source: 'none', matrix: null, reason: 'neural_not_ready' };
      }
      if (this.neuralPacket.generation !== this.generation) {
        return { source: 'none', matrix: null, reason: 'neural_invalid' };
      }
      if (now > this.neuralPacket.expiresAt) {
        return { source: 'none', matrix: null, reason: 'neural_expired' };
      }
      if (
        this.neuralPacket.inputSource === 'camera' &&
        (this.neuralPacket.capturedAt > now + 50 || now - this.neuralPacket.capturedAt > maxRawFrameAgeMs)
      ) {
        return { source: 'none', matrix: null, reason: 'neural_frame_stale' };
      }
      return { source: 'neural_echodepth', matrix: this.neuralPacket.matrix };
    }

    if (!this.sobelPacket) {
      return { source: 'none', matrix: null, reason: 'sobel_not_ready' };
    }
    // P0 (V8.6): Enforce strict frame freshness on live camera Sobel audio too (never play frozen camera frames!)
    if (
      this.sobelPacket.inputSource === 'camera' &&
      (this.sobelPacket.capturedAt > now + 50 || now - this.sobelPacket.capturedAt > maxRawFrameAgeMs)
    ) {
      return { source: 'none', matrix: null, reason: 'sobel_frame_stale' };
    }
    return { source: 'sobel', matrix: this.sobelPacket.matrix };
  }
}
