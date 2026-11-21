// Acoustic Matrix Router (P0-1 V8.4)
// Strictly isolates Sobel edge acoustic matrices from EchoDepthNet ONNX neural matrices
// so live camera Sobel updates can NEVER overwrite the active neural matrix during HRTF rendering.

export type ScannerViewMode = 'source' | 'sobel' | 'matrix' | 'depth';
export type ScannerInputSource = 'preset' | 'camera' | 'upload';
export type ActiveAcousticSource = 'sobel' | 'neural_echodepth' | 'none';

export interface NeuralAcousticPacket {
  matrix: number[][];
  inputSource: ScannerInputSource;
  frameId: number;
  generation: number;
  startedAt: number;
  completedAt: number;
  expiresAt: number;
  modelVersion: string;
}

export interface ResolvedAcousticFeed {
  source: ActiveAcousticSource;
  matrix: number[][] | null;
  reason?:
    | 'page_hidden'
    | 'neural_not_ready'
    | 'neural_expired'
    | 'neural_invalid'
    | 'sobel_not_ready';
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
  private sobelMatrix: number[][] | null = null;
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

  public updateSobel(matrix: number[][]): boolean {
    if (!validateAcousticMatrix(matrix, this.cols, this.rows)) return false;
    this.sobelMatrix = matrix;
    return true;
  }

  public invalidateNeural(): void {
    this.neuralPacket = null;
  }

  public commitNeuralPacket(packet: NeuralAcousticPacket): { accepted: boolean; reason?: string } {
    if (packet.generation !== this.generation) {
      return { accepted: false, reason: 'stale_generation' };
    }
    if (!Number.isInteger(packet.frameId) || packet.frameId <= 0) {
      return { accepted: false, reason: 'invalid_frame_id' };
    }
    if (packet.frameId < this.lastAcceptedNeuralFrameId) {
      return { accepted: false, reason: 'out_of_order_frame' };
    }
    if (
      !Number.isFinite(packet.startedAt) ||
      !Number.isFinite(packet.completedAt) ||
      packet.startedAt > packet.completedAt ||
      packet.completedAt > packet.expiresAt
    ) {
      return { accepted: false, reason: 'invalid_timestamps' };
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
    pageHidden = false
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
      return { source: 'neural_echodepth', matrix: this.neuralPacket.matrix };
    }

    if (!this.sobelMatrix) {
      return { source: 'none', matrix: null, reason: 'sobel_not_ready' };
    }
    return { source: 'sobel', matrix: this.sobelMatrix };
  }
}
