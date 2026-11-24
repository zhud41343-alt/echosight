// EchoDepthNet Browser ONNX Runtime Engine (Front-End <-> PyTorch Research Unification)
// Dynamically loads `onnxruntime-web` and our self-trained 1.11M EchoDepthNet (+Residual EG-DFM)
// checkpoint (`./models/echodepth_net.onnx`) to predict:
//   1) Dense normalized proximity depth map (rendered as Inferno heatmap)
//   2) 48x16 Acoustic Spatial Hazard Matrix aligned directly with EchoSight's 3D HRTF engine

const MODEL_INPUT_SIZE = 256;
// P1-02 (V8.9): SHA-256 short-hash query parameter prevents stale Cache-First SW weight collisions across releases
export const ECHO_DEPTH_MODEL_SHA256_SHORT = '95ebf7fa';
const MODEL_URL = `./models/echodepth_net.onnx?v=${ECHO_DEPTH_MODEL_SHA256_SHORT}`;
export const ECHO_DEPTH_MODEL_VERSION = 'EchoDepthNet-ResidualEGDFM-1.11M';

type OrtModule = typeof import('onnxruntime-web');
type OrtSession = import('onnxruntime-web').InferenceSession;

let ortMod: OrtModule | null = null;
let session: OrtSession | null = null;
let loadPromise: Promise<boolean> | null = null;
let lastError: string | null = null;

// Compact Inferno-inspired colormap LUT (256 RGB entries) for crisp proximity visualization
const INFERNO_STOPS: Array<[number, number, number, number]> = [
  [0.0, 12, 8, 24],
  [0.25, 87, 16, 110],
  [0.5, 188, 55, 84],
  [0.75, 249, 142, 9],
  [1.0, 252, 255, 164],
];

const INFERNO_LUT = new Uint8ClampedArray(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let seg = 0;
  while (seg < INFERNO_STOPS.length - 2 && t > INFERNO_STOPS[seg + 1][0]) seg++;
  const [t0, r0, g0, b0] = INFERNO_STOPS[seg];
  const [t1, r1, g1, b1] = INFERNO_STOPS[seg + 1];
  const f = (t - t0) / Math.max(1e-6, t1 - t0);
  INFERNO_LUT[i * 3] = Math.round(r0 + (r1 - r0) * f);
  INFERNO_LUT[i * 3 + 1] = Math.round(g0 + (g1 - g0) * f);
  INFERNO_LUT[i * 3 + 2] = Math.round(b0 + (b1 - b0) * f);
}

let scratchCanvas: HTMLCanvasElement | null = null;

function getScratchContext(): CanvasRenderingContext2D | null {
  if (typeof document === 'undefined') return null;
  if (!scratchCanvas) {
    scratchCanvas = document.createElement('canvas');
    scratchCanvas.width = MODEL_INPUT_SIZE;
    scratchCanvas.height = MODEL_INPUT_SIZE;
  }
  return scratchCanvas.getContext('2d', { willReadFrequently: true });
}

export function getEchoDepthStatus(): {
  ready: boolean;
  loading: boolean;
  error: string | null;
} {
  return {
    ready: session !== null,
    loading: loadPromise !== null,
    error: lastError,
  };
}

export async function loadEchoDepthOnnxModel(
  onStatus?: (msg: string) => void
): Promise<boolean> {
  if (session) return true;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    try {
      lastError = null;
      onStatus?.('正在加载 ONNX Runtime Web 与 EchoDepthNet (1.11M)...');
      const ort = await import('onnxruntime-web');
      // Single-threaded WASM works universally on GitHub Pages without COOP/COEP headers
      ort.env.wasm.numThreads = 1;
      ortMod = ort;

      session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      onStatus?.('EchoDepthNet 神经深度引擎就绪（本地 ONNX 推理）');
      return true;
    } catch (err) {
      lastError = err instanceof Error ? err.message : 'EchoDepthNet ONNX 加载失败';
      session = null;
      onStatus?.(`EchoDepthNet 加载失败（将保持 Sobel 边缘模式）：${lastError}`);
      return false;
    } finally {
      loadPromise = null;
    }
  })();

  return loadPromise;
}

export interface EchoDepthInferenceOutput {
  /** 256x256 Inferno heatmap ImageData representing predicted proximity */
  depthHeatmap: ImageData;
  /** 48 columns x 16 rows acoustic hazard matrix aligned with spatialAudio.playScanColumn */
  acousticMatrix: number[][];
  /** Mean predicted proximity in [0, 1] */
  meanProximity: number;
  inferenceMs: number;
}

/**
 * Runs real on-device EchoDepthNet ONNX inference on any video/image canvas.
 */
export async function runEchoDepthInference(
  sourceCanvas: HTMLCanvasElement
): Promise<EchoDepthInferenceOutput | null> {
  if (!session || !ortMod) return null;
  const ctx = getScratchContext();
  if (!ctx) return null;

  const t0 = performance.now();
  ctx.drawImage(sourceCanvas, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const rgba = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;

  const hw = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  const chw = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    chw[i] = rgba[i * 4] / 255;
    chw[hw + i] = rgba[i * 4 + 1] / 255;
    chw[2 * hw + i] = rgba[i * 4 + 2] / 255;
  }

  const inputTensor = new ortMod.Tensor('float32', chw, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  const outputs = await session.run({ rgb_image: inputTensor });
  const depthData = outputs.depth_map?.data as Float32Array | undefined;
  const acousticData = outputs.acoustic_matrix?.data as Float32Array | undefined; // [1, 1, 16, 48]
  if (!depthData || !acousticData || depthData.length !== hw || acousticData.length !== 16 * 48) {
    return null;
  }

  // Render 256x256 Inferno heatmap (strictly reject NaN / Infinity outputs)
  const heatPixels = new Uint8ClampedArray(hw * 4);
  let proxSum = 0;
  for (let i = 0; i < hw; i++) {
    const raw = depthData[i];
    if (!Number.isFinite(raw)) return null;
    const v = Math.max(0, Math.min(1, raw));
    proxSum += v;
    const lutIdx = Math.round(v * 255) * 3;
    heatPixels[i * 4] = INFERNO_LUT[lutIdx];
    heatPixels[i * 4 + 1] = INFERNO_LUT[lutIdx + 1];
    heatPixels[i * 4 + 2] = INFERNO_LUT[lutIdx + 2];
    heatPixels[i * 4 + 3] = 255;
  }
  const depthHeatmap = new ImageData(heatPixels, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  // Convert [16, 48] row-major acoustic tensor -> EchoSight [48][16] column-major matrix
  const cols = 48;
  const rows = 16;
  const acousticMatrix: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const rawA = acousticData[r * cols + c];
      if (!Number.isFinite(rawA)) return null;
      acousticMatrix[c][r] = Math.max(0, Math.min(1, rawA));
    }
  }

  return {
    depthHeatmap,
    acousticMatrix,
    meanProximity: proxSum / hw,
    inferenceMs: Math.round(performance.now() - t0),
  };
}
