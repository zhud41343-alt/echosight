// Real-time Computer Vision Pipeline: Grayscale + Sobel 3x3 Edge Extraction + Acoustic Matrix

export type PresetSceneId = 'crosswalk' | 'room' | 'mountain' | 'calibration';

export interface SpatialSceneMetrics {
  leftDensity: number; // 0..100
  centerDensity: number; // 0..100
  rightDensity: number; // 0..100
  edgeCoverage: number; // 0..100
  navigationCue: string;
}

export interface AcousticMatrixResult {
  cols: number;
  rows: number;
  matrix: number[][]; // [col][row] in 0..1
  edgeImageData: ImageData;
  metrics: SpatialSceneMetrics;
}

/**
 * Draw one of 4 rich procedural high-contrast visual scenes onto a Canvas.
 * Zero external image dependencies — guaranteed instant load on Qoder Sites.
 */
export function drawPresetScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  sceneId: PresetSceneId
) {
  ctx.clearRect(0, 0, width, height);

  if (sceneId === 'crosswalk') {
    // Night street crosswalk with traffic signal and obstacle on right
    const bg = ctx.createLinearGradient(0, 0, 0, height);
    bg.addColorStop(0, '#12110f');
    bg.addColorStop(0.55, '#1a1916');
    bg.addColorStop(1, '#24221e');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    // Horizon line
    ctx.strokeStyle = '#ff8a3d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, height * 0.52);
    ctx.lineTo(width, height * 0.52);
    ctx.stroke();

    // Perspective crosswalk stripes
    ctx.fillStyle = '#ece5d8';
    for (let i = 0; i < 5; i++) {
      const x = width * (0.14 + i * 0.16);
      ctx.beginPath();
      ctx.moveTo(x, height * 0.58);
      ctx.lineTo(x + width * 0.08, height * 0.58);
      ctx.lineTo(x + width * 0.11, height * 0.94);
      ctx.lineTo(x - width * 0.03, height * 0.94);
      ctx.closePath();
      ctx.fill();
    }

    // Traffic light pole on left
    ctx.fillStyle = '#8a867d';
    ctx.fillRect(width * 0.16, height * 0.16, 10, height * 0.4);
    ctx.fillStyle = '#1b1a17';
    ctx.fillRect(width * 0.12, height * 0.12, 34, 68);
    ctx.strokeStyle = '#ece5d8';
    ctx.lineWidth = 2;
    ctx.strokeRect(width * 0.12, height * 0.12, 34, 68);
    // Green pedestrian light
    ctx.fillStyle = '#7fbf8e';
    ctx.beginPath();
    ctx.arc(width * 0.12 + 17, height * 0.12 + 48, 10, 0, Math.PI * 2);
    ctx.fill();

    // Approaching car / obstacle silhouette on right
    ctx.fillStyle = '#f3ede2';
    ctx.beginPath();
    ctx.roundRect(width * 0.68, height * 0.44, width * 0.24, height * 0.26, 12);
    ctx.fill();
    ctx.fillStyle = '#1b1a17';
    ctx.fillRect(width * 0.72, height * 0.48, width * 0.16, height * 0.08);
    // Headlights
    ctx.fillStyle = '#f2c46b';
    ctx.beginPath();
    ctx.arc(width * 0.73, height * 0.62, 7, 0, Math.PI * 2);
    ctx.arc(width * 0.87, height * 0.62, 7, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  if (sceneId === 'room') {
    // Indoor living room with doorframe in center and table obstacle on left
    ctx.fillStyle = '#141310';
    ctx.fillRect(0, 0, width, height);

    // Bright open doorway in center-right
    ctx.fillStyle = '#efe6d6';
    ctx.fillRect(width * 0.56, height * 0.14, width * 0.22, height * 0.66);
    ctx.strokeStyle = '#ff8a3d';
    ctx.lineWidth = 4;
    ctx.strokeRect(width * 0.56, height * 0.14, width * 0.22, height * 0.66);

    // Table obstacle on left
    ctx.fillStyle = '#cfc7b8';
    ctx.fillRect(width * 0.08, height * 0.56, width * 0.34, 16);
    ctx.fillRect(width * 0.12, height * 0.62, 12, height * 0.26);
    ctx.fillRect(width * 0.36, height * 0.62, 12, height * 0.26);

    // Lamp on table
    ctx.fillStyle = '#f2c46b';
    ctx.beginPath();
    ctx.moveTo(width * 0.18, height * 0.42);
    ctx.lineTo(width * 0.28, height * 0.42);
    ctx.lineTo(width * 0.25, height * 0.3);
    ctx.lineTo(width * 0.21, height * 0.3);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#8a867d';
    ctx.fillRect(width * 0.225, height * 0.42, 6, height * 0.14);
    return;
  }

  if (sceneId === 'mountain') {
    // Starry night sky + glowing moon + crisp mountain ridge (melodic contour!)
    const sky = ctx.createLinearGradient(0, 0, 0, height);
    sky.addColorStop(0, '#100f0d');
    sky.addColorStop(1, '#1b1a17');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, width, height);

    // Bright moon top-right
    ctx.fillStyle = '#f3ede2';
    ctx.beginPath();
    ctx.arc(width * 0.78, height * 0.22, 24, 0, Math.PI * 2);
    ctx.fill();

    // Stars
    const stars = [
      [0.14, 0.18], [0.28, 0.12], [0.44, 0.22], [0.58, 0.14], [0.88, 0.34],
    ];
    stars.forEach(([sx, sy]) => {
      ctx.beginPath();
      ctx.arc(width * sx, height * sy, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Mountain ridge line (sweeping ascending & descending melody!)
    ctx.strokeStyle = '#ff8a3d';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(0, height * 0.82);
    ctx.lineTo(width * 0.22, height * 0.46);
    ctx.lineTo(width * 0.4, height * 0.64);
    ctx.lineTo(width * 0.62, height * 0.32);
    ctx.lineTo(width * 0.82, height * 0.68);
    ctx.lineTo(width, height * 0.56);
    ctx.stroke();

    // Fill subtle under-ridge
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 138, 61, 0.16)';
    ctx.fill();
    return;
  }

  // 'calibration': Diagonal ascending sweep + concentric rings for spatial audio calibration
  ctx.fillStyle = '#141310';
  ctx.fillRect(0, 0, width, height);

  // Ascending diagonal line (left-bottom -> right-top: hear pitch rise as sound moves L -> R!)
  ctx.strokeStyle = '#ff8a3d';
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.moveTo(width * 0.08, height * 0.86);
  ctx.lineTo(width * 0.92, height * 0.14);
  ctx.stroke();

  // Concentric circles in center
  ctx.strokeStyle = '#f3ede2';
  ctx.lineWidth = 4;
  [36, 72].forEach((r) => {
    ctx.beginPath();
    ctx.arc(width * 0.5, height * 0.5, r, 0, Math.PI * 2);
    ctx.stroke();
  });
}

/**
 * Run Sobel 3x3 edge detection + luminance fusion to extract a (cols x rows) acoustic sonification matrix
 * and a vivid cyan/emerald Sobel edge heatmap ImageData.
 */
export function processFrameToAcousticMatrix(
  sourceData: ImageData,
  cols = 48,
  rows = 16,
  edgeThreshold = 38,
  edgeWeight = 0.72
): AcousticMatrixResult {
  const { width, height, data } = sourceData;
  const gray = new Float32Array(width * height);

  // 1. Luminance extraction (Rec. 709)
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
  }

  // 2. Sobel 3x3 Gradient Magnitude
  const edges = new Float32Array(width * height);
  const outImage = new ImageData(width, height);
  const out = outImage.data;

  let edgePixels = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const gx =
        -gray[idx - width - 1] +
        gray[idx - width + 1] -
        2 * gray[idx - 1] +
        2 * gray[idx + 1] -
        gray[idx + width - 1] +
        gray[idx + width + 1];

      const gy =
        -gray[idx - width - 1] -
        2 * gray[idx - width] -
        gray[idx - width + 1] +
        gray[idx + width - 1] +
        2 * gray[idx + width] +
        gray[idx + width + 1];

      const mag = Math.min(255, Math.sqrt(gx * gx + gy * gy) * 0.55);
      const filtered = mag >= edgeThreshold ? mag : 0;
      edges[idx] = filtered;

      if (filtered > 0) edgePixels++;

      // Render warm phosphor edge heatmap (instrument-screen style)
      const pIdx = idx * 4;
      const lum = gray[idx] * 0.1 + 14;
      out[pIdx] = Math.min(255, lum + filtered * 1.0);
      out[pIdx + 1] = Math.min(255, lum + filtered * 0.74);
      out[pIdx + 2] = Math.min(255, lum * 0.9 + filtered * 0.5);
      out[pIdx + 3] = 255;
    }
  }

  // 3. Downsample into (cols x rows) acoustic matrix
  const matrix: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
  const cellW = width / cols;
  const cellH = height / rows;

  let leftSum = 0;
  let centerSum = 0;
  let rightSum = 0;

  for (let c = 0; c < cols; c++) {
    const x0 = Math.floor(c * cellW);
    const x1 = Math.min(width - 1, Math.floor((c + 1) * cellW));
    for (let r = 0; r < rows; r++) {
      const y0 = Math.floor(r * cellH);
      const y1 = Math.min(height - 1, Math.floor((r + 1) * cellH));

      let sumEdge = 0;
      let sumLum = 0;
      let count = 0;

      // Stride 2 for blazing 60fps performance
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          const idx = y * width + x;
          sumEdge += edges[idx];
          sumLum += gray[idx];
          count++;
        }
      }

      const avgEdge = count > 0 ? sumEdge / count / 255 : 0;
      const avgLum = count > 0 ? sumLum / count / 255 : 0;
      // Boost edges so obstacle boundaries pop out acoustically
      const combined = Math.min(1, avgEdge * edgeWeight * 2.2 + avgLum * (1 - edgeWeight));
      matrix[c][r] = combined;

      if (c < cols / 3) leftSum += combined;
      else if (c < (cols * 2) / 3) centerSum += combined;
      else rightSum += combined;
    }
  }

  const thirdCells = (cols / 3) * rows;
  const leftDensity = Math.min(100, Math.round((leftSum / thirdCells) * 165));
  const centerDensity = Math.min(100, Math.round((centerSum / thirdCells) * 165));
  const rightDensity = Math.min(100, Math.round((rightSum / thirdCells) * 165));
  const edgeCoverage = Math.min(100, Math.round((edgePixels / (width * height)) * 300));

  let navigationCue = `视觉边缘密度：左 ${leftDensity}% / 中 ${centerDensity}% / 右 ${rightDensity}%（仅供参考，不代表通道无障碍）`;
  if (centerDensity > 45) {
    navigationCue = `正前方视觉轮廓较密（${centerDensity}%），左侧 ${leftDensity}%、右侧 ${rightDensity}%，请结合盲杖与现场确认`;
  } else if (rightDensity > 48 && leftDensity < 30) {
    navigationCue = `右侧轮廓较密（${rightDensity}%），左侧轮廓较低（${leftDensity}%），仅供参考`;
  } else if (leftDensity > 48 && rightDensity < 30) {
    navigationCue = `左侧轮廓较密（${leftDensity}%），右侧轮廓较低（${rightDensity}%），仅供参考`;
  }

  return {
    cols,
    rows,
    matrix,
    edgeImageData: outImage,
    metrics: {
      leftDensity,
      centerDensity,
      rightDensity,
      edgeCoverage,
      navigationCue,
    },
  };
}
