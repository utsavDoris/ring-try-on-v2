import {
  findFingerSilhouette,
  MIN_FINGER_EDGE_CONFIDENCE,
  type FingerSilhouette,
} from './finger-silhouette.ts';

export type EdgePoint = { x: number; y: number };

/** Subset of OpenCV needed for Scharr edge magnitude (shares runtime with skin motion). */
export type EdgeCv = {
  Mat: new () => {
    rows: number;
    cols: number;
    data: Uint8Array;
    delete(): void;
  };
  Size: new (width: number, height: number) => unknown;
  matFromImageData(data: {
    width: number;
    height: number;
    data: Uint8ClampedArray;
  }): { delete(): void };
  cvtColor(source: { delete(): void }, destination: { delete(): void }, code: number): void;
  GaussianBlur(
    source: { delete(): void },
    destination: { delete(): void },
    ksize: unknown,
    sigmaX: number,
  ): void;
  Scharr(
    source: { delete(): void },
    destination: { delete(): void },
    depth: number,
    dx: number,
    dy: number,
  ): void;
  convertScaleAbs(source: { delete(): void }, destination: { delete(): void }): void;
  addWeighted(
    src1: { delete(): void },
    alpha: number,
    src2: { delete(): void },
    beta: number,
    gamma: number,
    destination: { delete(): void },
  ): void;
  COLOR_RGBA2GRAY: number;
  CV_16S: number;
};

export type FingerEdgeRoi = {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
  dw: number;
  dh: number;
  scale: number;
};

const EDGE_ROI_MAX_SIDE = 320;
const median = (values: number[]) =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? 0;

/**
 * Axis-aligned finger ROI in source pixels, optionally downscaled so the long
 * side is at most {@link EDGE_ROI_MAX_SIDE}.
 */
export function computeFingerEdgeRoi(
  base: EdgePoint,
  tip: EdgePoint,
  neighbors: { base: EdgePoint; tip: EdgePoint }[],
  sourceWidth: number,
  sourceHeight: number,
  maxSide = EDGE_ROI_MAX_SIDE,
): FingerEdgeRoi | null {
  if (!(sourceWidth > 0) || !(sourceHeight > 0)) return null;
  const length = Math.hypot(tip.x - base.x, tip.y - base.y);
  if (length < 8) return null;

  const xs = [base.x, tip.x];
  const ys = [base.y, tip.y];
  for (const neighbor of neighbors) {
    xs.push(neighbor.base.x, neighbor.tip.x);
    ys.push(neighbor.base.y, neighbor.tip.y);
  }
  const padAlong = length * 0.28;
  const padAcross = length * 0.72;
  const minX = Math.min(...xs) - padAcross;
  const maxX = Math.max(...xs) + padAcross;
  const minY = Math.min(...ys) - padAlong;
  const maxY = Math.max(...ys) + padAlong;

  const sx = Math.max(0, Math.floor(minX));
  const sy = Math.max(0, Math.floor(minY));
  const ex = Math.min(sourceWidth, Math.ceil(maxX));
  const ey = Math.min(sourceHeight, Math.ceil(maxY));
  const sw = ex - sx;
  const sh = ey - sy;
  if (sw < 8 || sh < 8) return null;

  const scale = Math.min(1, maxSide / Math.max(sw, sh));
  const dw = Math.max(8, Math.round(sw * scale));
  const dh = Math.max(8, Math.round(sh * scale));
  return { sx, sy, sw, sh, dw, dh, scale: dw / sw };
}

export function mapSourcePointToRoi(point: EdgePoint, roi: FingerEdgeRoi): EdgePoint {
  return {
    x: (point.x - roi.sx) * roi.scale,
    y: (point.y - roi.sy) * roi.scale,
  };
}

export function mapRoiPointToSource(point: EdgePoint, roi: FingerEdgeRoi): EdgePoint {
  return {
    x: point.x / roi.scale + roi.sx,
    y: point.y / roi.scale + roi.sy,
  };
}

/**
 * Gray ΓåÆ blur ΓåÆ Scharr magnitude packed as RGBA so {@link findFingerSilhouette}
 * can reuse its existing cross-section sampler.
 */
export function scharrMagnitudeRgba(
  cv: EdgeCv,
  image: { width: number; height: number; data: Uint8ClampedArray },
): ImageData | null {
  const temporary: { delete(): void }[] = [];
  const own = <T extends { delete(): void }>(mat: T) => {
    temporary.push(mat);
    return mat;
  };
  try {
    const source = own(cv.matFromImageData(image));
    const gray = own(new cv.Mat());
    const blur = own(new cv.Mat());
    const dx = own(new cv.Mat());
    const dy = own(new cv.Mat());
    const absX = own(new cv.Mat());
    const absY = own(new cv.Mat());
    const mag = own(new cv.Mat());
    cv.cvtColor(source, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blur, new cv.Size(3, 3), 0);
    cv.Scharr(blur, dx, cv.CV_16S, 1, 0);
    cv.Scharr(blur, dy, cv.CV_16S, 0, 1);
    cv.convertScaleAbs(dx, absX);
    cv.convertScaleAbs(dy, absY);
    cv.addWeighted(absX, 0.5, absY, 0.5, 0, mag);
    const out = new ImageData(image.width, image.height);
    const pixels = mag.data;
    for (let i = 0, p = 0; i < pixels.length; i++, p += 4) {
      const v = pixels[i];
      out.data[p] = v;
      out.data[p + 1] = v;
      out.data[p + 2] = v;
      out.data[p + 3] = 255;
    }
    return out;
  } catch {
    return null;
  } finally {
    for (const mat of temporary) mat.delete();
  }
}

export function measureFingerSilhouetteOnRoi(
  rgba: { width: number; height: number; data: Uint8ClampedArray },
  base: EdgePoint,
  tip: EdgePoint,
  neighbors: { base: EdgePoint; tip: EdgePoint }[],
  measureT: number,
  cv?: EdgeCv | null,
): FingerSilhouette | null {
  if (cv) {
    const mag = scharrMagnitudeRgba(cv, rgba);
    if (mag) {
      const filtered = findFingerSilhouette(mag, base, tip, neighbors, measureT);
      if (filtered) return filtered;
    }
  }
  return findFingerSilhouette(rgba, base, tip, neighbors, measureT);
}

/**
 * Median of recent width/offset samples; holds last clear sample briefly when
 * confidence dips so fit does not unlock on a single noisy frame.
 */
export class EdgeMeasurementSmoother {
  private widths: number[] = [];
  private offsets: number[] = [];
  private lastGood: {
    width: number;
    offset: number;
    revision: number;
    confidence: number;
  } | null = null;

  reset() {
    this.widths = [];
    this.offsets = [];
    this.lastGood = null;
  }

  smooth(
    sample:
      | { width: number; offset: number; revision: number; confidence: number }
      | undefined,
    now: number,
  ): { width: number; offset: number; revision: number; confidence: number } | undefined {
    if (
      sample
      && sample.confidence >= MIN_FINGER_EDGE_CONFIDENCE
      && Number.isFinite(sample.width)
      && Number.isFinite(sample.offset)
    ) {
      this.widths.push(sample.width);
      this.offsets.push(sample.offset);
      if (this.widths.length > 3) this.widths.shift();
      if (this.offsets.length > 3) this.offsets.shift();
      const smoothed = {
        width: median(this.widths),
        offset: median(this.offsets),
        revision: sample.revision,
        confidence: sample.confidence,
      };
      this.lastGood = smoothed;
      return smoothed;
    }

    if (this.lastGood && now - this.lastGood.revision < 120) {
      return this.lastGood;
    }
    return undefined;
  }
}
