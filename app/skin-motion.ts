import { fitPalmMotion, type ScreenPoint } from './ring-tracking.ts';

export type FlowMat = {
  rows: number; cols: number;
  data: Uint8Array; data32F: Float32Array;
  delete(): void;
};
export type FlowCv = {
  Mat: new () => FlowMat;
  Size: new (x: number, y: number) => unknown;
  matFromArray(rows: number, cols: number, type: number, data: ArrayLike<number>): FlowMat;
  matFromImageData(data: { width: number; height: number; data: Uint8ClampedArray }): FlowMat;
  cvtColor(source: FlowMat, destination: FlowMat, code: number): void;
  goodFeaturesToTrack(image: FlowMat, corners: FlowMat, count: number, quality: number, distance: number, mask: FlowMat): void;
  calcOpticalFlowPyrLK(previous: FlowMat, current: FlowMat, points: FlowMat, next: FlowMat, status: FlowMat, error: FlowMat, window: unknown, level: number): void;
  COLOR_RGBA2GRAY: number; CV_8UC1: number; CV_32FC2: number;
};
export type SkinMotion = { before: ScreenPoint[]; after: ScreenPoint[] };
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? 0;

// Tracks actual skin texture between adjacent video frames. Landmarks still
// supply anatomy; optical flow carries the attachment during motion so noisy
// landmark positions do not directly shake the ring every frame.
export class SkinMotionTracker {
  private previous: FlowMat | null = null;
  private region: ScreenPoint[] = [];
  private time = 0;
  reset() { this.previous?.delete(); this.previous = null; this.region = []; this.time = 0; }
  update(cv: FlowCv, image: { width: number; height: number; data: Uint8ClampedArray }, region: ScreenPoint[], time: number): SkinMotion | null {
    const source = cv.matFromImageData(image);
    const gray = new cv.Mat();
    try { cv.cvtColor(source, gray, cv.COLOR_RGBA2GRAY); }
    catch (error) { gray.delete(); throw error; }
    finally { source.delete(); }
    const temporary: FlowMat[] = [];
    const own = (mat: FlowMat) => { temporary.push(mat); return mat; };
    try {
      if (!this.previous || this.previous.cols !== image.width || this.previous.rows !== image.height || time - this.time > 140 || this.region.length < 2 || region.length < 2) return null;
      const pixels = new Uint8Array(image.width * image.height);
      const [base, tip] = this.region;
      const bx = base.x * image.width, by = base.y * image.height;
      const dx = (tip.x - base.x) * image.width, dy = (tip.y - base.y) * image.height;
      const length = Math.hypot(dx, dy);
      if (length < 8) return null;
      // Restrict feature selection to the ring finger, clear of background
      // edges and neighboring fingers. The region is from the previous frame.
      const radius = Math.max(4, length * 0.25);
      for (let y = Math.max(0, Math.floor(Math.min(by, by + dy) - radius)); y < Math.min(image.height, Math.ceil(Math.max(by, by + dy) + radius)); y++) {
        for (let x = Math.max(0, Math.floor(Math.min(bx, bx + dx) - radius)); x < Math.min(image.width, Math.ceil(Math.max(bx, bx + dx) + radius)); x++) {
          const along = ((x - bx) * dx + (y - by) * dy) / (length * length);
          const across = Math.abs((x - bx) * dy - (y - by) * dx) / length;
          if (along > 0.12 && along < 0.95 && across < radius) pixels[y * image.width + x] = 255;
        }
      }
      const mask = own(cv.matFromArray(image.height, image.width, cv.CV_8UC1, pixels));
      const points = own(new cv.Mat());
      cv.goodFeaturesToTrack(this.previous, points, 32, 0.015, 3, mask);
      if (points.rows < 6) return null;
      const next = own(new cv.Mat()), status = own(new cv.Mat()), error = own(new cv.Mat());
      const window = new cv.Size(15, 15);
      cv.calcOpticalFlowPyrLK(this.previous, gray, points, next, status, error, window, 2);
      const back = own(new cv.Mat()), backStatus = own(new cv.Mat()), backError = own(new cv.Mat());
      cv.calcOpticalFlowPyrLK(gray, this.previous, next, back, backStatus, backError, window, 2);
      const pairs: { before: ScreenPoint; after: ScreenPoint }[] = [];
      for (let i = 0; i < points.rows; i++) {
        const p = { x: points.data32F[2 * i], y: points.data32F[2 * i + 1] };
        const q = { x: next.data32F[2 * i], y: next.data32F[2 * i + 1] };
        if (!status.data[i] || !backStatus.data[i] || error.data32F[i] > 22 || !Number.isFinite(q.x + q.y)) continue;
        if (Math.hypot(back.data32F[2 * i] - p.x, back.data32F[2 * i + 1] - p.y) > 0.65) continue;
        if (q.x < 0 || q.y < 0 || q.x >= image.width || q.y >= image.height) continue;
        pairs.push({ before: p, after: q });
      }
      if (pairs.length < 6) return null;
      const mx = median(pairs.map(p => p.after.x - p.before.x));
      const my = median(pairs.map(p => p.after.y - p.before.y));
      const inliers = pairs.filter(p => Math.hypot(p.after.x - p.before.x - mx, p.after.y - p.before.y - my) < Math.max(2, length * 0.1));
      if (inliers.length < 6 || inliers.length < pairs.length * 0.65) return null;
      const fit = fitPalmMotion(inliers.map(p => p.before), inliers.map(p => p.after));
      if (!fit || fit.residual > 0.7 || fit.scale < 0.8 || fit.scale > 1.25) return null;
      // Compare displacement with current finger landmarks to reject a
      // background patch that accidentally entered the previous feature mask.
      const center = { x: bx + dx * 0.53, y: by + dy * 0.53 };
      const tracked = fit.map(center);
      const expected = { x: (region[0].x * 0.47 + region[1].x * 0.53) * image.width, y: (region[0].y * 0.47 + region[1].y * 0.53) * image.height };
      if (Math.hypot(tracked.x - expected.x, tracked.y - expected.y) > Math.max(4, length * 0.2)) return null;
      const normalize = (p: ScreenPoint) => ({ x: p.x / image.width, y: p.y / image.height });
      return { before: inliers.map(p => normalize(p.before)), after: inliers.map(p => normalize(p.after)) };
    } finally {
      for (const mat of temporary) mat.delete();
      this.previous?.delete(); this.previous = gray; this.region = region; this.time = time;
    }
  }
}
