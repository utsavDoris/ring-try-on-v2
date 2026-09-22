import { fitPalmMotion, type ScreenPoint } from './ring-tracking.ts';
import type { SkinMotion } from './skin-motion.ts';

type GrayFrame = { width: number; height: number; data: Uint8Array };
type ImageFrame = { width: number; height: number; data: Uint8ClampedArray };

export function usesLightweightTracking(userAgent: string, touchPoints: number) {
  return /iPad|iPhone|iPod/.test(userAgent) || (/Macintosh/.test(userAgent) && touchPoints > 0);
}

function gray(image: ImageFrame): GrayFrame {
  const data = new Uint8Array(image.width * image.height);
  for (let i = 0; i < data.length; i++) {
    const j = i * 4;
    data[i] = (image.data[j] * 77 + image.data[j + 1] * 150 + image.data[j + 2] * 29) >> 8;
  }
  return { width: image.width, height: image.height, data };
}

function inside(frame: GrayFrame, x: number, y: number) {
  return x >= 3 && y >= 3 && x < frame.width - 3 && y < frame.height - 3;
}

// Compare zero-mean 5x5 patches, tolerating exposure changes. Search is
// bounded regardless of screen size or distance travelled by the hand.
function match(a: GrayFrame, b: GrayFrame, p: ScreenPoint, expected: ScreenPoint) {
  let best = Infinity, bestX = 0, bestY = 0;
  const patch: number[] = [];
  let sum = 0, square = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    const value = a.data[(p.y + dy) * a.width + p.x + dx];
    patch.push(value); sum += value; square += value * value;
  }
  if (square / 25 - (sum / 25) ** 2 < 12) return null;
  const score = (x: number, y: number) => {
    if (!inside(b, x, y)) return Infinity;
    let difference = 0, squaredDifference = 0, index = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const d = patch[index++] - b.data[(y + dy) * b.width + x + dx];
      difference += d; squaredDifference += d * d;
    }
    return Math.max(0, squaredDifference / 25 - (difference / 25) ** 2);
  };
  const cx = Math.round(expected.x), cy = Math.round(expected.y);
  // Full integer search avoids missing a textured patch on a coarse grid.
  for (let y = cy - 7; y <= cy + 7; y++) for (let x = cx - 7; x <= cx + 7; x++) {
    const error = score(x, y);
    if (error < best) { best = error; bestX = x; bestY = y; }
  }
  if (best > 180) return null;
  // Parabolic refinement gives sub-pixel motion rather than 1px steps.
  const refine = (minus: number, plus: number) => {
    const denominator = minus + plus - 2 * best;
    return Number.isFinite(denominator) && denominator > 0.01
      ? Math.max(-0.5, Math.min(0.5, (minus - plus) / (2 * denominator))) : 0;
  };
  return {
    x: bestX + refine(score(bestX - 1, bestY), score(bestX + 1, bestY)),
    y: bestY + refine(score(bestX, bestY - 1), score(bestX, bestY + 1)),
  };
}

// Small-frame sparse texture tracking without a second WebAssembly runtime.
// Anatomy remains supplied by MediaPipe; uncertain texture falls back to it.
export class PatchMotionTracker {
  private previous: GrayFrame | null = null;
  private region: ScreenPoint[] = [];
  private time = 0;
  reset() { this.previous = null; this.region = []; this.time = 0; }
  update(image: ImageFrame, region: ScreenPoint[], time: number): SkinMotion | null {
    const current = gray(image), previous = this.previous, oldRegion = this.region, oldTime = this.time;
    this.previous = current; this.region = region; this.time = time;
    if (!previous || previous.width !== current.width || previous.height !== current.height ||
        time - oldTime > 140 || oldRegion.length < 2 || region.length < 2) return null;
    const point = (p: ScreenPoint) => ({ x: p.x * image.width, y: p.y * image.height });
    const [base, tip] = oldRegion.map(point), [newBase, newTip] = region.map(point);
    const dx = tip.x - base.x, dy = tip.y - base.y, length = Math.hypot(dx, dy);
    if (length < 12) return null;
    const pairs: { before: ScreenPoint; after: ScreenPoint }[] = [];
    for (const along of [0.22, 0.38, 0.54, 0.70, 0.86]) for (const across of [-0.15, 0, 0.15]) {
      const p = { x: Math.round(base.x + dx * along - dy * across), y: Math.round(base.y + dy * along + dx * across) };
      if (!inside(previous, p.x, p.y)) continue;
      const expected = {
        x: p.x + (newBase.x - base.x) * (1 - along) + (newTip.x - tip.x) * along,
        y: p.y + (newBase.y - base.y) * (1 - along) + (newTip.y - tip.y) * along,
      };
      const q = match(previous, current, p, expected);
      if (!q) continue;
      const back = match(current, previous, { x: Math.round(q.x), y: Math.round(q.y) }, p);
      if (!back || Math.hypot(back.x - p.x, back.y - p.y) > 0.8) continue;
      pairs.push({ before: p, after: q });
    }
    if (pairs.length < 6) return null;
    const median = (v: number[]) => v.sort((a, b) => a - b)[Math.floor(v.length / 2)];
    const mx = median(pairs.map(p => p.after.x - p.before.x));
    const my = median(pairs.map(p => p.after.y - p.before.y));
    const inliers = pairs.filter(p => Math.hypot(p.after.x - p.before.x - mx, p.after.y - p.before.y - my) < Math.max(2, length * 0.08));
    if (inliers.length < 6 || inliers.length < pairs.length * 0.65) return null;
    const fit = fitPalmMotion(inliers.map(p => p.before), inliers.map(p => p.after));
    if (!fit || fit.residual > 0.8 || fit.scale < 0.8 || fit.scale > 1.25) return null;
    const normalize = (p: ScreenPoint) => ({ x: p.x / image.width, y: p.y / image.height });
    return { before: inliers.map(p => normalize(p.before)), after: inliers.map(p => normalize(p.after)) };
  }
}
