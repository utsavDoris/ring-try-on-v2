import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import Module from 'node:module';
import { fileURLToPath } from 'node:url';
import { SkinMotionTracker } from '../app/skin-motion.ts';
import { fitPalmMotion } from '../app/ring-tracking.ts';
import { findFingerSilhouette } from '../app/finger-silhouette.ts';

const runtimePath = fileURLToPath(new URL('../public/opencv/opencv.cjs', import.meta.url));
const runtime = new Module(runtimePath);
runtime.filename = runtimePath;
runtime.paths = Module._nodeModulePaths(process.cwd());
runtime._compile(readFileSync(new URL('../public/opencv/opencv.js', import.meta.url), 'utf8'), runtimePath);
const cv = runtime.exports;
await new Promise(resolve => { if (cv.Mat) resolve(); else cv.onRuntimeInitialized = resolve; });


function frame(tx = 0, ty = 0, textured = true) {
  const width = 180, height = 320;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const px = x - tx, py = y - ty;
    const inside = px >= 58 && px <= 122 && py >= 45 && py <= 280;
    const noise = textured ? ((Math.imul(px + 713, 374761393) ^ Math.imul(py + 177, 668265263)) >>> 5) % 80 : 0;
    const offset = (y * width + x) * 4;
    data[offset] = inside ? 160 + noise : 30;
    data[offset + 1] = inside ? 120 + noise : 30;
    data[offset + 2] = inside ? 95 + noise : 30;
    data[offset + 3] = 255;
  }
  const region = [{ x: (90 + tx) / width, y: (250 + ty) / height }, { x: (90 + tx) / width, y: (70 + ty) / height }];
  return { image: { width, height, data }, region };
}

test('bundled OpenCV follows actual texture translation with forward/backward validation', () => {
  const tracker = new SkinMotionTracker();
  const first = frame(); tracker.update(cv, first.image, first.region, 0);
  const moved = frame(4, -6);
  const motion = tracker.update(cv, moved.image, moved.region, 33);
  assert.ok(motion, 'enough reliable skin features');
  const fit = fitPalmMotion(motion.before, motion.after);
  const p = fit.map({ x: 0.5, y: 0.5 });
  assert.ok(Math.abs((p.x - 0.5) * 180 - 4) < 0.25);
  assert.ok(Math.abs((p.y - 0.5) * 320 + 6) < 0.25);
  tracker.reset();
});

test('stationary texture stays fixed despite noisy landmark regions', () => {
  const tracker = new SkinMotionTracker();
  const first = frame(); tracker.update(cv, first.image, first.region, 0);
  const shiftedRegion = first.region.map(p => ({ x: p.x + 0.012, y: p.y - 0.008 }));
  const motion = tracker.update(cv, first.image, shiftedRegion, 33);
  assert.ok(motion);
  const fit = fitPalmMotion(motion.before, motion.after);
  const p = fit.map({ x: 0.5, y: 0.5 });
  assert.ok(Math.hypot(p.x - 0.5, p.y - 0.5) < 0.001);
  tracker.reset();
});

test('blurred/featureless skin falls back instead of inventing image motion', () => {
  const tracker = new SkinMotionTracker();
  const first = frame(0, 0, false); tracker.update(cv, first.image, first.region, 0);
  assert.equal(tracker.update(cv, first.image, first.region, 33), null);
  tracker.reset();
});

test('multi-section edge fit finds the two outer finger boundaries', () => {
  const { image } = frame(0, 0, false);
  const edge = findFingerSilhouette(image, { x: 90, y: 250 }, { x: 90, y: 70 });
  assert.ok(edge);
  assert.ok(Math.abs(edge.width - 64) <= 4, `width ${edge.width}`);
  assert.ok(edge.support >= 3);
});

test('supplied palm capture measures the skin boundary beyond the undersized ring guides', async t => {
  const path = new URL('../screenshots/ring-tryon-2026-09-08T10-53-22-998Z.png', import.meta.url);
  if (!existsSync(path)) { t.skip('local user capture is not included in source control'); return; }
  const { default: sharp } = await import('sharp');
  const { data, info } = await sharp(fileURLToPath(path)).resize({ height: 640 }).raw().toBuffer({ resolveWithObject: true });
  const scale = 640 / 1562;
  const point = (x, y) => ({ x: x * scale, y: y * scale });
  // Endpoints read from the capture's yellow longitudinal guide. Sample rows
  // bypass the virtual band itself, so they measure the visible skin edges.
  const edges = findFingerSilhouette({ ...info, data }, point(490, 1047), point(565, 730), [
    { base: point(333, 1040), tip: point(348, 700) },
    { base: point(665, 1130), tip: point(780, 800) },
  ]);
  assert.ok(edges && edges.support >= 4);
  assert.ok(edges.width / scale > 150 && edges.width / scale < 178);
});
