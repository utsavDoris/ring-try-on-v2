import test from 'node:test';
import assert from 'node:assert/strict';
import { PatchMotionTracker, usesLightweightTracking } from '../app/patch-motion.ts';
import { fitPalmMotion } from '../app/ring-tracking.ts';

function frame(tx = 0, ty = 0, textured = true, exposure = 0) {
  const width = 240, height = 320, data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const px = x - tx, py = y - ty;
    const noise = textured ? ((Math.imul(px + 713, 374761393) ^ Math.imul(py + 177, 668265263)) >>> 5) % 80 : 0;
    const offset = (y * width + x) * 4;
    data.set([130 + noise + exposure, 95 + noise + exposure, 70 + noise + exposure, 255], offset);
  }
  return {
    image: { width, height, data },
    region: [{ x: (110 + tx) / width, y: (230 + ty) / height }, { x: (110 + tx) / width, y: (90 + ty) / height }],
  };
}

test('iPad desktop-mode user agent uses lightweight tracking, real Macs do not', () => {
  assert.equal(usesLightweightTracking('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)', 5), true);
  assert.equal(usesLightweightTracking('Mozilla/5.0 (iPad; CPU OS 18)', 5), true);
  assert.equal(usesLightweightTracking('Mozilla/5.0 (iPhone; CPU iPhone OS)', 5), true);
  assert.equal(usesLightweightTracking('Mozilla/5.0 (Macintosh; Intel Mac OS X)', 0), false);
});

test('lightweight texture tracking follows fast movement and reversal without a catch-up animation', () => {
  const tracker = new PatchMotionTracker();
  const offsets = [[0, 0], [35, -24], [-20, 30], [0, 0]];
  for (let i = 0; i < offsets.length; i++) {
    const [x, y] = offsets[i], f = frame(x, y, true, i * 3);
    const motion = tracker.update(f.image, f.region, i * 33);
    if (!i) { assert.equal(motion, null); continue; }
    assert.ok(motion);
    const fit = fitPalmMotion(motion.before, motion.after);
    const p = fit.map({ x: 0.5, y: 0.5 });
    assert.ok(Math.abs((p.x - 0.5) * 240 - (x - offsets[i - 1][0])) < 0.2);
    assert.ok(Math.abs((p.y - 0.5) * 320 - (y - offsets[i - 1][1])) < 0.2);
  }
});

test('stationary texture stays stationary despite noisy anatomical landmarks', () => {
  const tracker = new PatchMotionTracker(), f = frame();
  tracker.update(f.image, f.region, 0);
  const noisy = f.region.map(p => ({ x: p.x + 3 / 240, y: p.y - 2 / 320 }));
  const motion = tracker.update(f.image, noisy, 33);
  assert.ok(motion);
  const fit = fitPalmMotion(motion.before, motion.after);
  assert.ok(fit.residual < 0.001);
  const p = fit.map({ x: 0.5, y: 0.5 });
  assert.ok(Math.hypot(p.x - 0.5, p.y - 0.5) < 0.001);
});

test('featureless frames and reacquisition fall back safely', () => {
  const tracker = new PatchMotionTracker(), f = frame(0, 0, false);
  tracker.update(f.image, f.region, 0);
  assert.equal(tracker.update(f.image, f.region, 33), null);
  const textured = frame();
  tracker.update(textured.image, textured.region, 66);
  assert.equal(tracker.update(textured.image, textured.region, 600), null);
  tracker.reset();
  assert.equal(tracker.update(textured.image, textured.region, 633), null);
});
