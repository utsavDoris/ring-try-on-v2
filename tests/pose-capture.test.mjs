import test from 'node:test';
import assert from 'node:assert/strict';
import { Quaternion } from 'three';
import {
  HOLD_MS,
  PoseHoldTimer,
  buildPhotoSnapshot,
  evaluateCapturePose,
  evaluateFullHandTrack,
  evaluateManualCaptureGate,
  evaluateOpenHandGesture,
  hasProperRingFit,
  landmarkMotion,
  snapshotHasOrientation,
} from '../app/pose-capture.ts';

function openHandLandmarks({
  cx = 0.5,
  cy = 0.5,
  scale = 0.35,
} = {}) {
  const pts = Array.from({ length: 21 }, () => ({ x: cx, y: cy, z: 0 }));
  // Wrist + finger chain: open palm with tips farther than PIPs.
  pts[0] = { x: cx, y: cy + scale * 0.55, z: 0 };
  const bases = [
    [1, 2, 3, 4, -0.22], // thumb
    [5, 6, 7, 8, -0.1],
    [9, 10, 11, 12, 0],
    [13, 14, 15, 16, 0.1],
    [17, 18, 19, 20, 0.22],
  ];
  for (const [mcp, pip, dip, tip, dx] of bases) {
    pts[mcp] = { x: cx + dx * scale, y: cy + scale * 0.15, z: 0 };
    pts[pip] = { x: cx + dx * scale, y: cy - scale * 0.05, z: 0 };
    pts[dip] = { x: cx + dx * scale, y: cy - scale * 0.25, z: 0 };
    pts[tip] = { x: cx + dx * scale, y: cy - scale * 0.45, z: 0 };
  }
  return pts;
}

function fistLandmarks() {
  const pts = openHandLandmarks({ scale: 0.38 });
  // Keep middle tip for hand span; curl other fingers so tip sits nearer MCP than PIP.
  for (const tip of [8, 16, 20]) {
    const mcp = tip - 3;
    const pip = tip - 2;
    pts[pip] = { x: pts[mcp].x, y: pts[mcp].y - 0.06, z: 0 };
    pts[tip - 1] = { x: pts[mcp].x, y: pts[mcp].y - 0.03, z: 0 };
    pts[tip] = { x: pts[mcp].x, y: pts[mcp].y - 0.015, z: 0 };
  }
  return pts;
}

function placement(width = 40) {
  return {
    x: 200,
    y: 300,
    width,
    orientation: new Quaternion(),
    surfaceFacing: 1,
    anchors: [{ x: 180, y: 280 }, { x: 220, y: 320 }],
    fingerLength: 80,
    handScale: 100,
  };
}

test('full-hand track rejects missing, edge-clipped, tiny, and off-center hands', () => {
  assert.equal(evaluateFullHandTrack(null).ok, false);
  assert.equal(evaluateFullHandTrack([]).reason, 'noHand');

  const clipped = openHandLandmarks({ cx: 0.5, cy: 0.5, scale: 0.4 });
  clipped[12] = { x: 0.01, y: 0.5, z: 0 };
  assert.equal(evaluateFullHandTrack(clipped).reason, 'partial');

  const tiny = openHandLandmarks({ scale: 0.08 });
  assert.equal(evaluateFullHandTrack(tiny).reason, 'size');

  const off = openHandLandmarks({ cx: 0.1, cy: 0.5, scale: 0.3 });
  // Edge clip usually wins before center when the hand sits near a border.
  assert.ok(['center', 'partial'].includes(evaluateFullHandTrack(off).reason));

  const good = openHandLandmarks();
  assert.equal(evaluateFullHandTrack(good).ok, true);
});

test('open-hand gesture detects fist vs open', () => {
  const open = evaluateOpenHandGesture(openHandLandmarks());
  assert.equal(open.isFist, false);

  const fist = evaluateOpenHandGesture(fistLandmarks());
  assert.equal(fist.isFist, true);
});

test('manual gate requires hand + locked finger fit', () => {
  assert.equal(
    evaluateManualCaptureGate({ hasHand: false, fitCalibrated: true, placement: placement() }).ok,
    false,
  );
  assert.equal(
    evaluateManualCaptureGate({ hasHand: true, fitCalibrated: false, placement: null }).ok,
    false,
  );
  assert.equal(
    evaluateManualCaptureGate({
      hasHand: true,
      fitCalibrated: true,
      placement: placement(0),
    }).ok,
    false,
  );
  assert.equal(
    evaluateManualCaptureGate({
      hasHand: true,
      fitCalibrated: true,
      placement: placement(42),
    }).ok,
    true,
  );
});

test('auto capture pose requires open hand, fit lock, and low motion', () => {
  const landmarks = openHandLandmarks();
  const fit = { fitCalibrated: true, placement: placement(36) };

  assert.equal(
    evaluateCapturePose({
      hasHand: true,
      landmarks: fistLandmarks(),
      ...fit,
      motion: 0,
    }).reason,
    'fingers',
  );

  assert.equal(
    evaluateCapturePose({
      hasHand: true,
      landmarks,
      fitCalibrated: false,
      placement: null,
      motion: 0,
    }).reason,
    'ringFit',
  );

  assert.equal(
    evaluateCapturePose({
      hasHand: true,
      landmarks,
      ...fit,
      motion: 0.05,
    }).reason,
    'track',
  );

  assert.equal(
    evaluateCapturePose({
      hasHand: true,
      landmarks,
      ...fit,
      motion: 0.005,
    }).ok,
    true,
  );
});

test('PoseHoldTimer reaches 1 only after sustained OK past arm + hold', () => {
  const timer = new PoseHoldTimer(HOLD_MS, 220, 480);
  let t = 1000;
  // Timer clamps dt to 100ms, so advance in frame-sized steps.
  let last = timer.update(true, t);
  for (let i = 0; i < 3; i += 1) {
    t += 80;
    last = timer.update(true, t);
  }
  assert.ok(last.progress < 1);
  assert.equal(last.captured, false);

  const holdFrames = Math.ceil(HOLD_MS / 80);
  for (let i = 0; i < holdFrames - 2; i += 1) {
    t += 80;
    last = timer.update(true, t);
  }
  assert.ok(last.progress < 1);
  assert.equal(last.captured, false);

  for (let i = 0; i < 4; i += 1) {
    t += 80;
    last = timer.update(true, t);
  }
  assert.equal(last.progress, 1);
  assert.equal(last.captured, true);

  // Brief drop does not wipe progress immediately.
  const dropTimer = new PoseHoldTimer(HOLD_MS, 0, 480);
  t = 5000;
  dropTimer.update(true, t);
  for (let i = 0; i < Math.ceil(HOLD_MS / 2 / 80); i += 1) {
    t += 80;
    last = dropTimer.update(true, t);
  }
  assert.ok(last.progress > 0.4);
  t += 80;
  const paused = dropTimer.update(false, t);
  assert.ok(paused.progress > 0.4);
  for (let i = 0; i < 7; i += 1) {
    t += 80;
    last = dropTimer.update(false, t);
  }
  assert.equal(last.progress, 0);
});

test('hasProperRingFit and snapshot clone width/orientation', () => {
  assert.equal(hasProperRingFit({ fitCalibrated: true, placement: placement(0) }), false);
  assert.equal(hasProperRingFit({ fitCalibrated: true, placement: placement(20) }), true);

  const source = placement(55);
  source.orientation.setFromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4);
  const landmarks = openHandLandmarks();
  const snap = buildPhotoSnapshot({
    landmarks,
    placement: source,
    handedness: 'Right',
    isMirrored: true,
    videoDimensions: { width: 1280, height: 720 },
    fitCalibrated: true,
  });

  assert.equal(snap.fitCalibrated, true);
  assert.equal(snap.handedness, 'Right');
  assert.equal(snap.isMirrored, true);
  assert.equal(snap.placement?.width, 55);
  assert.equal(snap.videoDimensions.width, 1280);
  assert.ok(snapshotHasOrientation(snap));
  // Clone, not alias.
  assert.notEqual(snap.placement, source);
  assert.notEqual(snap.landmarks, landmarks);
  assert.notEqual(snap.placement?.orientation, source.orientation);
  snap.landmarks[0].x = 9;
  assert.notEqual(landmarks[0].x, 9);
});

test('landmarkMotion averages travel between frames', () => {
  const a = openHandLandmarks();
  const b = openHandLandmarks();
  b[0] = { ...b[0], x: b[0].x + 0.02 };
  const motion = landmarkMotion(a, b);
  assert.ok(motion > 0);
  assert.ok(motion < 0.02);
});
