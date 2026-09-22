import test from 'node:test';
import assert from 'node:assert/strict';
import { Box3, Euler, Group, Mesh, MeshBasicMaterial, Quaternion, Raycaster, Vector3, OrthographicCamera } from 'three';
import { detectionSize, solveRingPose, projectedFingerAxis, RingPoseFilter, FingerFitFilter, RingSurfaceConstraint, anatomicalHandedness, swapHandedness, HandednessResolver, anatomicalChiralityFromWorld, MAX_FIT_OFFSET_RATIO, MAX_LOCK_OFFSET_RATIO, PLACEMENT_OFFSET_GATE } from '../app/ring-tracking.ts';
import {
  DEFAULT_INNER_TO_OUTER_DIAMETER,
  OCCLUSION_LENGTH_RATIO,
  OCCLUSION_RADIUS_RATIO,
  WEAR_CLEARANCE,
  applyStoneAxisBasis,
  centerRingOnMetalHole,
  createFingerOccluderGeometry,
  inferWearStoneAxis,
  measureMetalBounds,
  measureRingInnerDiameter,
  normalizeRingToUnitHole,
  readRingModelExtras,
  resolveFingerWidthPx,
  resolveRawInnerDiameter,
  ringWorldScale,
} from '../app/ring-model.ts';
import { loadRingGltf } from './load-ring-gltf.mjs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const near = (a, b, tolerance = 1e-6) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b} (tolerance ${tolerance})`);
// A metric left hand with its dorsal side facing the camera (index on the
// right of the image). Values are
// camera-space metres; screen landmarks are independently projected below.
const joints = Array.from({ length: 21 }, () => [0, 0, 0]);
joints[0] = [0, -0.055, 0];
joints[5] = [0.035, 0.005, 0];
joints[6] = [0.038, 0.048, 0];
joints[9] = [0.012, 0.013, 0];
joints[10] = [0.013, 0.055, 0];
joints[13] = [-0.01, 0.009, 0];
joints[14] = [-0.012, 0.05, 0];
joints[17] = [-0.032, 0, 0];
joints[18] = [-0.04, 0.034, 0];
function sample({ pitch = 0, yaw = 0, roll = 0, mirror = false, pixelsPerMeter = 2000, tx = 420, ty = 650, left = false, mcp = 13, pip = 14 } = {}) {
  // `left: true` mirrors the metric skeleton to a right hand and labels it
  // MediaPipe 'Right'; the default joints are a left hand labeled 'Left'.
  const q = new Quaternion().setFromEuler(new Euler(pitch, yaw, roll, 'ZYX'));
  const world = joints.map(p => new Vector3(p[0] * (left ? -1 : 1), p[1], p[2]).applyQuaternion(q));
  const screen = world.map(p => ({ x: tx + p.x * pixelsPerMeter * (mirror ? -1 : 1), y: ty - p.y * pixelsPerMeter }));
  const landmarks = world.map(p => ({ x: (tx + p.x * pixelsPerMeter) / 900, y: (ty - p.y * pixelsPerMeter) / 1600, z: -p.z, worldX: p.x, worldY: -p.y, worldZ: -p.z }));
  return { landmarks, screen, pose: solveRingPose(landmarks, screen, left ? 'Right' : 'Left', mirror, undefined, { mcp, pip }) };
}
function placement(s) {
  assert.ok(s.pose);
  return {
    x: s.screen[13].x * 0.47 + s.screen[14].x * 0.53,
    y: s.screen[13].y * 0.47 + s.screen[14].y * 0.53,
    width: s.pose.handScale * 0.4,
    orientation: s.pose.orientation,
    anchors: s.pose.anchors,
    surfaceFacing: s.pose.facing,
    fingerLength: s.pose.fingerLength,
    handScale: s.pose.handScale,
  };
}

test('portrait and landscape inference retain the full frame and aspect ratio', () => {
  assert.deepEqual(detectionSize(1080, 1920), { width: 360, height: 640 });
  assert.deepEqual(detectionSize(1920, 1080), { width: 640, height: 360 });
});

test('both hands and both camera modes keep the band axis on the observed finger', () => {
  for (const left of [false, true]) for (const mirror of [false, true]) {
    for (const yaw of [-1.15, -0.6, 0, 0.6, 1.15, Math.PI]) {
      const s = sample({ left, mirror, yaw, roll: 0.7, pitch: 0.3 });
      assert.ok(s.pose);
      const axis = projectedFingerAxis(s.pose.orientation);
      const x = s.screen[14].x - s.screen[13].x, y = s.screen[14].y - s.screen[13].y;
      near(axis.x, x / Math.hypot(x, y)); near(axis.y, y / Math.hypot(x, y));
    }
  }
});

test('both hands and both camera modes keep the setting on the correct side of the finger', () => {
  for (const left of [false, true]) for (const mirror of [false, true]) {
    const front = sample({ left, mirror }).pose;
    const side = sample({ left, mirror, yaw: 1.15 }).pose;
    const palm = sample({ left, mirror, yaw: Math.PI, pitch: 0.28 }).pose;
    const stone = p => new Vector3(0, 0, -1).applyQuaternion(p.orientation);
    assert.ok(front && side && palm, `pose missing for left=${left} mirror=${mirror}`);
    assert.ok(stone(front).z > 0.99, `front stone for left=${left} mirror=${mirror}`);
    assert.ok(Math.abs(stone(side).x) > 0.8, `side stone for left=${left} mirror=${mirror}`);
    assert.ok(stone(palm).z < -0.95, `palm stone for left=${left} mirror=${mirror}`);
    // Softened palm depth squash keeps a rear profile without opening into a U.
    assert.ok(Math.abs(new Vector3(0, 1, 0).applyQuaternion(palm.orientation).z) < 0.16);
  }
});

test('the setting follows a sideways turn and is on the far side of an open palm', () => {
  for (const mirror of [false, true]) {
    const front = sample({ mirror }).pose;
    const side = sample({ mirror, yaw: 1.15 }).pose;
    const palm = sample({ mirror, yaw: Math.PI, pitch: 0.28 }).pose;
    const stone = p => new Vector3(0, 0, -1).applyQuaternion(p.orientation);
    assert.ok(stone(front).z > 0.99);
    assert.ok(Math.abs(stone(side).x) > 0.8);
    assert.ok(stone(palm).z < -0.95);
    // Softened palm depth squash keeps a rear profile without opening into a U.
    assert.ok(Math.abs(new Vector3(0, 1, 0).applyQuaternion(palm.orientation).z) < 0.16);
  }
});

test('continuous rotation crosses edge-on without a half-turn sign flip', () => {
  let previous;
  for (let yaw = 0; yaw <= Math.PI * 2; yaw += 0.025) {
    const pose = sample({ yaw }).pose;
    assert.ok(pose);
    if (previous) assert.ok(previous.angleTo(pose.orientation) < 0.12);
    previous = pose.orientation;
  }
});

test('ring diameter stays constant on tilt and responds to camera distance', () => {
  for (const pitch of [-0.8, 0, 0.8]) for (const yaw of [-1, 0, 1]) {
    const scale = sample({ pitch, yaw }).pose.handScale;
    near(scale, 100);
    near(sample({ pitch, yaw, pixelsPerMeter: 3000 }).pose.handScale / scale, 1.5);
  }
  const fit = new FingerFitFilter();
  let calibrated;
  for (const revision of [-80, -40, 0]) calibrated = fit.update(100, 1, { width: 40, offset: 0, revision, confidence: 1 });
  near(calibrated.width, 40);
  near(fit.update(150, 0.2).width, 60);
  near(fit.update(150, 0.2, { width: 18, offset: 0, revision: 2, confidence: 1 }).width, 60);
});

test('anatomicalHandedness swaps rear-camera labels and keeps selfie labels', () => {
  assert.equal(anatomicalHandedness('Left', 'environment'), 'Right');
  assert.equal(anatomicalHandedness('Right', 'environment'), 'Left');
  assert.equal(anatomicalHandedness('Left', 'user'), 'Left');
  assert.equal(anatomicalHandedness('Right', 'user'), 'Right');
  assert.equal(anatomicalHandedness(null, 'environment'), null);
  assert.equal(swapHandedness('Left'), 'Right');
});

test('left and right hands share the same handScale for mirrored geometry', () => {
  for (const mirror of [false, true]) {
    const left = sample({ left: false, mirror, yaw: 0.4, pitch: 0.2 }).pose.handScale;
    const right = sample({ left: true, mirror, yaw: 0.4, pitch: 0.2 }).pose.handScale;
    near(left, right, 1e-6);
  }
});

test('solveRingPose tracks index middle and ring bones on both hands', () => {
  const bones = [
    { mcp: 5, pip: 6 },
    { mcp: 9, pip: 10 },
    { mcp: 13, pip: 14 },
  ];
  for (const left of [false, true]) {
    for (const bone of bones) {
      const { pose, screen } = sample({ left, ...bone, yaw: 0.25 });
      assert.ok(pose);
      assert.ok(pose.fingerLength > 10);
      near(
        pose.fingerLength,
        Math.hypot(screen[bone.pip].x - screen[bone.mcp].x, screen[bone.pip].y - screen[bone.mcp].y),
        1e-6,
      );
      // Head (-Z) stays palm-facing for either hand.
      const normalZ = new Vector3(0, 0, 1).applyQuaternion(pose.orientation).z;
      assert.ok(Math.abs(normalZ) > 0.2, `facing too edge-on: ${normalZ}`);
    }
  }
});

test('FingerFitFilter locks the same width for either hand sample', () => {
  const widths = [];
  for (const left of [false, true]) {
    const fit = new FingerFitFilter();
    const anchors = sample({ left }).pose.anchors;
    let result;
    for (const revision of [0, 40, 80]) {
      result = fit.update(100, 1, { width: 40, offset: 0, revision, confidence: 1 }, anchors);
    }
    widths.push(result.width);
  }
  near(widths[0], widths[1]);
  near(widths[0], 40);
});

test('high-confidence edges with large lateral offset do not lock finger width', () => {
  const fit = new FingerFitFilter();
  const largeOffset = 40 * MAX_LOCK_OFFSET_RATIO * 2;
  let result = null;
  for (const revision of [0, 40, 80]) {
    result = fit.update(100, 1, { width: 40, offset: largeOffset, revision, confidence: 1 });
  }
  assert.equal(result, null);
  assert.equal(fit.calibrated, false);
});

test('small trustworthy lateral offset still locks and stays within placement gate', () => {
  const fit = new FingerFitFilter();
  const smallOffset = 40 * PLACEMENT_OFFSET_GATE * 0.5;
  let result;
  for (const revision of [0, 40, 80]) {
    result = fit.update(100, 1, { width: 40, offset: smallOffset, revision, confidence: 1 });
  }
  assert.ok(result);
  near(result.width, 40);
  assert.ok(Math.abs(result.offset) <= 40 * MAX_FIT_OFFSET_RATIO + 1e-6);
  assert.ok(Math.abs(result.offset) <= 40 * PLACEMENT_OFFSET_GATE + 1e-6);
});

test('moderate off-center edges lock width for left and right hands', () => {
  const offset = 40 * 0.18;
  assert.ok(offset / 40 > 0.12);
  assert.ok(offset / 40 < MAX_LOCK_OFFSET_RATIO);
  for (const left of [false, true]) {
    const fit = new FingerFitFilter();
    const hand = left ? 'Right' : 'Left';
    const anchors = sample({ left }).pose.anchors;
    let result;
    for (const revision of [0, 40, 80]) {
      result = fit.update(
        100, 1,
        { width: 40, offset, revision, confidence: 1 },
        anchors,
        { finger: 'ring', hand },
      );
    }
    assert.equal(fit.calibrated, true);
    assert.ok(result);
    near(result.width, 40);
    // Diameter locks; the nudge stays outside the placement gate so the ring stays on the bone.
    assert.ok(Math.abs(result.offset) <= 40 * MAX_FIT_OFFSET_RATIO + 1e-6);
    assert.ok(Math.abs(result.offset) > 40 * PLACEMENT_OFFSET_GATE);
  }
});

test('each hand keeps its own palm scale when switching back', () => {
  const fit = new FingerFitFilter();
  const rightAnchors = sample({ pixelsPerMeter: 2000 }).pose.anchors;
  const leftAnchors = sample({ left: true, pixelsPerMeter: 3200 }).pose.anchors;
  const edge = (revision, width) => ({ width, offset: 0, revision, confidence: 1 });
  let right;
  for (const revision of [0, 40]) {
    right = fit.update(100, 1, edge(revision, 40), rightAnchors, { finger: 'ring', hand: 'Right' });
  }
  assert.equal(fit.calibrated, true);
  near(right.width, 40);
  fit.update(100, 1, edge(80, 52), leftAnchors, { finger: 'ring', hand: 'Left' });
  let left;
  for (let i = 0; i < 8; i += 1) {
    left = fit.update(100, 1, edge(120 + i * 40, 52), leftAnchors, { finger: 'ring', hand: 'Left' });
  }
  assert.equal(fit.lockedHand, 'Left');
  assert.ok(left);
  near(left.width, 52);
  fit.update(100, 1, edge(500, 40), rightAnchors, { finger: 'ring', hand: 'Right' });
  const restored = fit.update(100, 1, edge(540, 40), rightAnchors, { finger: 'ring', hand: 'Right' });
  assert.equal(fit.lockedHand, 'Right');
  assert.equal(fit.calibrated, true);
  near(restored.width, 40, 1);
});

test('sustained off-bone edges after lock reset calibration', () => {
  const fit = new FingerFitFilter();
  for (const revision of [0, 40, 80]) {
    fit.update(100, 1, { width: 40, offset: 0, revision, confidence: 1 });
  }
  assert.equal(fit.calibrated, true);
  const drifted = 40 * MAX_LOCK_OFFSET_RATIO * 2;
  let result = fit.update(100, 1, { width: 40, offset: 0, revision: 120, confidence: 1 });
  assert.ok(result);
  for (let i = 0; i < 5; i += 1) {
    result = fit.update(100, 1, {
      width: 40,
      offset: drifted,
      revision: 160 + i * 40,
      confidence: 1,
    });
  }
  assert.equal(result, null);
  assert.equal(fit.calibrated, false);
});

test('HandednessResolver keeps camera-canonical labels and does not override left dorsal', () => {
  const resolver = new HandednessResolver();
  // Right-dorsal silhouette (also matches left-palm) — winding alone would say Right.
  const rightShaped = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  rightShaped[0] = { x: 0.5, y: 0.7, z: 0 };
  rightShaped[5] = { x: 0.35, y: 0.45, z: 0 };
  rightShaped[17] = { x: 0.65, y: 0.45, z: 0 };
  rightShaped[4] = { x: 0.28, y: 0.55, z: 0 };
  // Left-dorsal silhouette — same "opposite-sign" shape as right-palm; must not
  // force Right over a camera-correct Left label.
  const leftDorsal = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  leftDorsal[0] = { x: 0.5, y: 0.7, z: 0 };
  leftDorsal[5] = { x: 0.65, y: 0.45, z: 0 };
  leftDorsal[17] = { x: 0.35, y: 0.45, z: 0 };
  leftDorsal[4] = { x: 0.72, y: 0.55, z: 0 };

  let label;
  for (let i = 0; i < 5; i++) {
    label = resolver.resolve('Left', 'user', leftDorsal);
  }
  assert.equal(label, 'Left');

  // Rear camera: MediaPipe's selfie-trained label is swapped once, then held.
  for (let i = 0; i < 5; i++) {
    label = resolver.resolve('Right', 'environment', leftDorsal);
  }
  assert.equal(label, 'Left');

  for (let i = 0; i < 5; i++) {
    label = resolver.resolve('Left', 'user', rightShaped);
  }
  assert.equal(label, 'Left');
});

test('left and right dorsal poses keep the setting on the same anatomical side', () => {
  for (const mirror of [false, true]) {
    const left = sample({ left: false, mirror }).pose;
    const right = sample({ left: true, mirror }).pose;
    assert.ok(left && right);
    const stone = (p) => new Vector3(0, 0, -1).applyQuaternion(p.orientation);
    // Dorsal view: setting toward camera for both hands.
    assert.ok(stone(left).z > 0.99, `left stone z=${stone(left).z}`);
    assert.ok(stone(right).z > 0.99, `right stone z=${stone(right).z}`);
    // Local +Z (palm normal) points away from camera for both.
    const n = (p) => new Vector3(0, 0, 1).applyQuaternion(p.orientation);
    assert.ok(n(left).z < -0.99);
    assert.ok(n(right).z < -0.99);
  }
});

test('left and right hands share dorsal head and palm rear shank after surface constraint', () => {
  const stone = (orientation) => new Vector3(0, 0, -1).applyQuaternion(orientation);
  for (const left of [false, true]) {
    for (const mirror of [false, true]) {
      const constraint = new RingSurfaceConstraint();
      const dorsal = sample({ left, mirror });
      constraint.update(dorsal.pose.orientation, dorsal.pose.anchors, dorsal.pose.fingerLength, 40);
      const dorsalOri = constraint.update(
        dorsal.pose.orientation, dorsal.pose.anchors, dorsal.pose.fingerLength, 40,
      );
      assert.ok(
        stone(dorsalOri).z > 0.99,
        `dorsal head left=${left} mirror=${mirror} z=${stone(dorsalOri).z}`,
      );

      let palmOri = dorsalOri;
      for (let yaw = 0; yaw <= Math.PI; yaw += 0.05) {
        const s = sample({ left, mirror, yaw, pitch: yaw > 2 ? 0.28 : 0 });
        palmOri = constraint.update(s.pose.orientation, s.pose.anchors, s.pose.fingerLength, 40);
      }
      assert.ok(
        stone(palmOri).z < -0.95,
        `palm shank left=${left} mirror=${mirror} z=${stone(palmOri).z}`,
      );
    }
  }
});

test('fast translation and reversal stay with the finger on the first frame', () => {
  const filter = new RingPoseFilter();
  filter.update(placement(sample()), 0);
  const moved = placement(sample({ tx: 120, ty: 1250 }));
  const result = filter.update(moved, 33);
  near(result.x, moved.x); near(result.y, moved.y);
  const reverse = placement(sample({ tx: 620, ty: 350 }));
  const returned = filter.update(reverse, 66);
  near(returned.x, reverse.x); near(returned.y, reverse.y);
});

test('filtering a turn cannot shrink the ring with the foreshortened palm', () => {
  const filter = new RingPoseFilter();
  // Size lock supplies a stable target width; pose turns must not shrink display scale.
  for (let i = 0; i < 30; i++) {
    const warm = placement(sample());
    warm.width = 40;
    filter.update(warm, i * 16.667);
  }
  for (let i = 30; i < 80; i++) {
    const raw = placement(sample({ yaw: (i - 30) * 0.018 }));
    raw.width = 40;
    const result = filter.update(raw, i * 16.667);
    near(result.width, 40, 0.05);
  }
});

test('small continuous movement does not wait for a large distance threshold', () => {
  const filter = new RingPoseFilter();
  for (let i = 0; i < 60; i++) {
    const raw = placement(sample({ tx: 420 + i * 0.75 }));
    const result = filter.update(raw, i * 16.667);
    assert.ok(Math.hypot(result.x - raw.x, result.y - raw.y) < 0.7);
  }
});

test('RingPoseFilter holds tiny width flicker and eases toward a sustained jump', () => {
  const filter = new RingPoseFilter();
  const initial = placement(sample());
  initial.width = 40;
  // Warm past ease-in so deadband/catch-up behavior is measurable.
  for (let i = 0; i < 30; i++) {
    filter.update({ ...initial, width: 40 }, i * 16.667);
  }
  let held = null;
  for (let i = 30; i < 50; i++) {
    const flicker = 40 * (i % 2 ? 1.015 : 0.985);
    held = filter.update({ ...initial, width: flicker }, i * 16.667);
    near(held.width, 40, 0.5);
  }
  assert.ok(held);
  // Sustained +20% target must ease (not snap on the first frame).
  const jumped = filter.update({ ...initial, width: 48 }, 50 * 16.667);
  assert.ok(jumped.width < 47, `expected ease not snap, got ${jumped.width}`);
  let caught = jumped;
  for (let i = 51; i < 90; i++) {
    caught = filter.update({ ...initial, width: 48 }, i * 16.667);
  }
  assert.ok(Math.abs(caught.width / 48 - 1) < 0.03, `expected catch-up toward 48, got ${caught.width}`);
});

test('RingPoseFilter eases in display width on first lock', () => {
  const filter = new RingPoseFilter();
  const raw = placement(sample());
  raw.width = 40;
  const first = filter.update(raw, 0);
  assert.ok(first.width < 40);
  assert.ok(first.width > 30);
  near(first.width, 40 * 0.85, 1e-6);
  let later = first;
  for (let i = 1; i <= 40; i++) {
    later = filter.update({ ...raw, width: 40 }, i * 16.667);
  }
  assert.ok(later.width > first.width, 'width should rise toward the locked target');
  near(later.width, 40, 0.5);
});

test('RingPoseFilter keeps smoothing XY when width target jumps', () => {
  const filter = new RingPoseFilter();
  const base = placement(sample());
  base.width = 40;
  for (let i = 0; i < 20; i++) filter.update({ ...base, width: 40 }, i * 16.667);
  const moved = { ...base, x: base.x + 2, width: 52 };
  const result = filter.update(moved, 20 * 16.667);
  // Position still blends; width does not snap to the new target in one frame.
  assert.ok(Math.abs(result.x - moved.x) < 2.1);
  assert.ok(result.width < 51);
});

test('FingerFitFilter holds locked size when palm distance briefly fails', () => {
  const fit = new FingerFitFilter();
  const anchors = sample().pose.anchors;
  let locked;
  for (const revision of [0, 40, 80]) {
    locked = fit.update(100, 1, { width: 40, offset: 0, revision, confidence: 1 }, anchors);
  }
  assert.ok(locked);
  near(locked.width, 40);
  // Degenerate anchors make PalmScaleLock return null (side view).
  const badAnchors = anchors.map(() => ({ x: 0, y: 0 }));
  const held = fit.update(100, 0.2, undefined, badAnchors);
  assert.ok(held, 'calibrated fit must survive a bad palm frame');
  near(held.width, 40);
  assert.equal(fit.status, 'side view');
});

test('new palm edges cannot silently replace an established fit', () => {
  const fit = new FingerFitFilter();
  for (const revision of [-80, -40, 0]) fit.update(100, 1, { width: 40, offset: 0, revision, confidence: 1 });
  for (let revision = 40; revision < 1000; revision += 40) {
    near(fit.update(100, 0.95, { width: 56, offset: 0, revision, confidence: 1 }).width, 40);
  }
});

test('a single bad edge measurement cannot resize the ring', () => {
  const fit = new FingerFitFilter();
  for (const revision of [-80, -40, 0]) fit.update(100, 1, { width: 40, offset: 0, revision, confidence: 1 });
  near(fit.update(100, 1, { width: 58, offset: 0, revision: 2, confidence: 1 }).width, 40);
  near(fit.update(100, 1, { width: 40, offset: 0, revision: 3, confidence: 1 }).width, 40);
});

test('stationary ring-point noise is suppressed while palm landmarks remain fixed', () => {
  const filter = new RingPoseFilter();
  const raw = placement(sample());
  filter.update(raw, 0);
  let energy = 0;
  for (let i = 1; i <= 120; i++) {
    const noisy = { ...raw, x: raw.x + (i % 2 ? 2 : -2), y: raw.y + (i % 2 ? -2 : 2) };
    const result = filter.update(noisy, i * 16.667);
    energy += (result.x - raw.x) ** 2 + (result.y - raw.y) ** 2;
  }
  assert.ok(Math.sqrt(energy / 120) < 0.8);
});

test('alternating depth estimates do not repeatedly release the fast rotation filter', () => {
  const filter = new RingPoseFilter();
  const initial = placement(sample()); filter.update(initial, 0);
  let maximum = 0;
  for (let i = 1; i <= 120; i++) {
    const noisy = placement(sample({ yaw: i % 2 ? 0.065 : -0.065 }));
    const value = filter.update(noisy, i * 16.667);
    maximum = Math.max(maximum, value.orientation.angleTo(initial.orientation));
  }
  assert.ok(maximum < 0.02, `angular shimmer ${maximum} radians`);
});

test('a sustained real rotation follows promptly after its first frame', () => {
  const filter = new RingPoseFilter();
  for (let i = 0; i < 40; i++) {
    const raw = placement(sample({ yaw: i * 0.035 }));
    const value = filter.update(raw, i * 16.667);
    assert.ok(value.orientation.angleTo(raw.orientation) < 0.05);
  }
});

test('loss and reacquisition do not animate from an old location', () => {
  const filter = new RingPoseFilter();
  filter.update(placement(sample()), 0);
  assert.ok(filter.update(null, 30));
  assert.ok(filter.update(null, 300), 'brief gaps keep the last pose for ~400ms');
  assert.equal(filter.update(null, 450), null);
  const moved = placement(sample({ tx: 600, ty: 1100 }));
  const reacquired = filter.update(moved, 600);
  near(reacquired.x, moved.x); near(reacquired.y, moved.y);
});

test('RingPoseFilter holds last pose through a brief MediaPipe gap', () => {
  const filter = new RingPoseFilter();
  const first = placement(sample());
  filter.update(first, 0);
  const held = filter.update(null, 200);
  assert.ok(held);
  near(held.x, first.x);
  near(held.y, first.y);
  assert.equal(filter.update(null, 401), null);
});

test('the overlay does not add stone parallax when translated to a corner', () => {
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  camera.position.z = 4; camera.updateMatrixWorld();
  const a = new Vector3(0, 0, 0.2).project(camera);
  const b = new Vector3(0.8, -0.8, 0.2).project(camera);
  near(b.x - a.x, 0.8); near(b.y - a.y, -0.8);
});

test('LR1844 inner diameter is measured from the metal band', async () => {
  const asset = await loadRingGltf();
  const group = new Group();
  group.add(asset.scene);
  const bounds = new Box3().setFromObject(group);
  const center = bounds.getCenter(new Vector3());
  asset.scene.position.x -= center.x;
  asset.scene.position.y -= center.y;
  group.updateMatrixWorld(true);
  const outer = bounds.getSize(new Vector3()).x;
  const extras = readRingModelExtras(asset.scene);
  if (extras.innerDiameter != null) assert.equal(extras.innerDiameter, 1.91);
  if (extras.stoneAxis != null) assert.equal(extras.stoneAxis, '-Z');
  const measured = measureRingInnerDiameter(group, DEFAULT_INNER_TO_OUTER_DIAMETER);
  near(measured, 1.91, 0.08);
  assert.ok(measured < outer * 0.95);
  assert.ok(measured > outer * 0.5);
  group.traverse(o => {
    if (o.isMesh) {
      o.geometry.dispose();
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose();
    }
  });
});

test('every catalog ring wears like LR1844: hole along the finger, head on -Z', async () => {
  const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');
  const files = ['LR1844.glb', 'LR1025.glb', 'LR1032.glb', 'LR1035.glb', 'LR1101.glb', 'LR1157.glb'];
  for (const file of files) {
    const asset = await loadRingGltf(join(root, file));
    const model = new Group();
    model.add(asset.scene);
    model.updateMatrixWorld(true);
    const extras = readRingModelExtras(asset.scene);
    if (!extras.stoneAxis) {
      const inferred = inferWearStoneAxis(asset.scene);
      assert.equal(inferred, '+Y', `${file} should be a +Y head / Z hole`);
      applyStoneAxisBasis(asset.scene, inferred);
    }
    centerRingOnMetalHole(model, asset.scene);
    const metal = measureMetalBounds(model);
    assert.ok(metal, `${file} metal bounds`);
    assert.ok(metal.size.y < metal.size.x * 0.85, `${file} finger axis is not the thin band (y=${metal.size.y} x=${metal.size.x})`);
    assert.ok(metal.size.y < metal.size.z * 0.85, `${file} head axis collapsed into the band`);
    const raw = resolveRawInnerDiameter(model, {
      authored: extras.innerDiameter,
      hint: metal.size.x * DEFAULT_INNER_TO_OUTER_DIAMETER,
    });
    if (file === 'LR1844.glb') {
      near(raw, 1.91, 0.02);
    } else {
      const fallback = metal.size.x * DEFAULT_INNER_TO_OUTER_DIAMETER;
      assert.ok(Math.abs(raw - fallback) > 0.01, `${file} still used the outer-width fallback ${fallback} (raw ${raw})`);
      assert.ok(raw > metal.size.x * 0.5 && raw < metal.size.x * 0.95, `${file} inner ${raw} vs outer ${metal.size.x}`);
    }
    if (file !== 'LR1035.glb') {
      let weight = 0;
      const gem = new Vector3();
      model.updateMatrixWorld(true);
      model.traverse((object) => {
        if (!object.isMesh || !/gem|diamond|stone/i.test(`${object.name} ${object.material?.name || ''}`)) return;
        const box = new Box3().setFromObject(object);
        const size = box.getSize(new Vector3());
        const volume = Math.abs(size.x * size.y * size.z);
        gem.addScaledVector(box.getCenter(new Vector3()), volume);
        weight += volume;
      });
      gem.multiplyScalar(1 / weight);
      assert.ok(gem.z < -0.25, `${file} head z=${gem.z}`);
      assert.ok(Math.abs(gem.z) > Math.abs(gem.y), `${file} head still along the finger`);
    }
    model.traverse((object) => {
      if (object.isMesh) {
        object.geometry.dispose();
        for (const material of [].concat(object.material)) material.dispose?.();
      }
    });
  }
});

test('LR1844 normalizes to a unit hole then scales with wear clearance', async () => {
  const asset = await loadRingGltf();
  const model = new Group();
  model.add(asset.scene);
  const bounds = new Box3().setFromObject(model);
  const center = bounds.getCenter(new Vector3());
  asset.scene.position.x -= center.x;
  asset.scene.position.y -= center.y;
  model.updateMatrixWorld(true);
  const extras = readRingModelExtras(asset.scene);
  const raw = extras.innerDiameter ?? measureRingInnerDiameter(model, DEFAULT_INNER_TO_OUTER_DIAMETER);
  normalizeRingToUnitHole(model, raw);
  assert.equal(model.userData.innerDiameter, 1);
  near(model.userData.rawInnerDiameter, raw, 1e-6);
  const fingerWorld = 0.12;
  const scale = ringWorldScale(fingerWorld);
  near(scale, fingerWorld * WEAR_CLEARANCE, 1e-9);
  assert.ok(scale > fingerWorld);
  model.traverse(o => {
    if (o.isMesh) {
      o.geometry.dispose();
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose();
    }
  });
});

test('bone-clamped finger width rejects collapsed and inflated silhouettes', () => {
  const bone = 120;
  near(resolveFingerWidthPx(36, bone), 36);
  assert.ok(resolveFingerWidthPx(4, bone) >= bone * 0.18);
  assert.ok(resolveFingerWidthPx(100, bone) <= bone * 0.52);
  near(resolveFingerWidthPx(50, bone), 50);
});

test('ring model extras prefer authored innerDiameter when present', () => {
  const root = new Group();
  root.userData = { extras: { innerDiameter: 2.05, stoneAxis: '-Z', metalTint: '#d4af37' } };
  const extras = readRingModelExtras(root);
  assert.equal(extras.innerDiameter, 2.05);
  assert.equal(extras.stoneAxis, '-Z');
  assert.equal(extras.metalTint, '#d4af37');
});

test('LR1844 palm view hides its stone behind the finger and retains a thin rear band', async () => {
  const asset = await loadRingGltf();
  const model = new Group();
  model.add(asset.scene);
  centerRingOnMetalHole(model, asset.scene);
  const extras = readRingModelExtras(asset.scene);
  const raw = extras.innerDiameter ?? measureRingInnerDiameter(model, DEFAULT_INNER_TO_OUTER_DIAMETER);
  normalizeRingToUnitHole(model, raw);
  centerRingOnMetalHole(model, model);
  // Exact finger diameter (no wear clearance) for the depth mask.
  const fingerWorld = 1;
  const ringGroup = new Group();
  ringGroup.add(model);
  ringGroup.scale.setScalar(ringWorldScale(fingerWorld, 1));
  const palm = sample({ yaw: Math.PI, pitch: 0.28 }).pose;
  ringGroup.quaternion.copy(palm.orientation);
  const mask = new Mesh(createFingerOccluderGeometry(), new MeshBasicMaterial());
  mask.quaternion.copy(palm.orientation);
  mask.scale.set(
    fingerWorld * OCCLUSION_RADIUS_RATIO,
    fingerWorld * OCCLUSION_LENGTH_RATIO,
    fingerWorld * OCCLUSION_RADIUS_RATIO,
  );
  mask.updateMatrixWorld(true);
  ringGroup.updateMatrixWorld(true);
  const ray = new Raycaster();
  const visible = [];
  let visibleGems = 0;
  for (let ix = -35; ix <= 35; ix++) for (let iy = -14; iy <= 14; iy++) {
    const x = ix / 60 * fingerWorld, y = iy / 60 * fingerWorld;
    ray.set(new Vector3(x, y, 5), new Vector3(0, 0, -1));
    const ringHit = ray.intersectObject(ringGroup, true)[0];
    const fingerHit = ray.intersectObject(mask, false)[0];
    if (ringHit && (!fingerHit || ringHit.distance < fingerHit.distance)) {
      visible.push({ x, y });
      if (/gem|diamond|stone/i.test(ringHit.object.name)) visibleGems++;
    }
  }
  assert.ok(visible.length > 20, 'rear band must remain visible');
  assert.equal(visibleGems, 0, 'the palm must occlude every visible stone sample');
  const spanX = Math.max(...visible.map(p => p.x)) - Math.min(...visible.map(p => p.x));
  const spanY = Math.max(...visible.map(p => p.y)) - Math.min(...visible.map(p => p.y));
  // Cleared ring is slightly wider than the finger; band must still reach both edges.
  assert.ok(spanX >= fingerWorld * 0.95, 'band must reach both finger edges');
  // Softened palm depth keeps a short rear shank (not a full open U).
  assert.ok(spanY < fingerWorld * 0.36, `rear band too deep: ${spanY / fingerWorld}`);
  mask.geometry.dispose(); mask.material.dispose();
  ringGroup.traverse(o => { if (o.isMesh) { o.geometry.dispose(); for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose(); } });
});

test('initial calibration sits on the measured center immediately', () => {
  const fit = new FingerFitFilter();
  let result;
  // Offset within lock + placement gates so calibration can accept it.
  const offset = 40 * 0.05;
  for (const revision of [0, 40, 80]) result = fit.update(100, 1, { width: 40, offset, revision, confidence: 1 });
  near(result.offset, offset);
  // A sub-frame edge-center flip must only nudge the fitted offset (not snap).
  const jumped = fit.update(100, 1, { width: 40, offset: -offset, revision: 81, confidence: 1 });
  assert.ok(Math.abs(jumped.offset - offset) < Math.abs(offset) * 0.5);
});

test('sustained finger motion follows promptly even when the palm stays still', () => {
  const filter = new RingPoseFilter();
  const raw = placement(sample());
  filter.update(raw, 0);
  for (let i = 1; i <= 30; i++) {
    const moved = { ...raw, x: raw.x + i * 1.5 };
    const result = filter.update(moved, i * 16.667);
    assert.ok(Math.abs(result.x - moved.x) < (i === 1 ? 1.6 : 1.3), `finger lag: ${result.x - moved.x}`);
  }
});

test('rejected optical flow falls back to valid palm motion', () => {
  const filter = new RingPoseFilter();
  filter.update(placement(sample()), 0);
  const raw = placement(sample({ tx: 423 }));
  raw.imageMotion = {
    before: [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 0, y: 10 }],
    after: [{ x: 0, y: 0 }, { x: 30, y: 0 }, { x: 0, y: 30 }],
  };
  const result = filter.update(raw, 33);
  near(result.x, raw.x); near(result.y, raw.y);
});

test('a strongly pitched hand uses the remaining reliable palm span for scale', () => {
  const s = sample({ pitch: 1.4 });
  near(s.pose.handScale, 100);
  near(sample({ pitch: 1.4, pixelsPerMeter: 3000 }).pose.handScale, 150);
});

function lockedFit() {
  const filter = new FingerFitFilter();
  const anchors = sample().pose.anchors;
  for (const revision of [0, 40, 80]) filter.update(100, 1, { width: 40, offset: 0, revision, confidence: 1 }, anchors);
  return filter;
}

test('metric hand-size noise and repeated edge corrections cannot pump a stationary ring', () => {
  const filter = lockedFit();
  const anchors = sample().pose.anchors;
  for (let i = 0; i < 300; i++) {
    const result = filter.update(60 + (i % 7) * 20, 1,
      { width: i % 2 ? 48 : 33, offset: 0, revision: 120 + i * 33, confidence: 1 }, anchors);
    near(result.width, 40);
  }
});

test('image distance stays fixed through palm turns without relying on world bone size', () => {
  const filter = lockedFit();
  for (let i = 0; i <= 120; i++) {
    const s = sample({ yaw: i * Math.PI / 120, pitch: 0.2, roll: 0.7, tx: 500, ty: 400 });
    const result = filter.update(i % 2 ? 60 : 160, s.pose.facing, undefined, s.pose.anchors);
    near(result.width, 40);
  }
});

test('a real camera-distance change resizes the locked fit in both directions', () => {
  const filter = lockedFit();
  for (const factor of [1.5, 0.65, 1]) {
    let result;
    const anchors = sample({ pixelsPerMeter: 2000 * factor, yaw: 0.9 }).pose.anchors;
    for (let i = 0; i < 30; i++) result = filter.update(100, 0.6, undefined, anchors);
    assert.ok(Math.abs(result.width / (40 * factor) - 1) <= 0.026, `distance scale ${result.width}`);
  }
});

test('a large sustained camera-distance step catches up within a short window', () => {
  const expectNear = (actual, expected, tol = 0.08) => {
    assert.ok(Math.abs(actual / expected - 1) <= tol, `got ${actual}, expected ~${expected}`);
  };
  {
    const filter = lockedFit();
    const far = sample({ pixelsPerMeter: 1200 }).pose.anchors;
    let result;
    for (let i = 0; i < 30; i++) result = filter.update(100, 0.85, undefined, far);
    expectNear(result.width, 24);
  }
  {
    const filter = lockedFit();
    const near = sample({ pixelsPerMeter: 3200 }).pose.anchors;
    let result;
    for (let i = 0; i < 30; i++) result = filter.update(100, 0.85, undefined, near);
    expectNear(result.width, 64);
  }
});

test('small image-size noise and one bad palm frame cannot resize the ring', () => {
  const filter = lockedFit();
  for (let i = 0; i < 60; i++) {
    const factor = i === 30 ? 1.8 : i % 2 ? 1.015 : 0.985;
    near(filter.update(100, 1, undefined, sample({ pixelsPerMeter: 2000 * factor }).pose.anchors).width, 40);
  }
});

test('refitting clears the distance reference and accepts a new finger width', () => {
  const filter = lockedFit(); filter.reset();
  const anchors = sample({ pixelsPerMeter: 3000 }).pose.anchors;
  let result;
  for (const revision of [0, 40, 80]) result = filter.update(150, 1, { width: 75, offset: 0, revision, confidence: 1 }, anchors);
  near(result.width, 75);
});

test('initial image reference waits for a front-facing hand', () => {
  const filter = new FingerFitFilter();
  const anchors = sample({ yaw: 1.2 }).pose.anchors;
  for (const revision of [0, 40, 80]) {
    assert.equal(filter.update(100, 0.4, { width: 40, offset: 0, revision, confidence: 1 }, anchors), null);
  }
  assert.equal(filter.calibrated, false);
});

test('a deformed palm fit holds size instead of inventing a camera-distance change', () => {
  const filter = lockedFit();
  const anchors = sample().pose.anchors.map((p, i) => ({ x: p.x + (i === 2 ? 100 : 0), y: p.y }));
  for (let i = 0; i < 30; i++) near(filter.update(200, 1, undefined, anchors).width, 40);
});

test('an open finger does not inherit invented world pitch and twist', () => {
  const constraint = new RingSurfaceConstraint();
  const flat = sample();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  for (const pitch of [-0.45, 0.45]) for (const yaw of [-0.7, 0.7]) {
    const noisy = sample({ pitch, yaw });
    const result = constraint.update(noisy.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
    const axis = new Vector3(0, 1, 0).applyQuaternion(result);
    const stone = new Vector3(0, 0, -1).applyQuaternion(result);
    near(axis.z, 0);
    near(stone.x, 0); near(stone.y, 0);
    near(stone.z, 1);
  }
});

test('image-constrained surface turns continuously from back to palm', () => {
  const constraint = new RingSurfaceConstraint();
  const flat = sample();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  let previous;
  for (let yaw = 0; yaw < Math.PI + 0.01; yaw += 0.02) {
    const s = sample({ yaw });
    const result = constraint.update(s.pose.orientation, s.pose.anchors, s.pose.fingerLength, 40);
    if (previous) assert.ok(previous.angleTo(result) < 0.14);
    if (yaw > 3) assert.ok(new Vector3(0, 0, -1).applyQuaternion(result).z < -0.95);
    previous = result;
  }
});

test('constrained orientation shows setting, band side, then rear shank by finger view', () => {
  const constraint = new RingSurfaceConstraint();
  const flat = sample();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  const stoneOf = (yaw) => {
    const s = sample({ yaw });
    return new Vector3(0, 0, -1).applyQuaternion(
      constraint.update(s.pose.orientation, s.pose.anchors, s.pose.fingerLength, 40),
    );
  };
  // Re-seed so each sample advances from a shared dorsal reference without
  // carrying a midway state from the helper above.
  constraint.reset();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  const dorsal = stoneOf(0);
  assert.ok(dorsal.z > 0.99, 'dorsal must show the setting toward the camera');

  constraint.reset();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  let edge;
  for (let yaw = 0; yaw <= 1.15; yaw += 0.05) edge = stoneOf(yaw);
  assert.ok(Math.abs(edge.x) > 0.7, 'edge-on must present the band profile, not a forced frontal head');
  assert.ok(Math.abs(edge.z) < 0.75, 'edge-on must not snap the setting fully front-facing');

  constraint.reset();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  let palm;
  for (let yaw = 0; yaw <= Math.PI; yaw += 0.05) palm = stoneOf(yaw);
  assert.ok(palm.z < -0.95, 'palm must keep the setting behind the finger for the rear shank');
});

test('palm-first surface lock still shows rear shank then dorsal head after a turn', () => {
  const constraint = new RingSurfaceConstraint();
  const palm = sample({ yaw: Math.PI, pitch: 0.28 });
  constraint.update(palm.pose.orientation, palm.pose.anchors, palm.pose.fingerLength, 40);
  const stonePalm = new Vector3(0, 0, -1).applyQuaternion(
    constraint.update(palm.pose.orientation, palm.pose.anchors, palm.pose.fingerLength, 40),
  );
  assert.ok(stonePalm.z < -0.95, 'palm-seeded lock must show rear shank');

  // Turn continuously back toward dorsal; head must face the camera again.
  let stone;
  for (let yaw = Math.PI; yaw >= 0; yaw -= 0.05) {
    const s = sample({ yaw, pitch: yaw > 2 ? 0.28 : 0 });
    stone = new Vector3(0, 0, -1).applyQuaternion(
      constraint.update(s.pose.orientation, s.pose.anchors, s.pose.fingerLength, 40),
    );
  }
  assert.ok(stone.z > 0.95, 'after turning to dorsal the setting must face the camera');
});

test('after size lock the ring keeps placing through side and palm turns', () => {
  const fit = lockedFit();
  const constraint = new RingSurfaceConstraint();
  const flat = sample();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  for (const yaw of [0, 0.8, 1.4, Math.PI / 2, 2.4, Math.PI]) {
    const s = sample({ yaw });
    const size = fit.update(100, s.pose.facing, undefined, s.pose.anchors);
    assert.ok(size && size.width > 0, `fit must survive yaw=${yaw}`);
    near(size.width, 40);
    const orientation = constraint.update(
      s.pose.orientation, s.pose.anchors, s.pose.fingerLength, size.width,
    );
    assert.ok(orientation);
  }
});

test('true proximal foreshortening still permits ring pitch', () => {
  const constraint = new RingSurfaceConstraint();
  const flat = sample();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  const bent = sample({ pitch: -0.85 });
  const result = constraint.update(bent.pose.orientation, bent.pose.anchors, bent.pose.fingerLength, 40);
  const axis = new Vector3(0, 1, 0).applyQuaternion(result);
  assert.ok(axis.z < -0.6, 'a real pitch cannot be flattened');
  const projected = projectedFingerAxis(result);
  const expected = projectedFingerAxis(bent.pose.orientation);
  near(projected.x, expected.x); near(projected.y, expected.y);
});

test('surface constraints are invariant to camera distance and clear on refit', () => {
  const constraint = new RingSurfaceConstraint();
  const flat = sample();
  constraint.update(flat.pose.orientation, flat.pose.anchors, flat.pose.fingerLength, 40);
  for (const factor of [0.7, 1.5]) {
    const s = sample({ pixelsPerMeter: 2000 * factor });
    near(constraint.update(s.pose.orientation, s.pose.anchors, s.pose.fingerLength, 40 * factor).angleTo(flat.pose.orientation), 0);
  }
  constraint.reset();
  const palm = sample({ yaw: Math.PI });
  const result = constraint.update(palm.pose.orientation, palm.pose.anchors, palm.pose.fingerLength, 40);
  assert.ok(new Vector3(0, 0, -1).applyQuaternion(result).z < -0.99);
});

test('high-frequency landmark noise on stationary hand produces rock-solid ring stability with minimal RMS jitter', () => {
  const filter = new RingPoseFilter();
  const base = placement(sample());
  filter.update(base, 0);
  let totalPositionVariance = 0;
  let maxAngularDeviation = 0;
  for (let i = 1; i <= 90; i++) {
    const noiseX = (i % 2 === 0 ? 1.5 : -1.5);
    const noiseY = (i % 2 === 0 ? -1.2 : 1.2);
    const noiseYaw = (i % 2 === 0 ? 0.03 : -0.03);
    const samplePose = sample({ yaw: noiseYaw });
    const noisy = {
      ...placement(samplePose),
      x: base.x + noiseX,
      y: base.y + noiseY,
      anchors: base.anchors,
    };
    const result = filter.update(noisy, i * 33.33);
    totalPositionVariance += (result.x - base.x) ** 2 + (result.y - base.y) ** 2;
    maxAngularDeviation = Math.max(maxAngularDeviation, result.orientation.angleTo(base.orientation));
  }
  const rmsError = Math.sqrt(totalPositionVariance / 90);
  assert.ok(rmsError < 0.65, `RMS position jitter should be suppressed: ${rmsError}`);
  assert.ok(maxAngularDeviation < 0.015, `Angular wobble should be suppressed: ${maxAngularDeviation} rad`);
});

test('anatomicalChiralityFromWorld accurately identifies left vs right hands with 3D thumb across yaw/pitch', () => {
  // Construct metric 3D hands with wrist(0), index(5), pinky(17), and thumb(2)
  for (const yaw of [-0.6, -0.3, 0, 0.3, 0.6]) {
    for (const pitch of [-0.4, 0, 0.4]) {
      const q = new Quaternion().setFromEuler(new Euler(pitch, yaw, 0, 'ZYX'));

      // Right hand: index on left (-X), pinky on right (+X), thumb on radial side (-X, -Y, -Z)
      const rightWorld = [
        new Vector3(0, -0.055, 0).applyQuaternion(q), // wrist 0
        new Vector3(0, 0, 0), // 1
        new Vector3(-0.045, -0.02, -0.01).applyQuaternion(q), // thumb 2
        new Vector3(0, 0, 0), // 3
        new Vector3(0, 0, 0), // 4
        new Vector3(-0.035, 0.005, 0).applyQuaternion(q), // index 5
        ...Array.from({ length: 11 }, () => new Vector3(0, 0, 0)),
        new Vector3(0.032, 0, 0).applyQuaternion(q), // pinky 17
      ];
      const rightLandmarks = rightWorld.map((p) => ({
        x: p.x, y: p.y, z: p.z,
        worldX: p.x, worldY: p.y, worldZ: p.z,
      }));
      assert.equal(anatomicalChiralityFromWorld(rightLandmarks), 'Right', `Should identify Right hand at yaw=${yaw}, pitch=${pitch}`);

      // Left hand: mirror reflection across X
      const leftWorld = [
        new Vector3(0, -0.055, 0).applyQuaternion(q), // wrist 0
        new Vector3(0, 0, 0), // 1
        new Vector3(0.045, -0.02, -0.01).applyQuaternion(q), // thumb 2
        new Vector3(0, 0, 0), // 3
        new Vector3(0, 0, 0), // 4
        new Vector3(0.035, 0.005, 0).applyQuaternion(q), // index 5
        ...Array.from({ length: 11 }, () => new Vector3(0, 0, 0)),
        new Vector3(-0.032, 0, 0).applyQuaternion(q), // pinky 17
      ];
      const leftLandmarks = leftWorld.map((p) => ({
        x: p.x, y: p.y, z: p.z,
        worldX: p.x, worldY: p.y, worldZ: p.z,
      }));
      assert.equal(anatomicalChiralityFromWorld(leftLandmarks), 'Left', `Should identify Left hand at yaw=${yaw}, pitch=${pitch}`);
    }
  }
});

test('solveRingPose accurately places rings on all 5 fingers for both Left and Right hands', () => {
  const allFingers = [
    { finger: 'thumb', mcp: 2, pip: 3 },
    { finger: 'index', mcp: 5, pip: 6 },
    { finger: 'middle', mcp: 9, pip: 10 },
    { finger: 'ring', mcp: 13, pip: 14 },
    { finger: 'pinky', mcp: 17, pip: 18 },
  ];

  for (const isLeft of [false, true]) {
    for (const f of allFingers) {
      // Build hand skeleton with all 5 finger rays
      const q = new Quaternion();
      const localJoints = Array.from({ length: 21 }, () => [0, 0, 0]);
      localJoints[0] = [0, -0.055, 0];
      localJoints[2] = [-0.045, -0.02, -0.008];
      localJoints[3] = [-0.055, 0.015, -0.012];
      localJoints[5] = [-0.035, 0.005, 0];
      localJoints[6] = [-0.038, 0.048, 0];
      localJoints[9] = [-0.012, 0.013, 0];
      localJoints[10] = [-0.013, 0.055, 0];
      localJoints[13] = [0.01, 0.009, 0];
      localJoints[14] = [0.012, 0.05, 0];
      localJoints[17] = [0.032, 0, 0];
      localJoints[18] = [0.04, 0.034, 0];

      const sign = isLeft ? -1 : 1;
      const world = localJoints.map(p => new Vector3(p[0] * sign, p[1], p[2]).applyQuaternion(q));
      const screen = world.map(p => ({ x: 420 + p.x * 2000, y: 650 - p.y * 2000 }));
      const landmarks = world.map(p => ({
        x: (420 + p.x * 2000) / 900,
        y: (650 - p.y * 2000) / 1600,
        z: -p.z,
        worldX: p.x,
        worldY: -p.y,
        worldZ: -p.z,
      }));

      const pose = solveRingPose(
        landmarks,
        screen,
        isLeft ? 'Left' : 'Right',
        false,
        undefined,
        f,
      );

      assert.ok(pose, `Pose should solve for ${f.finger} on ${isLeft ? 'Left' : 'Right'} hand`);
      assert.ok(pose.fingerLength > 5, `Finger length should be non-zero for ${f.finger}`);
      assert.ok(pose.handScale > 20, `Hand scale should be valid for ${f.finger}`);

      // Normal Z: dorsal view should have outward facing normal
      const normalZ = new Vector3(0, 0, 1).applyQuaternion(pose.orientation).z;
      assert.ok(Math.abs(normalZ) > 0.05, `Surface normal should be defined for ${f.finger}: ${normalZ}`);
    }
  }
});

test('RingSurfaceConstraint resets cleanly when handedness changes to prevent false palm flip', () => {
  const constraint = new RingSurfaceConstraint();
  const rightSample = sample({ left: false, yaw: 0 });
  const leftSample = sample({ left: true, yaw: 0 });

  // 1. Initialize on Right hand
  const rightPose = constraint.update(
    rightSample.pose.orientation,
    rightSample.pose.anchors,
    rightSample.pose.fingerLength,
    40,
    'Right',
  );
  assert.ok(rightPose);

  // 2. Switch to Left hand with handedness='Left': should reset reference rather than treating area sign as palm turn
  const leftPose = constraint.update(
    leftSample.pose.orientation,
    leftSample.pose.anchors,
    leftSample.pose.fingerLength,
    40,
    'Left',
  );
  assert.ok(leftPose);

  // Dorsal head should remain on dorsal side, not flipped to rear shank
  const leftNormalZ = new Vector3(0, 0, 1).applyQuaternion(leftPose).z;
  const rightNormalZ = new Vector3(0, 0, 1).applyQuaternion(rightPose).z;
  assert.equal(Math.sign(leftNormalZ), Math.sign(rightNormalZ), 'Both dorsal poses should have consistent outward normal sign');
});


