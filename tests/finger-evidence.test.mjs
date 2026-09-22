import test from 'node:test';
import assert from 'node:assert/strict';
import { findFingerSilhouette, MIN_FINGER_EDGE_CONFIDENCE, sectionMedianWidth, softSeatCrossPosition } from '../app/finger-silhouette.ts';
import { FingerFitFilter } from '../app/ring-tracking.ts';
import {
  computeFingerEdgeRoi,
  mapRoiPointToSource,
  mapSourcePointToRoi,
  measureFingerSilhouetteOnRoi,
  EdgeMeasurementSmoother,
} from '../app/finger-edge-opencv.ts';

function finger({ center = 90, background = 30, skin = 160, highlight = false } = {}) {
  const width = 180, height = 320, data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const inside = x >= center - 32 && x < center + 32 && y >= 45 && y <= 280;
    const value = inside ? highlight && Math.abs(x - center) < 25 ? 220 : skin : background;
    data.set([value, value, value, 255], (y * width + x) * 4);
  }
  return { image: { width, height, data }, base: { x: center, y: 250 }, tip: { x: center, y: 70 } };
}
function detect(options) {
  const { image, base, tip } = finger(options);
  return findFingerSilhouette(image, base, tip);
}
const evidence = (revision, changes = {}) => ({ width: 40, offset: 0, revision, confidence: 1, ...changes });
function calibrate(filter) {
  for (const revision of [0, 40, 80]) filter.update(100, 1, evidence(revision));
}

test('clear outer boundaries give a usable measurement with raw cross-sections', () => {
  const edge = detect();
  assert.ok(edge && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE);
  assert.equal(edge.reason, 'clear');
  assert.equal(edge.support, 6);
  assert.equal(edge.sections.length, 6);
  assert.ok(Math.abs(edge.width - 64) < 2);
});

test('a consistent central highlight is shown but cannot calibrate the ring', () => {
  const edge = detect({ background: 90, skin: 110, highlight: true });
  assert.ok(edge, 'retain raw candidates for diagnosis');
  assert.equal(edge.support, 6, 'consistency alone is not confidence');
  assert.ok(edge.confidence < MIN_FINGER_EDGE_CONFIDENCE);
  const filter = new FingerFitFilter();
  for (let revision = 0; revision < 400; revision += 40) {
    assert.equal(filter.update(100, 1, evidence(revision, { width: edge.width, confidence: edge.confidence })), null);
  }
});

test('weak boundaries remain visible as uncertain evidence', () => {
  const edge = detect({ background: 90, skin: 110 });
  assert.ok(edge);
  assert.ok(edge.confidence < MIN_FINGER_EDGE_CONFIDENCE);
  assert.equal(edge.reason, 'weak contrast');
});

test('a clipped finger or touching neighbor cannot supply a clear boundary pair', () => {
  const clipped = detect({ center: 20 });
  assert.ok(!clipped || clipped.confidence < MIN_FINGER_EDGE_CONFIDENCE);
  const { image, base, tip } = finger();
  const crowded = findFingerSilhouette(image, base, tip, [
    { base: { x: 125, y: 250 }, tip: { x: 125, y: 70 } },
  ]);
  assert.ok(!crowded || crowded.confidence < MIN_FINGER_EDGE_CONFIDENCE);
});

test('initial fit requires three confirming clear observations', () => {
  const filter = new FingerFitFilter();
  assert.equal(filter.update(100, 1, evidence(0)), null);
  assert.equal(filter.confirmationFrames, 1);
  assert.equal(filter.update(100, 1, evidence(40)), null);
  assert.equal(filter.confirmationFrames, 2);
  assert.equal(filter.update(100, 1, evidence(80)).width, 40);
  assert.equal(filter.status, 'accepted');
});

test('uncertain evidence cannot change width or offset, but distance still tracks', () => {
  const filter = new FingerFitFilter(); calibrate(filter);
  for (let revision = 120; revision < 500; revision += 40) {
    const result = filter.update(150, 1, evidence(revision, { width: 95, offset: 12, confidence: 0.4 }));
    assert.equal(result.width, 60);
    assert.equal(result.offset, 0);
    assert.equal(filter.status, 'uncertain');
  }
});

test('uncertain observations and long gaps restart confirmation', () => {
  const filter = new FingerFitFilter();
  filter.update(100, 1, evidence(0));
  filter.update(100, 1, evidence(80, { confidence: 0.2 }));
  assert.equal(filter.update(100, 1, evidence(120)), null);
  assert.equal(filter.confirmationFrames, 1);
  assert.equal(filter.update(100, 1, evidence(400)), null);
  assert.equal(filter.confirmationFrames, 1);
});

test('diameter stays locked through brief mismatched edges', () => {
  const filter = new FingerFitFilter(); calibrate(filter);
  assert.equal(filter.update(100, 1, evidence(120, { width: 56 })).width, 40);
  assert.equal(filter.update(100, 1, evidence(160, { width: 56 })).width, 40);
  assert.equal(filter.calibrated, true);
  assert.equal(filter.update(100, 1, evidence(200, { width: 40 })).width, 40);
});

test('diameter stays locked until an explicit refit', () => {
  const filter = new FingerFitFilter(); calibrate(filter);
  for (let revision = 120; revision < 1000; revision += 40) {
    assert.equal(filter.update(100, 1, evidence(revision, { width: 56 })).width, 40);
  }
  filter.reset();
  assert.equal(filter.calibrated, false);
  let result;
  for (const revision of [1040, 1080, 1120]) result = filter.update(100, 1, evidence(revision, { width: 56 }));
  assert.ok(Math.abs(result.width - 56) < 1e-6);
});

test('one or two bad edge frames cannot unlock a stable fit', () => {
  const filter = new FingerFitFilter(); calibrate(filter);
  assert.equal(filter.update(100, 1, evidence(120, { width: 56 })).width, 40);
  assert.equal(filter.update(100, 1, evidence(160, { width: 40 })).width, 40);
  assert.equal(filter.calibrated, true);
});

test('side views cannot initialize or recalibrate width', () => {
  const filter = new FingerFitFilter();
  for (const revision of [0, 40, 80]) assert.equal(filter.update(100, 0.3, evidence(revision)), null);
  filter.reset(); calibrate(filter);
  for (const revision of [120, 160, 200]) assert.equal(filter.update(100, 0.3, evidence(revision, { width: 70 })).width, 40);
});

test('confirmation uses relative width while the hand approaches the camera', () => {
  const filter = new FingerFitFilter();
  assert.equal(filter.update(100, 1, evidence(0)), null);
  assert.equal(filter.update(125, 1, evidence(40, { width: 50 })), null);
  assert.equal(filter.update(150, 1, evidence(80, { width: 60 })).width, 60);
});

test('init facing of 0.75 is enough to start a size lock with anchors', () => {
  const filter = new FingerFitFilter();
  const anchors = [{ x: 0, y: 0 }, { x: 40, y: 0 }, { x: 20, y: 10 }, { x: 10, y: 8 }, { x: 0, y: 5 }];
  assert.equal(filter.update(100, 0.75, evidence(0), anchors), null);
  assert.equal(filter.update(100, 0.75, evidence(40), anchors), null);
  const locked = filter.update(100, 0.75, evidence(80), anchors);
  assert.ok(locked && locked.width > 0);
  assert.equal(filter.calibrated, true);
});

test('a background edge on one side lowers confidence without blocking a stable fit', () => {
  const { image, base, tip } = finger();
  for (let y = 0; y < image.height; y++) for (let x = 138; x < image.width; x++) {
    const p = (y * image.width + x) * 4;
    image.data.set([90, 90, 90, 255], p);
  }
  const edge = findFingerSilhouette(image, base, tip);
  assert.ok(edge && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE);
  assert.ok(edge.confidence < 1, 'retain a penalty for the extra background edge');
  assert.ok(Math.abs(edge.width - 64) < 2, 'measure the finger, not the further background edge');
  const filter = new FingerFitFilter();
  for (const revision of [0, 40, 80]) filter.update(100, 1, evidence(revision, { width: edge.width, confidence: edge.confidence }));
  assert.equal(filter.calibrated, true);
});

test('one cross-section clipped by a converging neighbor does not veto five clear sections', () => {
  const { image, base, tip } = finger();
  const edge = findFingerSilhouette(image, base, tip, [
    { base: { x: 129, y: 250 }, tip: { x: 175, y: 70 } },
  ]);
  assert.ok(edge && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE);
  assert.equal(edge.support, 5);
  assert.ok(Math.abs(edge.width - 64) < 2);
});

test('a nearby but separated neighbor leaves enough room to measure the boundary', () => {
  const { image, base, tip } = finger();
  const edge = findFingerSilhouette(image, base, tip, [
    { base: { x: 149, y: 250 }, tip: { x: 149, y: 70 } },
  ]);
  assert.ok(edge && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE);
  assert.ok(Math.abs(edge.width - 64) < 2);
});

test('distant narrow finger at long range produces a valid silhouette', () => {
  const width = 120, height = 160;
  const data = new Uint8ClampedArray(width * height * 4);
  const center = 60;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const inside = x >= center - 8 && x <= center + 8 && y >= 20 && y <= 140;
    const value = inside ? 160 : 30;
    data.set([value, value, value, 255], (y * width + x) * 4);
  }
  // Proximal bone long enough that width/length stays under the dual-finger reject.
  const edge = findFingerSilhouette(
    { width, height, data },
    { x: center, y: 110 },
    { x: center, y: 50 },
  );
  assert.ok(edge);
  assert.ok(Math.abs(edge.width - 16) <= 2, `expected width ~16, got ${edge.width}`);
  assert.ok(edge.support >= 4);
});

test('sectionMedianWidth matches silhouette width for clear outer boundaries', () => {
  const edge = detect();
  assert.ok(edge);
  const medianWidth = sectionMedianWidth(edge);
  assert.ok(Math.abs(medianWidth - edge.width) < 2);
  assert.ok(Math.abs(medianWidth - 64) < 2);
});

test('softSeatCrossPosition nudges palm-ward when base sections are thicker', () => {
  const edge = detect();
  assert.ok(edge);
  // Artificial taper: palm-ward sections wider than tip-ward.
  const tapered = {
    ...edge,
    sections: edge.sections.map((section, index) => {
      const half = edge.sections.length / 2;
      const scale = index < half ? 1.12 : 0.88;
      const midX = (section.left.x + section.right.x) / 2;
      const midY = (section.left.y + section.right.y) / 2;
      const hx = (section.right.x - section.left.x) * 0.5 * scale;
      const hy = (section.right.y - section.left.y) * 0.5 * scale;
      return {
        left: { x: midX - hx, y: midY - hy },
        right: { x: midX + hx, y: midY + hy },
      };
    }),
  };
  const seat = softSeatCrossPosition(tapered, 0.38, 0.32, 0.44);
  assert.ok(seat <= 0.38);
  assert.ok(seat >= 0.32);
  assert.equal(softSeatCrossPosition(null, 0.38), 0.38);
  // Placement and silhouette share the same soft-seat station.
  const { image, base, tip } = finger();
  const remMeasure = findFingerSilhouette(image, base, tip, [], seat);
  assert.ok(remMeasure);
  const seatAgain = softSeatCrossPosition(remMeasure, 0.38, 0.32, 0.44);
  assert.ok(Math.abs(seatAgain - seat) < 0.04);
});

test('FingerFitFilter keeps locked width when confirmed edges drift within 8%', () => {
  const filter = new FingerFitFilter();
  calibrate(filter);
  let result = filter.update(100, 1, evidence(120, { width: 40 }));
  assert.equal(result.width, 40);
  // 5% wider — formerly soft-tracked; size lock now freezes ratio.
  for (let revision = 121; revision < 200; revision += 1) {
    result = filter.update(100, 1, evidence(revision, { width: 42 }));
  }
  assert.equal(result.width, 40);
  assert.equal(filter.calibrated, true);
});

test('clear outer edges lock edge-to-edge width matching the silhouette span', () => {
  const edge = detect();
  assert.ok(edge && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE);
  const span = sectionMedianWidth(edge);
  assert.ok(Math.abs(span - 64) < 2);
  const filter = new FingerFitFilter();
  let locked;
  for (const revision of [0, 40, 80]) {
    locked = filter.update(100, 1, evidence(revision, {
      width: span,
      confidence: edge.confidence,
    }));
  }
  assert.ok(locked);
  assert.ok(Math.abs(locked.width - span) < 1e-6);
  assert.equal(filter.calibrated, true);
});

test('front-facing palm and dorsal both confirm the same edge-to-edge lock path', () => {
  // facing is |normal.z| — palm and dorsal fronts share the fit gate.
  for (const facing of [0.95, 1.0]) {
    const filter = new FingerFitFilter();
    let locked;
    for (const revision of [0, 40, 80]) {
      locked = filter.update(100, facing, evidence(revision, { width: 48 }));
    }
    assert.ok(locked && Math.abs(locked.width - 48) < 1e-6, `facing=${facing}`);
    assert.equal(filter.calibrated, true);
  }
});

test('far-hand silhouette locks edge-to-edge width for a distant narrow finger', () => {
  const width = 120, height = 160;
  const data = new Uint8ClampedArray(width * height * 4);
  const center = 60;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const inside = x >= center - 8 && x <= center + 8 && y >= 20 && y <= 140;
    const value = inside ? 160 : 30;
    data.set([value, value, value, 255], (y * width + x) * 4);
  }
  const edge = findFingerSilhouette(
    { width, height, data },
    { x: center, y: 110 },
    { x: center, y: 50 },
  );
  assert.ok(edge && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE);
  const span = sectionMedianWidth(edge);
  assert.ok(Math.abs(span - 16) <= 2, `far span ${span}`);
  const filter = new FingerFitFilter();
  let locked;
  for (const revision of [0, 40, 80]) {
    locked = filter.update(50, 1, evidence(revision, { width: span, confidence: edge.confidence }));
  }
  assert.ok(locked && Math.abs(locked.width - span) < 1e-6);
  assert.equal(filter.calibrated, true);
});

test('near-hand silhouette locks a large finger without collapsing to a highlight', () => {
  const width = 320, height = 480;
  const data = new Uint8ClampedArray(width * height * 4);
  const center = 160;
  const half = 56;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const inside = x >= center - half && x < center + half && y >= 60 && y <= 420;
    const value = inside ? 170 : 25;
    data.set([value, value, value, 255], (y * width + x) * 4);
  }
  const edge = findFingerSilhouette(
    { width, height, data },
    { x: center, y: 380 },
    { x: center, y: 100 },
  );
  assert.ok(edge && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE);
  const span = sectionMedianWidth(edge);
  assert.ok(Math.abs(span - half * 2) < 4, `near span ${span}`);
  const filter = new FingerFitFilter();
  let locked;
  for (const revision of [0, 40, 80]) {
    locked = filter.update(280, 1, evidence(revision, { width: span, confidence: edge.confidence }));
  }
  assert.ok(locked && Math.abs(locked.width - span) < 1e-6);
});

test('post-lock width stays frozen when edges drift within 20%', () => {
  const filter = new FingerFitFilter();
  calibrate(filter);
  let result = filter.update(100, 1, evidence(120, { width: 40 }));
  assert.equal(result.width, 40);
  // 15% wider — still trustworthy for offset/drift, but must not change size.
  for (let revision = 121; revision < 280; revision += 1) {
    result = filter.update(100, 1, evidence(revision, { width: 46 }));
  }
  assert.equal(result.width, 40);
  assert.equal(filter.calibrated, true);
});

test('finger edge ROI remaps landmarks and keeps silhouette width lockable', () => {
  const { image, base, tip } = finger();
  const roi = computeFingerEdgeRoi(base, tip, [], image.width, image.height, 320);
  assert.ok(roi);
  assert.ok(roi.dw <= 320 && roi.dh <= 320);
  // Crop ROI pixels into a packed buffer (same as edge canvas draw).
  const data = new Uint8ClampedArray(roi.dw * roi.dh * 4);
  for (let y = 0; y < roi.dh; y++) {
    for (let x = 0; x < roi.dw; x++) {
      const sx = Math.min(image.width - 1, Math.max(0, Math.floor(x / roi.scale + roi.sx)));
      const sy = Math.min(image.height - 1, Math.max(0, Math.floor(y / roi.scale + roi.sy)));
      const src = (sy * image.width + sx) * 4;
      const dst = (y * roi.dw + x) * 4;
      data[dst] = image.data[src];
      data[dst + 1] = image.data[src + 1];
      data[dst + 2] = image.data[src + 2];
      data[dst + 3] = 255;
    }
  }
  const edge = measureFingerSilhouetteOnRoi(
    { width: roi.dw, height: roi.dh, data },
    mapSourcePointToRoi(base, roi),
    mapSourcePointToRoi(tip, roi),
    [],
    0.53,
    null,
  );
  assert.ok(edge && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE);
  const sourceLeft = mapRoiPointToSource(edge.left, roi);
  const sourceRight = mapRoiPointToSource(edge.right, roi);
  const span = Math.hypot(sourceRight.x - sourceLeft.x, sourceRight.y - sourceLeft.y);
  assert.ok(Math.abs(span - 64) < 3, `roi span ${span}`);
  const filter = new FingerFitFilter();
  let locked;
  for (const revision of [0, 40, 80]) {
    locked = filter.update(100, 1, evidence(revision, { width: span, confidence: edge.confidence }));
  }
  assert.ok(locked && Math.abs(locked.width - span) < 1e-6);
});

test('EdgeMeasurementSmoother medians width and holds last clear sample briefly', () => {
  const smoother = new EdgeMeasurementSmoother();
  const a = smoother.smooth({ width: 40, offset: 0, revision: 10, confidence: 1 }, 10);
  const b = smoother.smooth({ width: 42, offset: 1, revision: 20, confidence: 1 }, 20);
  const c = smoother.smooth({ width: 44, offset: 2, revision: 30, confidence: 1 }, 30);
  assert.ok(a && b && c);
  assert.equal(c.width, 42);
  assert.equal(c.offset, 1);
  const held = smoother.smooth({ width: 99, offset: 9, revision: 40, confidence: 0.2 }, 40);
  assert.ok(held);
  assert.equal(held.width, 42);
  assert.equal(held.revision, 30);
  const dropped = smoother.smooth(undefined, 200);
  assert.equal(dropped, undefined);
});

test('changing finger drops the locked size and relocks the new finger', () => {
  const filter = new FingerFitFilter();
  const target = (finger, hand = 'Right') => ({ finger, hand });
  for (const revision of [0, 40, 80]) {
    filter.update(100, 1, evidence(revision), undefined, target('ring'));
  }
  assert.equal(filter.calibrated, true);
  const preview = filter.update(100, 1, evidence(120, { width: 28 }), undefined, target('pinky'));
  assert.equal(filter.calibrated, false);
  assert.equal(filter.relocking, true);
  assert.ok(preview && Math.abs(preview.width - 28) < 1e-6);
  // 400ms later and 3% wider — initial lock would reject this gap; relock accepts it.
  const locked = filter.update(100, 1, evidence(520, { width: 29 }), undefined, target('pinky'));
  assert.equal(filter.calibrated, true);
  assert.equal(filter.relocking, false);
  assert.ok(Math.abs(locked.width - 29) < 1e-6);
  // New lock stays frozen; it does not grow back to the previous finger.
  const held = filter.update(100, 1, evidence(560, { width: 40 }), undefined, target('pinky'));
  assert.ok(Math.abs(held.width - 29) < 1e-6);
});

test('one mismatched hand frame keeps the lock; the next hand relocks', () => {
  const filter = new FingerFitFilter();
  const target = (hand) => ({ finger: 'ring', hand });
  for (const revision of [0, 40, 80]) {
    filter.update(100, 1, evidence(revision), undefined, target('Right'));
  }
  const blip = filter.update(100, 1, evidence(120, { width: 52 }), undefined, target('Left'));
  assert.equal(filter.calibrated, true);
  assert.equal(blip.width, 40);
  const back = filter.update(100, 1, evidence(160), undefined, target('Right'));
  assert.equal(filter.calibrated, true);
  assert.equal(back.width, 40);

  filter.update(100, 1, evidence(200, { width: 52 }), undefined, target('Left'));
  const preview = filter.update(100, 1, evidence(240, { width: 52 }), undefined, target('Left'));
  assert.equal(filter.calibrated, false);
  assert.ok(preview && Math.abs(preview.width - 52) < 1e-6);
  const locked = filter.update(100, 1, evidence(280, { width: 52 }), undefined, target('Left'));
  assert.equal(filter.calibrated, true);
  assert.ok(Math.abs(locked.width - 52) < 1e-6);
});

test('explicit finger relock keeps measuring when edge frames are slower than 180ms', () => {
  const filter = new FingerFitFilter();
  calibrate(filter);
  filter.relock('finger');
  assert.equal(filter.calibrated, false);
  const preview = filter.update(100, 0.7, evidence(1000, { width: 32 }));
  assert.equal(filter.calibrated, false);
  assert.ok(preview && Math.abs(preview.width - 32) < 1e-6);
  const locked = filter.update(100, 0.7, evidence(1400, { width: 32 }));
  assert.equal(filter.calibrated, true);
  assert.ok(Math.abs(locked.width - 32) < 1e-6);
});

test('post-lock FingerFitFilter ignores tiny offset noise within deadband', () => {
  const filter = new FingerFitFilter();
  calibrate(filter);
  const locked = filter.update(100, 1, evidence(120, { width: 40, offset: 0 }));
  assert.ok(locked);
  assert.ok(Math.abs(locked.offset) < 1e-6);
  const quiet = filter.update(100, 1, evidence(160, { width: 40, offset: 1.0 }));
  assert.ok(quiet);
  // offsetNorm = 1/100 = 0.01 < 0.018 deadband → no chase
  assert.ok(Math.abs(quiet.offset) < 1e-6, `expected held offset, got ${quiet.offset}`);
});

test('switching back to a locked hand restores its size without remeasuring', () => {
  const filter = new FingerFitFilter();
  const target = (hand) => ({ finger: 'ring', hand });
  for (const revision of [0, 40]) {
    filter.update(100, 1, evidence(revision, { width: 40 }), undefined, target('Right'));
  }
  assert.equal(filter.calibrated, true);
  assert.equal(filter.lockedHand, 'Right');
  // First opposite frame is a label blip and keeps the right-hand lock.
  const blip = filter.update(100, 1, evidence(80, { width: 52 }), undefined, target('Left'));
  assert.equal(filter.calibrated, true);
  assert.equal(filter.lockedHand, 'Right');
  assert.ok(Math.abs(blip.width - 40) < 1e-6);
  filter.update(100, 1, evidence(120, { width: 52 }), undefined, target('Left'));
  const left = filter.update(100, 1, evidence(200, { width: 52 }), undefined, target('Left'));
  assert.equal(filter.calibrated, true);
  assert.equal(filter.lockedHand, 'Left');
  assert.ok(Math.abs(left.width - 52) < 1e-6);
  // One frame back stays on the left lock; the next frame restores 40px.
  const flicker = filter.update(100, 1, evidence(240, { width: 40 }), undefined, target('Right'));
  assert.equal(filter.lockedHand, 'Left');
  assert.ok(Math.abs(flicker.width - 52) < 1e-6);
  const restored = filter.update(100, 1, evidence(280, { width: 99 }), undefined, target('Right'));
  assert.equal(filter.calibrated, true);
  assert.equal(filter.lockedHand, 'Right');
  assert.ok(Math.abs(restored.width - 40) < 1e-6);
});

test('switching back to a locked finger restores its size', () => {
  const filter = new FingerFitFilter();
  const target = (finger) => ({ finger, hand: 'Left' });
  for (const revision of [0, 40]) {
    filter.update(100, 1, evidence(revision, { width: 40 }), undefined, target('ring'));
  }
  filter.update(100, 1, evidence(80, { width: 28 }), undefined, target('pinky'));
  const pinky = filter.update(100, 1, evidence(160, { width: 28 }), undefined, target('pinky'));
  assert.equal(filter.calibrated, true);
  assert.ok(Math.abs(pinky.width - 28) < 1e-6);
  const restored = filter.update(100, 1, evidence(200, { width: 90 }), undefined, target('ring'));
  assert.equal(filter.calibrated, true);
  assert.equal(filter.lockedFinger, 'ring');
  assert.ok(Math.abs(restored.width - 40) < 1e-6);
});

test('refit clears left and right size locks', () => {
  const filter = new FingerFitFilter();
  const target = (hand) => ({ finger: 'ring', hand });
  for (const revision of [0, 40]) {
    filter.update(100, 1, evidence(revision, { width: 40 }), undefined, target('Right'));
  }
  filter.update(100, 1, evidence(80, { width: 52 }), undefined, target('Left'));
  filter.update(100, 1, evidence(160, { width: 52 }), undefined, target('Left'));
  const left = filter.update(100, 1, evidence(200, { width: 52 }), undefined, target('Left'));
  assert.equal(filter.calibrated, true);
  assert.ok(left && Math.abs(left.width - 52) < 1e-6);
  filter.reset();
  assert.equal(filter.calibrated, false);
  assert.equal(filter.lockedHand, null);
  const preview = filter.update(100, 1, evidence(200, { width: 40 }), undefined, target('Right'));
  assert.equal(filter.calibrated, false);
  assert.ok(preview && Math.abs(preview.width - 40) < 1e-6);
  const relocked = filter.update(100, 1, evidence(280, { width: 40 }), undefined, target('Right'));
  assert.equal(filter.calibrated, true);
  assert.ok(Math.abs(relocked.width - 40) < 1e-6);
});

