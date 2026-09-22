import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_TRY_ON_FINGER,
  TRY_ON_FINGERS,
  getFingerDef,
  neighborsForFinger,
} from '../app/hand-landmarks.ts';

test('try-on fingers are thumb index middle ring pinky with anatomical defs', () => {
  assert.deepEqual(TRY_ON_FINGERS.map((f) => f.name), ['thumb', 'index', 'middle', 'ring', 'pinky']);
  assert.equal(DEFAULT_TRY_ON_FINGER, 'ring');
  assert.deepEqual(getFingerDef('thumb'), { name: 'thumb', label: 'Thumb', mcp: 2, pip: 3, dip: 3, tip: 4, widthRatio: 1.18, defaultSeat: 0.46 });
  assert.deepEqual(getFingerDef('index'), { name: 'index', label: 'Index', mcp: 5, pip: 6, dip: 7, tip: 8, widthRatio: 1.02, defaultSeat: 0.42 });
  assert.deepEqual(getFingerDef('middle'), { name: 'middle', label: 'Middle', mcp: 9, pip: 10, dip: 11, tip: 12, widthRatio: 1.07, defaultSeat: 0.42 });
  assert.deepEqual(getFingerDef('ring'), { name: 'ring', label: 'Ring', mcp: 13, pip: 14, dip: 15, tip: 16, widthRatio: 1.0, defaultSeat: 0.60 });
  assert.deepEqual(getFingerDef('pinky'), { name: 'pinky', label: 'Pinky', mcp: 17, pip: 18, dip: 19, tip: 20, widthRatio: 0.8, defaultSeat: 0.38 });
});

test('neighborsForFinger clips against adjacent proximal bones', () => {
  assert.deepEqual(neighborsForFinger('thumb'), [{ mcp: 5, pip: 6 }]);
  assert.deepEqual(neighborsForFinger('index'), [{ mcp: 9, pip: 10 }]);
  assert.deepEqual(neighborsForFinger('middle'), [
    { mcp: 5, pip: 6 },
    { mcp: 13, pip: 14 },
  ]);
  assert.deepEqual(neighborsForFinger('ring'), [
    { mcp: 9, pip: 10 },
    { mcp: 17, pip: 18 },
  ]);
  assert.deepEqual(neighborsForFinger('pinky'), [{ mcp: 13, pip: 14 }]);
});
