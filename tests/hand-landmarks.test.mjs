import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_TRY_ON_FINGER,
  TRY_ON_FINGERS,
  getFingerDef,
  neighborsForFinger,
} from '../app/hand-landmarks.ts';

test('try-on fingers are index middle ring with MCP→PIP bones', () => {
  assert.deepEqual(TRY_ON_FINGERS.map((f) => f.name), ['index', 'middle', 'ring']);
  assert.equal(DEFAULT_TRY_ON_FINGER, 'ring');
  assert.deepEqual(getFingerDef('index'), { name: 'index', label: 'Index', mcp: 5, pip: 6, dip: 7, tip: 8 });
  assert.deepEqual(getFingerDef('middle'), { name: 'middle', label: 'Middle', mcp: 9, pip: 10, dip: 11, tip: 12 });
  assert.deepEqual(getFingerDef('ring'), { name: 'ring', label: 'Ring', mcp: 13, pip: 14, dip: 15, tip: 16 });
});

test('neighborsForFinger clips against adjacent proximal bones', () => {
  assert.deepEqual(neighborsForFinger('index'), [{ mcp: 9, pip: 10 }]);
  assert.deepEqual(neighborsForFinger('middle'), [
    { mcp: 5, pip: 6 },
    { mcp: 13, pip: 14 },
  ]);
  assert.deepEqual(neighborsForFinger('ring'), [
    { mcp: 9, pip: 10 },
    { mcp: 17, pip: 18 },
  ]);
});
