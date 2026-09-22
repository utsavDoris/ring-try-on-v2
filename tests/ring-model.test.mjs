import test from 'node:test';
import assert from 'node:assert/strict';
import { BoxGeometry, Group, Mesh, MeshBasicMaterial, Vector3 } from 'three';
import {
  DEFAULT_INNER_TO_OUTER_DIAMETER,
  MAX_WIDTH_BONE_RATIO,
  MIN_WIDTH_BONE_RATIO,
  OCCLUSION_RADIUS_RATIO,
  WEAR_CLEARANCE,
  applyStoneAxisBasis,
  bakeMeshScale,
  centerRingOnMetalHole,
  createFingerOccluderGeometry,
  geometrySignature,
  measureMetalBounds,
  measureRingInnerDiameter,
  normalizeRingToUnitHole,
  readRingModelExtras,
  resolveFingerWidthPx,
  ringWorldScale,
} from '../app/ring-model.ts';

test('bakeMeshScale folds signed scale into geometry and restores unit scale', () => {
  const mesh = new Mesh(new BoxGeometry(1, 1, 1));
  mesh.scale.set(-1.2, 0.8, 1.1);
  bakeMeshScale(mesh);
  assert.deepEqual([mesh.scale.x, mesh.scale.y, mesh.scale.z], [1, 1, 1]);
  const position = mesh.geometry.getAttribute('position');
  let maxAbsX = 0;
  for (let i = 0; i < position.count; i++) {
    maxAbsX = Math.max(maxAbsX, Math.abs(position.getX(i)));
  }
  assert.ok(Math.abs(maxAbsX - 0.6) < 1e-5);
  mesh.geometry.dispose();
  mesh.material.dispose();
});

test('geometrySignature is stable for identical buffers and differs otherwise', () => {
  const a = new BoxGeometry(1, 1, 1);
  const b = new BoxGeometry(1, 1, 1);
  const c = new BoxGeometry(2, 1, 1);
  assert.equal(geometrySignature(a), geometrySignature(b));
  assert.notEqual(geometrySignature(a), geometrySignature(c));
  a.dispose();
  b.dispose();
  c.dispose();
});

test('createFingerOccluderGeometry is a closed tapered cylinder', () => {
  const geometry = createFingerOccluderGeometry();
  assert.ok(geometry.getAttribute('position'));
  assert.ok(geometry.getIndex());
  geometry.dispose();
});

test('measureRingInnerDiameter falls back when no metal meshes exist', () => {
  const root = new Group();
  const mesh = new Mesh(new BoxGeometry(2.4, 1, 1), new MeshBasicMaterial({ name: 'Plastic' }));
  root.add(mesh);
  root.updateMatrixWorld(true);
  const measured = measureRingInnerDiameter(root, DEFAULT_INNER_TO_OUTER_DIAMETER);
  assert.ok(Math.abs(measured - 2.4 * DEFAULT_INNER_TO_OUTER_DIAMETER) < 1e-6);
  mesh.geometry.dispose();
  mesh.material.dispose();
});

test('readRingModelExtras reads nested extras and top-level userData', () => {
  const root = new Group();
  root.userData = { innerDiameter: 1.5 };
  assert.equal(readRingModelExtras(root).innerDiameter, 1.5);
  root.userData = { extras: { stoneAxis: '-Z' } };
  assert.equal(readRingModelExtras(root).stoneAxis, '-Z');
});

test('normalizeRingToUnitHole scales so inner diameter is 1', () => {
  const root = new Group();
  root.add(new Mesh(new BoxGeometry(1, 1, 1), new MeshBasicMaterial({ name: 'Metal 01' })));
  const factor = normalizeRingToUnitHole(root, 1.91);
  assert.ok(Math.abs(factor - 1 / 1.91) < 1e-9);
  assert.equal(root.userData.innerDiameter, 1);
  assert.equal(root.userData.rawInnerDiameter, 1.91);
  assert.ok(Math.abs(root.scale.x - 1 / 1.91) < 1e-9);
});

test('normalizeRingToUnitHole bakes fitScale into the unit-hole factor', () => {
  const root = new Group();
  root.add(new Mesh(new BoxGeometry(1, 1, 1), new MeshBasicMaterial({ name: 'Metal 01' })));
  const factor = normalizeRingToUnitHole(root, 2, 1.1);
  assert.ok(Math.abs(factor - (1 / 2) * 1.1) < 1e-9);
  assert.equal(root.userData.fitScale, 1.1);
  assert.ok(Math.abs(root.scale.x - 0.55) < 1e-9);
});

test('ringWorldScale applies wear clearance to finger diameter', () => {
  assert.equal(WEAR_CLEARANCE, 1.02);
  assert.ok(Math.abs(ringWorldScale(1) - 1.02) < 1e-9);
  assert.ok(Math.abs(ringWorldScale(2) - 2.04) < 1e-9);
  assert.ok(ringWorldScale(0) > 0);
});

test('resolveFingerWidthPx clamps absurd edges to bone anatomy', () => {
  const bone = 100;
  assert.ok(Math.abs(resolveFingerWidthPx(30, bone) - 30) < 1e-9);
  // Above min px but below bone floor → clamp up.
  assert.ok(Math.abs(resolveFingerWidthPx(10, bone) - bone * MIN_WIDTH_BONE_RATIO) < 1e-9);
  assert.ok(Math.abs(resolveFingerWidthPx(90, bone) - bone * MAX_WIDTH_BONE_RATIO) < 1e-9);
  assert.equal(MAX_WIDTH_BONE_RATIO, 0.52);
  assert.ok(Math.abs(resolveFingerWidthPx(45, bone) - 45) < 1e-9);
  assert.equal(OCCLUSION_RADIUS_RATIO, 0.5);
  assert.ok(resolveFingerWidthPx(3, 0) >= 8);
});

test('centerRingOnMetalHole puts metal AABB center at origin in XYZ', () => {
  const root = new Group();
  const scene = new Group();
  root.add(scene);
  const metal = new Mesh(
    new BoxGeometry(2, 0.4, 2),
    new MeshBasicMaterial({ name: 'Metal Band' }),
  );
  metal.position.set(3, -1.5, 2);
  const gem = new Mesh(
    new BoxGeometry(0.5, 0.5, 0.5),
    new MeshBasicMaterial({ name: 'Diamond Gem' }),
  );
  gem.position.set(5, 2, 4);
  scene.add(metal);
  scene.add(gem);
  root.updateMatrixWorld(true);

  centerRingOnMetalHole(root, scene);
  const bounds = measureMetalBounds(root);
  assert.ok(bounds);
  assert.ok(bounds.center.length() < 1e-5, `metal center ${bounds.center.toArray()}`);

  normalizeRingToUnitHole(root, 1.5);
  centerRingOnMetalHole(root, root, { x: 0, y: 0, z: 0 });
  const after = measureMetalBounds(root);
  assert.ok(after);
  assert.ok(after.center.length() < 1e-4, `post-normalize metal center ${after.center.toArray()}`);
  // Gem must not pull the metal origin off-axis in XZ.
  assert.ok(Math.hypot(after.center.x, after.center.z) < 1e-4);

  metal.geometry.dispose();
  metal.material.dispose();
  gem.geometry.dispose();
  gem.material.dispose();
});

test('applyStoneAxisBasis maps +Z extras onto runtime -Z head direction', () => {
  const root = new Group();
  const marker = new Mesh(new BoxGeometry(0.2, 0.2, 0.2), new MeshBasicMaterial({ name: 'Diamond Gem' }));
  marker.position.set(0, 0, 1); // authored +Z head
  root.add(marker);
  root.updateMatrixWorld(true);

  applyStoneAxisBasis(root, '+Z');
  root.updateMatrixWorld(true);
  const world = new Vector3();
  marker.getWorldPosition(world);
  assert.ok(world.z < -0.5, `expected head on -Z, got z=${world.z}`);
  assert.equal(root.userData.stoneAxis, '-Z');
  assert.equal(root.userData.stoneAxisSource, '+Z');

  const identity = new Group();
  applyStoneAxisBasis(identity, '-Z');
  assert.ok(Math.abs(identity.quaternion.w - 1) < 1e-9);

  marker.geometry.dispose();
  marker.material.dispose();
});
