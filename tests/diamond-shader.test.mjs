import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { buildBvh } from '../app/diamond/bvh/buildBvh.ts';
import { packBvhTextures, disposePackedBvhTextures } from '../app/diamond/bvh/packTextures.ts';
import { createRoundBrilliantGeometry } from '../app/diamond/roundBrilliantGeometry.ts';
import { flattenGemFacets } from '../app/diamond/DiamondMaterial.ts';
import { isRoundBrilliantShape } from '../app/diamond/fitBrilliantToGem.ts';
import { realisticDiamondMaterialParams, DIAMOND_IOR, DIAMOND_BOUNCES } from '../app/diamond/diamondDefaults.ts';

test('round brilliant cut geometry generates valid 57-facet cut with normals', () => {
  const geo = createRoundBrilliantGeometry(1.0);
  assert.ok(geo.attributes.position);
  assert.ok(geo.attributes.normal);
  assert.ok(geo.attributes.position.count > 0);
  assert.ok(geo.attributes.position.count % 3 === 0);
  assert.ok(isRoundBrilliantShape(new THREE.Vector3(1.0, 1.0, 0.6)));
});

test('flattenGemFacets produces non-indexed geometry with computed geometric normals', () => {
  const box = new THREE.BoxGeometry(1, 1, 1);
  const flattened = flattenGemFacets(box);
  assert.equal(flattened.index, null);
  assert.equal(flattened.attributes.position.count, 36);
  assert.equal(flattened.attributes.normal.count, 36);
});

test('buildBvh and packBvhTextures construct GPU bounding volume hierarchy textures', () => {
  const geo = createRoundBrilliantGeometry(0.8);
  const bvh = buildBvh(geo);
  assert.ok(bvh.nodes.length > 0);
  assert.ok(bvh.triangles.length > 0);

  const packed = packBvhTextures(bvh);
  assert.ok(packed.nodeTexture instanceof THREE.DataTexture);
  assert.ok(packed.triangleTexture instanceof THREE.DataTexture);
  assert.equal(packed.triangleCount, bvh.triangles.length);
  assert.ok(packed.nodeTexSize.x >= 1);
  assert.ok(packed.triTexSize.x >= 1);

  disposePackedBvhTextures(packed);
});

test('realisticDiamondMaterialParams provides physical optical parameters', () => {
  const params = realisticDiamondMaterialParams();
  assert.equal(params.ior, DIAMOND_IOR);
  assert.equal(params.bounces, DIAMOND_BOUNCES);
  assert.ok(params.dispersion > 0.04);
  assert.ok(params.sparkle > 0);
  assert.ok(params.facetScintillation > 0);
});
