import test from 'node:test';
import assert from 'node:assert/strict';
import { Texture, DataTexture, DataUtils, HalfFloatType, RGBAFormat, MeshPhysicalMaterial, NearestFilter } from 'three';
import { ringPixelRatio, limitEnvironmentSize, loadOptionalEnvironment, createFallbackDiamondMaterial } from '../app/ring-runtime.ts';

test('large portrait and landscape iPads stay within the framebuffer budget', () => {
  for (const [width, height] of [[1024, 1366], [1366, 1024], [768, 1024], [430, 932]]) {
    const ratio = ringPixelRatio(width, height, 2, true);
    assert.ok(width * height * ratio * ratio <= 1_200_001);
    assert.ok(ratio <= 1.5);
  }
  assert.equal(ringPixelRatio(400, 600, 1, false), 1);
});

test('an unavailable EXR cannot reject ring loading', async () => {
  const texture = await loadOptionalEnvironment(async () => { throw new Error('simulated EXR failure'); });
  assert.ok(texture instanceof Texture);
  assert.equal(texture.minFilter, NearestFilter);
  assert.ok(texture.image.data.byteLength < 4096);
  texture.dispose();
});

test('successful lighting retains the original texture', async () => {
  const original = new Texture();
  assert.equal(await loadOptionalEnvironment(async () => original), original);
  original.dispose();
});

test('mobile lighting downsamples in linear HDR without clipping studio highlights', () => {
  const data = new Uint16Array(4 * 2 * 4);
  for (let i = 0; i < data.length; i++) data[i] = DataUtils.toHalfFloat(i % 4 === 3 ? 1 : i < 16 ? 2 : 6);
  const texture = new DataTexture(data, 4, 2, RGBAFormat, HalfFloatType);
  assert.equal(limitEnvironmentSize(texture, 2), texture);
  assert.equal(texture.image.width, 2);
  assert.equal(texture.image.height, 1);
  assert.equal(DataUtils.fromHalfFloat(texture.image.data[0]), 4);
  assert.equal(DataUtils.fromHalfFloat(texture.image.data[3]), 1);
  texture.dispose();
});

test('emergency diamonds require no ray-tracing or transmission target', () => {
  const material = createFallbackDiamondMaterial();
  assert.ok(material instanceof MeshPhysicalMaterial);
  assert.equal(material.transmission, 0);
  assert.equal(material.depthWrite, true);
  material.dispose();
});

test('updateDiamondEnvironment dynamically swaps texture on diamond shaders', async () => {
  const { updateDiamondEnvironment } = await import('../app/ring-runtime.ts');
  const initial = new Texture();
  const updated = new Texture();
  const material = {
    uniforms: {
      envMap: { value: initial },
    },
    needsUpdate: false,
  };
  updateDiamondEnvironment([material], updated);
  assert.equal(material.uniforms.envMap.value, updated);
  assert.equal(material.needsUpdate, true);
  initial.dispose();
  updated.dispose();
});

test('createMetalMaterial and applyMetalPreset configure physical jewelry materials with presets', async () => {
  const { createMetalMaterial, applyMetalPreset } = await import('../app/ring-runtime.ts');
  const { METAL_OPTIONS } = await import('../app/ring-catalog.ts');

  const yellow = METAL_OPTIONS.find((m) => m.id === 'yellow');
  const rose = METAL_OPTIONS.find((m) => m.id === 'rose');
  const white = METAL_OPTIONS.find((m) => m.id === 'white');

  assert.ok(yellow && rose && white);

  const mat = createMetalMaterial(yellow);
  assert.ok(mat instanceof MeshPhysicalMaterial);
  assert.equal(mat.color.getHex(), yellow.color);
  assert.equal(mat.roughness, yellow.roughness);
  assert.equal(mat.metalness, yellow.metalness);
  assert.equal(mat.clearcoat, yellow.clearcoat);
  assert.equal(mat.clearcoatRoughness, yellow.clearcoatRoughness);
  assert.equal(mat.envMapIntensity, yellow.envMapIntensity);
  assert.equal(mat.specularIntensity, yellow.specularIntensity);
  assert.equal(mat.specularColor.getHex(), yellow.specularColor);

  applyMetalPreset(mat, rose);
  assert.equal(mat.color.getHex(), rose.color);
  assert.equal(mat.roughness, rose.roughness);
  assert.equal(mat.metalness, rose.metalness);
  assert.equal(mat.specularColor.getHex(), rose.specularColor);

  applyMetalPreset(mat, white);
  assert.equal(mat.color.getHex(), white.color);
  assert.equal(mat.roughness, white.roughness);
  assert.equal(mat.clearcoat, white.clearcoat);

  mat.dispose();
});


