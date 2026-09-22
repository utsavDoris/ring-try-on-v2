import * as THREE from 'three';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';

/** Primary diamond studio HDR (brighter highlights for camera-sized gems). */
export const DIAMOND_ENV_URL = '/diamond_env.exr';

const equirectCache = new Map<string, Promise<THREE.Texture>>();
const cubeEnvCache = new Map<string, THREE.WebGLCubeRenderTarget>();
const sharedEquirects = new Set<THREE.Texture>();

/**
 * Loads and caches an equirectangular HDR.
 */
export async function loadEquirectEnvironment(
  url: string,
  load: (target: string) => Promise<THREE.Texture> = (target) =>
    new EXRLoader().loadAsync(target),
): Promise<THREE.Texture> {
  const cached = equirectCache.get(url);
  if (cached) return cached;

  const promise = load(url)
    .then((texture) => {
      texture.mapping = THREE.EquirectangularReflectionMapping;
      texture.colorSpace = THREE.LinearSRGBColorSpace;
      sharedEquirects.add(texture);
      return texture;
    })
    .catch((error) => {
      if (equirectCache.get(url) === promise) equirectCache.delete(url);
      throw error;
    });
  equirectCache.set(url, promise);
  return promise;
}

/**
 * Convert an equirectangular EXR into a cube map for the diamond `samplerCube`.
 * Results are cached per renderer + URL so SKU swaps reuse the same cube RT.
 */
export function createCubeEnvironment(
  renderer: THREE.WebGLRenderer,
  equirect: THREE.Texture,
  size = 512,
  cacheKey?: string,
): THREE.WebGLCubeRenderTarget {
  const key = cacheKey ?? `default:${size}`;
  const cached = cubeEnvCache.get(key);
  if (cached) return cached;

  const cubeRT = new THREE.WebGLCubeRenderTarget(size, {
    type: THREE.HalfFloatType,
    colorSpace: THREE.LinearSRGBColorSpace,
    generateMipmaps: true,
    minFilter: THREE.LinearMipmapLinearFilter,
  });
  cubeRT.fromEquirectangularTexture(renderer, equirect);
  const cubeMap = cubeRT.texture;
  cubeMap.mapping = THREE.CubeReflectionMapping;
  cubeMap.generateMipmaps = true;
  cubeMap.minFilter = THREE.LinearMipmapLinearFilter;
  cubeMap.magFilter = THREE.LinearFilter;
  const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
  if (maxAnisotropy > 1) {
    cubeMap.anisotropy = Math.min(maxAnisotropy, 4);
  }
  renderer.setRenderTarget(null);
  cubeEnvCache.set(key, cubeRT);
  return cubeRT;
}

/**
 * True when this module owns the texture.
 */
export function isSharedEnvironmentTexture(texture: THREE.Texture): boolean {
  if (sharedEquirects.has(texture)) return true;
  for (const target of cubeEnvCache.values()) {
    if (target.texture === texture) return true;
  }
  return false;
}

/** Releases every cached environment. */
export function disposeSharedEnvironments(): void {
  for (const target of cubeEnvCache.values()) {
    target.dispose();
  }
  cubeEnvCache.clear();
  for (const texture of sharedEquirects) {
    texture.dispose();
  }
  sharedEquirects.clear();
  equirectCache.clear();
}
