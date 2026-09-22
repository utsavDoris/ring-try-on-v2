import * as THREE from 'three';

// A DPR limit alone still allocates a multi-megapixel framebuffer on an iPad.
export function ringPixelRatio(width: number, height: number, dpr: number, touch: boolean) {
  const area = Math.max(1, width * height);
  // Increased pixel ratio limits for sharper rendering
  return Math.min(dpr || 1, touch ? 2.0 : 2.5, Math.sqrt((touch ? 2_000_000 : 3_500_000) / area));
}

export function limitEnvironmentSize(texture: THREE.Texture, maxWidth: number) {
  if (!(texture instanceof THREE.DataTexture) || texture.type !== THREE.HalfFloatType) return texture;
  const source = texture.image;
  if (!(source.data instanceof Uint16Array) || source.width <= maxWidth) return texture;
  const step = 2 ** Math.ceil(Math.log2(source.width / maxWidth));
  const width = Math.max(1, Math.floor(source.width / step));
  const height = Math.max(1, Math.floor(source.height / step));
  const data = new Uint16Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < 4; c++) {
    let sum = 0;
    for (let sy = 0; sy < step; sy++) for (let sx = 0; sx < step; sx++) {
      sum += THREE.DataUtils.fromHalfFloat(source.data[((y * step + sy) * source.width + x * step + sx) * 4 + c]);
    }
    data[(y * width + x) * 4 + c] = THREE.DataUtils.toHalfFloat(sum / (step * step));
  }
  // Before the first GPU upload, release the large decoded buffer and bound
  // the subsequent PMREM cube allocation as well as the source texture.
  texture.image = { data, width, height };
  texture.needsUpdate = true;
  return texture;
}

export function createFallbackEnvironment() {
  // Small linear-HDR studio light map; used only if the supplied file fails.
  const data = new Float32Array(16 * 8 * 4);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 16; x++) {
    const light = y < 4 && (x === 3 || x === 4 || x === 11 || x === 12) ? 3 : y < 4 ? 0.5 : 0.08;
    const offset = (y * 16 + x) * 4;
    data.set([light, light, light, 1], offset);
  }
  const texture = new THREE.DataTexture(data, 16, 8, THREE.RGBAFormat, THREE.FloatType);
  // No float filtering extension is required for this emergency map.
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.needsUpdate = true;
  return texture;
}

export async function loadOptionalEnvironment(load: () => Promise<THREE.Texture>) {
  try { return await load(); }
  catch (error) {
    console.warn('Lighting file failed; keeping the ring available with studio fallback.', error);
    return createFallbackEnvironment();
  }
}

export function createFallbackDiamondMaterial() {
  // No BVH textures or offscreen transmission pass: keep stones visible if
  // the device rejects the custom shader, while retaining the original mesh.
  return new THREE.MeshPhysicalMaterial({
    color: '#ececec', metalness: 0, roughness: 0.0,
    ior: 2.4, envMapIntensity: 1.8, clearcoat: 1,
    transparent: true, depthWrite: true,
  });
}

const ASSET_CACHE_NAME = 'lustre-assets-v2';

export async function fetchWithProgressAndCache(
  url: string,
  onProgress?: (ratio: number) => void,
): Promise<ArrayBuffer> {
  if (typeof window !== 'undefined' && 'caches' in window) {
    try {
      const cache = await window.caches.open(ASSET_CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        onProgress?.(1);
        return await cached.arrayBuffer();
      }
    } catch {
      // Ignore cache open/match errors and fall back to network.
    }
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }

  const contentLengthHeader = response.headers.get('content-length');
  const totalBytes = contentLengthHeader ? parseInt(contentLengthHeader, 10) : 0;

  let buffer: ArrayBuffer;
  if (response.body && totalBytes > 0 && typeof ReadableStream !== 'undefined') {
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let receivedBytes = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        receivedBytes += value.length;
        if (onProgress) {
          onProgress(Math.min(1, receivedBytes / totalBytes));
        }
      }
    }

    const combined = new Uint8Array(receivedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    buffer = combined.buffer;
  } else {
    buffer = await response.arrayBuffer();
    onProgress?.(1);
  }

  if (typeof window !== 'undefined' && 'caches' in window) {
    try {
      const cache = await window.caches.open(ASSET_CACHE_NAME);
      const headers = new Headers(response.headers);
      headers.set('content-type', response.headers.get('content-type') || 'application/octet-stream');
      await cache.put(url, new Response(buffer.slice(0), { headers }));
    } catch {
      // Ignore cache put errors (e.g. storage quota exceeded or private mode).
    }
  }

  return buffer;
}

export function updateDiamondEnvironment(
  materials: THREE.ShaderMaterial[],
  texture: THREE.Texture,
) {
  for (const material of materials) {
    if (material.uniforms?.envMap) {
      material.uniforms.envMap.value = texture;
      material.needsUpdate = true;
    }
  }
}
