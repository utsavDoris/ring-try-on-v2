import * as THREE from 'three';
import type { PackedBvh } from './buildBvh.ts';

function makeDataTexture(texelCount: number): {
  texture: THREE.DataTexture;
  data: Float32Array;
  width: number;
  height: number;
} {
  const width = Math.max(1, Math.ceil(Math.sqrt(texelCount)));
  const height = Math.max(1, Math.ceil(texelCount / width));
  const data = new Float32Array(width * height * 4);
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.colorSpace = THREE.NoColorSpace;
  return { texture, data, width, height };
}

export interface PackedBvhTextures {
  nodeTexture: THREE.DataTexture;
  triangleTexture: THREE.DataTexture;
  nodeTexSize: THREE.Vector2;
  triTexSize: THREE.Vector2;
  triangleCount: number;
  nodeCount: number;
}

/**
 * Pack BVH nodes (2 texels) and triangles (3 texels) into RGBA32F textures.
 *
 * Node texel 0: min.xyz, left  (leaf: left = -(triOffset+1))
 * Node texel 1: max.xyz, right (leaf: right = triCount)
 * Tri  texel 0..2: v0, v1, v2
 */
export function packBvhTextures(bvh: PackedBvh): PackedBvhTextures {
  const nodeTexels = bvh.nodes.length * 2;
  const triTexels = bvh.triangles.length * 3;

  const nodes = makeDataTexture(nodeTexels);
  const tris = makeDataTexture(triTexels);

  for (let i = 0; i < bvh.nodes.length; i++) {
    const n = bvh.nodes[i];
    const o = i * 2 * 4;
    const left = n.isLeaf ? -(n.triOffset + 1) : n.left;
    const right = n.isLeaf ? n.triCount : n.right;

    nodes.data[o + 0] = n.min.x;
    nodes.data[o + 1] = n.min.y;
    nodes.data[o + 2] = n.min.z;
    nodes.data[o + 3] = left;

    nodes.data[o + 4] = n.max.x;
    nodes.data[o + 5] = n.max.y;
    nodes.data[o + 6] = n.max.z;
    nodes.data[o + 7] = right;
  }

  for (let i = 0; i < bvh.triangles.length; i++) {
    const t = bvh.triangles[i];
    const o = i * 3 * 4;
    tris.data[o + 0] = t.a.x;
    tris.data[o + 1] = t.a.y;
    tris.data[o + 2] = t.a.z;
    tris.data[o + 3] = 0;
    tris.data[o + 4] = t.b.x;
    tris.data[o + 5] = t.b.y;
    tris.data[o + 6] = t.b.z;
    tris.data[o + 7] = 0;
    tris.data[o + 8] = t.c.x;
    tris.data[o + 9] = t.c.y;
    tris.data[o + 10] = t.c.z;
    tris.data[o + 11] = 0;
  }

  nodes.texture.needsUpdate = true;
  tris.texture.needsUpdate = true;

  return {
    nodeTexture: nodes.texture,
    triangleTexture: tris.texture,
    nodeTexSize: new THREE.Vector2(nodes.width, nodes.height),
    triTexSize: new THREE.Vector2(tris.width, tris.height),
    triangleCount: bvh.triangleCount,
    nodeCount: bvh.nodeCount,
  };
}

/** GPU cleanup for SKU swap / unload. Safe to call once per packed pair. */
export function disposePackedBvhTextures(
  packed: Pick<PackedBvhTextures, 'nodeTexture' | 'triangleTexture'>,
): void {
  packed.nodeTexture.dispose();
  packed.triangleTexture.dispose();
}
