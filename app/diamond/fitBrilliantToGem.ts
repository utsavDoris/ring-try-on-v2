import * as THREE from 'three';
import { BRILLIANT_DEPTH, createRoundBrilliantGeometry } from './roundBrilliantGeometry.ts';

const ROUND_ASPECT_MIN = 0.85;
const ROUND_ASPECT_MAX = 1.15;
const ROUND_DEPTH_MIN = 0.50;
const ROUND_DEPTH_MAX = 0.70;
const PLANE_DOT = 0.992;
const PLANE_DIST = 1e-4;

const _size = new THREE.Vector3();
const _center = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _ab = new THREE.Vector3();
const _ac = new THREE.Vector3();
const _n = new THREE.Vector3();

interface GemBounds {
  size: THREE.Vector3;
  center: THREE.Vector3;
  diameter: number;
  depth: number;
  depthAxis: 'x' | 'y' | 'z';
}

function toNonIndexed(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  return geometry.index ? geometry.toNonIndexed() : geometry.clone();
}

function measureGem(geometry: THREE.BufferGeometry): GemBounds | null {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  if (!box || box.isEmpty()) return null;
  const size = box.getSize(_size).clone();
  const center = box.getCenter(_center).clone();
  const axes: Array<{ axis: 'x' | 'y' | 'z'; value: number }> = [
    { axis: 'x' as const, value: size.x },
    { axis: 'y' as const, value: size.y },
    { axis: 'z' as const, value: size.z },
  ].sort((a, b) => a.value - b.value);
  const depth = axes[0].value;
  const d1 = axes[1].value;
  const d2 = axes[2].value;
  return {
    size,
    center,
    diameter: (d1 + d2) * 0.5,
    depth,
    depthAxis: axes[0].axis,
  };
}

export function isRoundBrilliantShape(size: THREE.Vector3): boolean {
  const dims = [size.x, size.y, size.z].sort((a, b) => a - b);
  const depth = dims[0];
  const diameter = (dims[1] + dims[2]) * 0.5;
  if (diameter < 1e-8) return false;
  const aspect = dims[1] / dims[2];
  const depthRatio = depth / diameter;
  return (
    aspect >= ROUND_ASPECT_MIN
    && aspect <= ROUND_ASPECT_MAX
    && depthRatio >= ROUND_DEPTH_MIN
    && depthRatio <= ROUND_DEPTH_MAX
  );
}

/**
 * Largest coplanar triangle cluster → table normal, flipped outward from centroid.
 */
export function detectTableNormal(geometry: THREE.BufferGeometry): THREE.Vector3 {
  const geo = toNonIndexed(geometry);
  const pos = geo.attributes.position;
  const clusters: Array<{ normal: THREE.Vector3; area: number; origin: THREE.Vector3 }> = [];
  const centroid = new THREE.Vector3();
  let vertCount = 0;

  for (let i = 0; i < pos.count; i += 3) {
    _a.fromBufferAttribute(pos, i);
    _b.fromBufferAttribute(pos, i + 1);
    _c.fromBufferAttribute(pos, i + 2);
    centroid.add(_a).add(_b).add(_c);
    vertCount += 3;
    _n.crossVectors(_ab.subVectors(_b, _a), _ac.subVectors(_c, _a));
    const area = _n.length() * 0.5;
    if (area < 1e-12) continue;
    _n.normalize();
    let matched = false;
    for (const cluster of clusters) {
      if (Math.abs(cluster.normal.dot(_n)) < PLANE_DOT) continue;
      const dist = Math.abs(_a.clone().sub(cluster.origin).dot(cluster.normal));
      if (dist > Math.max(PLANE_DIST, area * 0.01)) continue;
      if (_n.dot(cluster.normal) < 0) _n.negate();
      cluster.normal.multiplyScalar(cluster.area).add(_n.multiplyScalar(area)).normalize();
      cluster.area += area;
      matched = true;
      break;
    }
    if (!matched) {
      clusters.push({
        normal: _n.clone(),
        area,
        origin: _a.clone(),
      });
    }
  }

  if (vertCount > 0) centroid.multiplyScalar(1 / vertCount);
  clusters.sort((a, b) => b.area - a.area);
  const table = clusters[0]?.normal.clone() ?? new THREE.Vector3(0, 0, 1);
  if (table.lengthSq() < 1e-8) table.set(0, 0, 1);
  else table.normalize();

  geo.computeBoundingBox();
  const center = geo.boundingBox?.getCenter(new THREE.Vector3()) ?? centroid;
  const tablePoint = clusters[0]?.origin ?? center.clone().add(table);
  if (tablePoint.clone().sub(center).dot(table) < 0) table.negate();
  if (geo !== geometry) geo.dispose();
  return table;
}

function axisVector(axis: 'x' | 'y' | 'z'): THREE.Vector3 {
  if (axis === 'x') return new THREE.Vector3(1, 0, 0);
  if (axis === 'y') return new THREE.Vector3(0, 1, 0);
  return new THREE.Vector3(0, 0, 1);
}

function fitBrilliant(geometry: THREE.BufferGeometry, bounds: GemBounds): THREE.BufferGeometry {
  const diameter = Math.max(bounds.diameter, 1e-6);
  const depthRatio = THREE.MathUtils.clamp(
    bounds.depth / diameter,
    ROUND_DEPTH_MIN,
    ROUND_DEPTH_MAX,
  );
  const brilliant = createRoundBrilliantGeometry({
    diameter,
    depthRatio: Number.isFinite(depthRatio) ? depthRatio : BRILLIANT_DEPTH,
  });

  let table = detectTableNormal(geometry);
  const depthDir = axisVector(bounds.depthAxis);
  if (Math.abs(table.dot(depthDir)) < 0.55) {
    const sign = Math.sign(table.dot(depthDir)) || 1;
    table = depthDir.multiplyScalar(sign);
  }

  const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), table.normalize());
  brilliant.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(quat));
  brilliant.translate(bounds.center.x, bounds.center.y, bounds.center.z);
  brilliant.computeBoundingBox();
  brilliant.computeBoundingSphere();
  return brilliant;
}

/**
 * Snap coplanar CAD triangles onto one plane per facet so inner TIR stays a hard cut.
 */
export function planarizeFacets(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  const geo = toNonIndexed(geometry);
  geo.computeBoundingBox();
  const maxDim = geo.boundingBox
    ? Math.max(
      geo.boundingBox.max.x - geo.boundingBox.min.x,
      geo.boundingBox.max.y - geo.boundingBox.min.y,
      geo.boundingBox.max.z - geo.boundingBox.min.z,
    )
    : 1;
  const planeDist = Math.max(PLANE_DIST, maxDim * 0.002);
  const pos = geo.attributes.position;
  const src = pos.array as Float32Array;
  const next = new Float32Array(src.length);

  type Cluster = { normal: THREE.Vector3; origin: THREE.Vector3; indices: number[] };
  const clusters: Cluster[] = [];

  for (let i = 0; i < pos.count; i += 3) {
    _a.fromBufferAttribute(pos, i);
    _b.fromBufferAttribute(pos, i + 1);
    _c.fromBufferAttribute(pos, i + 2);
    _n.crossVectors(_ab.subVectors(_b, _a), _ac.subVectors(_c, _a));
    if (_n.lengthSq() < 1e-20) {
      next.set(src.subarray(i * 3, i * 3 + 9), i * 3);
      continue;
    }
    _n.normalize();
    let cluster: Cluster | undefined;
    for (const candidate of clusters) {
      if (Math.abs(candidate.normal.dot(_n)) < PLANE_DOT) continue;
      const dist = Math.abs(_a.clone().sub(candidate.origin).dot(candidate.normal));
      if (dist > planeDist * 8) continue;
      cluster = candidate;
      if (_n.dot(candidate.normal) < 0) _n.negate();
      candidate.normal.add(_n).normalize();
      break;
    }
    if (!cluster) {
      cluster = { normal: _n.clone(), origin: _a.clone(), indices: [] };
      clusters.push(cluster);
    }
    cluster.indices.push(i, i + 1, i + 2);
  }

  const tmp = new THREE.Vector3();
  for (const cluster of clusters) {
    const { normal, origin } = cluster;
    for (const vi of cluster.indices) {
      tmp.fromBufferAttribute(pos, vi);
      const dist = tmp.clone().sub(origin).dot(normal);
      tmp.addScaledVector(normal, -dist);
      next[vi * 3] = tmp.x;
      next[vi * 3 + 1] = tmp.y;
      next[vi * 3 + 2] = tmp.z;
    }
  }

  geo.setAttribute('position', new THREE.BufferAttribute(next, 3));
  geo.computeVertexNormals();
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  return geo;
}

/**
 * Replace a round CAD gem with a 57-facet brilliant fitted to its AABB, or
 * planarize fancy cuts. Safe to call once per unique BufferGeometry.
 */
export function rebuildGemCut(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  const bounds = measureGem(geometry);
  if (!bounds) return geometry;

  if (isRoundBrilliantShape(bounds.size)) {
    return fitBrilliant(geometry, bounds);
  }
  return planarizeFacets(geometry);
}

/**
 * Rebuild every gem mesh on a loaded ring template. Shared geometries are
 * processed once so halo/melee instances keep the same cut.
 */
export function rebuildTemplateGems(root: THREE.Object3D): void {
  const cache = new Map<THREE.BufferGeometry, THREE.BufferGeometry>();

  root.traverse((child) => {
    if (!(child as THREE.Mesh).isMesh) return;
    const mesh = child as THREE.Mesh;
    const mat = mesh.material;
    const matName = (
      Array.isArray(mat) ? mat[0]?.name : (mat as THREE.Material | undefined)?.name || ''
    ).toLowerCase();
    const nodeName = (mesh.name || '').toLowerCase();
    const label = `${matName} ${nodeName}`;
    if (!/\b(gem|diamond|stone)\b/i.test(label) && !label.includes('gem')) return;

    const src = mesh.geometry;
    if (!src) return;
    let next = cache.get(src);
    if (!next) {
      try {
        next = rebuildGemCut(src);
      } catch {
        next = src;
      }
      cache.set(src, next);
    }
    mesh.geometry = next;
  });
}
