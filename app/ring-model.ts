import * as THREE from 'three';

/** Fallback when mesh measurement is unavailable (LR1844 inner/outer). */
export const DEFAULT_INNER_TO_OUTER_DIAMETER = 0.797;

/** Inner hole slightly larger than skin span so metal kisses the finger. */
export const WEAR_CLEARANCE = 1.02;

/** Soft floor vs proximal bone length (reject tiny collapsed widths). */
export const MIN_WIDTH_BONE_RATIO = 0.18;
/** Hard cap vs proximal bone length so a bad silhouette cannot explode scale. */
export const MAX_WIDTH_BONE_RATIO = 0.52;
/** Minimum usable finger width in screen pixels. */
export const MIN_FINGER_WIDTH_PX = 8;

/** Finger depth mask: radius as a fraction of measured finger width. */
export const OCCLUSION_RADIUS_RATIO = 0.5;
/** Finger depth mask: length along the finger as a fraction of finger width. */
export const OCCLUSION_LENGTH_RATIO = 1.35;
/** Tip radius / base radius for the tapered finger occluder. */
export const OCCLUSION_TIP_TAPER = 0.88;

export type RingModelExtras = {
  innerDiameter?: number;
  stoneAxis?: string;
  metalTint?: string;
};

function meshLabel(object: THREE.Mesh): string {
  const materials = Array.isArray(object.material)
    ? object.material
    : [object.material];
  return [object.name, ...materials.map((material) => material.name)]
    .join(' ')
    .toLowerCase();
}

export function isMetalMesh(object: THREE.Mesh): boolean {
  return /\b(metal|gold|silver|platinum)\b/.test(meshLabel(object));
}

export function isGemMesh(object: THREE.Mesh): boolean {
  return /\b(gem|diamond|stone)\b/.test(meshLabel(object));
}

/**
 * Metal-band AABB (falls back to the full model when no metal meshes exist).
 * Used so the unit hole shares the occluder / placement origin in XYZ.
 */
export function measureMetalBounds(
  root: THREE.Object3D,
): { center: THREE.Vector3; size: THREE.Vector3 } | null {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3();
  let has = false;

  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    if (isGemMesh(child)) return;
    if (!isMetalMesh(child)) return;
    const meshBox = new THREE.Box3().setFromObject(child);
    if (meshBox.isEmpty()) return;
    if (!has) {
      box.copy(meshBox);
      has = true;
    } else {
      box.union(meshBox);
    }
  });

  if (!has) {
    box.setFromObject(root);
    if (box.isEmpty()) return null;
  }

  return {
    center: box.getCenter(new THREE.Vector3()),
    size: box.getSize(new THREE.Vector3()),
  };
}

/**
 * Translate `target` so the metal AABB center lands at the origin (+ optional trim).
 * Returns the pre-translation metal center in world space.
 */
export function centerRingOnMetalHole(
  root: THREE.Object3D,
  target: THREE.Object3D = root,
  offset: { x?: number; y?: number; z?: number } = {},
): THREE.Vector3 {
  root.updateMatrixWorld(true);
  const metal = measureMetalBounds(root);
  const center =
    metal?.center.clone() ??
    new THREE.Box3().setFromObject(root).getCenter(new THREE.Vector3());

  target.position.x -= center.x;
  target.position.y -= center.y;
  target.position.z -= center.z;
  target.position.x += offset.x || 0;
  target.position.y += offset.y || 0;
  target.position.z += offset.z || 0;
  root.updateMatrixWorld(true);
  return center;
}

/**
 * Read optional SKU metadata from the glTF root / scene userData or extras.
 */
export function readRingModelExtras(root: THREE.Object3D): RingModelExtras {
  const sources: Record<string, unknown>[] = [];
  if (root.userData && typeof root.userData === 'object') {
    sources.push(root.userData as Record<string, unknown>);
  }
  root.traverse((object) => {
    if (object.userData && typeof object.userData === 'object') {
      sources.push(object.userData as Record<string, unknown>);
    }
  });

  const extras: RingModelExtras = {};
  for (const source of sources) {
    const nested =
      source.extras && typeof source.extras === 'object'
        ? (source.extras as Record<string, unknown>)
        : source;
    if (
      typeof nested.innerDiameter === 'number' &&
      Number.isFinite(nested.innerDiameter) &&
      nested.innerDiameter > 0
    ) {
      extras.innerDiameter = nested.innerDiameter;
    }
    if (typeof nested.stoneAxis === 'string' && nested.stoneAxis) {
      extras.stoneAxis = nested.stoneAxis;
    }
    if (typeof nested.metalTint === 'string' && nested.metalTint) {
      extras.metalTint = nested.metalTint;
    }
  }
  return extras;
}

/**
 * Measure the ring's inner opening from metal-band vertices near the midplane.
 * Falls back to outerAABB.x * fallbackRatio when measurement is unreliable.
 */
export function measureRingInnerDiameter(
  root: THREE.Object3D,
  fallbackRatio = DEFAULT_INNER_TO_OUTER_DIAMETER,
): number {
  root.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(root);
  const size = bounds.getSize(new THREE.Vector3());
  const outer = Math.max(size.x, 0.001);
  const fallback = Math.max(outer * fallbackRatio, 0.001);

  const metalMeshes: THREE.Mesh[] = [];
  root.traverse((object) => {
    if (object instanceof THREE.Mesh && isMetalMesh(object)) {
      metalMeshes.push(object);
    }
  });
  if (metalMeshes.length === 0) return fallback;

  const radii: number[] = [];
  const vertex = new THREE.Vector3();
  const bandYLimit = Math.max(size.y * 0.4, 0.05);
  const bandZLimit = outer * 0.55;

  for (const mesh of metalMeshes) {
    const position = mesh.geometry.getAttribute('position');
    if (!position) continue;
    for (let i = 0; i < position.count; i++) {
      vertex.fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld);
      if (Math.abs(vertex.y) > bandYLimit) continue;
      if (Math.abs(vertex.z) > bandZLimit) continue;
      // Finger axis is model Y; radial distance in the XZ cross-section.
      radii.push(Math.hypot(vertex.x, vertex.z));
    }
  }

  if (radii.length < 48) return fallback;

  radii.sort((a, b) => a - b);
  // Inner wall sits at the low end of the radial distribution.
  const index = Math.min(
    radii.length - 1,
    Math.max(0, Math.floor(radii.length * 0.015)),
  );
  const inner = radii[index] * 2;
  if (inner < outer * 0.5 || inner > outer * 0.95) return fallback;
  return Math.max(inner, 0.001);
}

/**
 * Prefer silhouette width; clamp to an anatomical range of the proximal bone
 * (MCP→PIP / landmarks 13→14) so bad edges cannot explode or collapse scale.
 */
export function resolveFingerWidthPx(
  widthPixels: number,
  boneLengthPx = 0,
): number {
  const bone = Number(boneLengthPx) || 0;
  const measured = Number(widthPixels);
  let width =
    Number.isFinite(measured) && measured >= MIN_FINGER_WIDTH_PX
      ? measured
      : Math.max(MIN_FINGER_WIDTH_PX, bone * 0.35);

  if (bone >= MIN_FINGER_WIDTH_PX) {
    const lo = bone * MIN_WIDTH_BONE_RATIO;
    const hi = bone * MAX_WIDTH_BONE_RATIO;
    width = Math.min(hi, Math.max(lo, width));
  }
  return width;
}

/**
 * Uniformly scale the model so its inner opening is 1.0 world unit.
 * Optional `fitScale` is baked into the same factor (SKU trim without a second child scale).
 * Runtime then sets `ring.scale = fingerWorld * WEAR_CLEARANCE`.
 */
export function normalizeRingToUnitHole(
  model: THREE.Object3D,
  rawInnerDiameter: number,
  fitScale = 1,
): number {
  const raw = Math.max(Number(rawInnerDiameter) || 0, 0.001);
  const trim =
    Number.isFinite(fitScale) && fitScale > 0 ? fitScale : 1;
  const factor = (1 / raw) * trim;
  model.scale.multiplyScalar(factor);
  model.updateMatrixWorld(true);
  model.userData.rawInnerDiameter = raw;
  model.userData.innerDiameter = 1;
  model.userData.fitScale = trim;
  return factor;
}

/**
 * Bake authored stone/head axis onto model −Z so runtime pose + Flip share one head direction.
 * Supported: -Z (default/no-op), +Z, ±Y, ±X.
 */
export function applyStoneAxisBasis(
  model: THREE.Object3D,
  stoneAxis?: string | null,
): THREE.Quaternion {
  const axis = String(stoneAxis || '-Z').trim().toUpperCase();
  const q = new THREE.Quaternion();
  if (axis === '+Z' || axis === 'Z') {
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
  } else if (axis === '+Y' || axis === 'Y') {
    q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  } else if (axis === '-Y') {
    q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
  } else if (axis === '+X' || axis === 'X') {
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
  } else if (axis === '-X') {
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
  }
  // -Z / unknown: identity (LR1844 native).
  if (q.x !== 0 || q.y !== 0 || q.z !== 0 || q.w !== 1) {
    model.quaternion.premultiply(q);
    model.updateMatrixWorld(true);
  }
  model.userData.stoneAxis = '-Z';
  model.userData.stoneAxisSource = axis;
  return q;
}

/** World scale for a unit-hole ring given exact finger diameter in world units. */
export function ringWorldScale(
  fingerWorldDiameter: number,
  wearClearance = WEAR_CLEARANCE,
): number {
  const clearance =
    Number.isFinite(wearClearance) && wearClearance > 0
      ? wearClearance
      : WEAR_CLEARANCE;
  return Math.max(fingerWorldDiameter, 1e-6) * clearance;
}

/**
 * Tapered closed capsule approximation of the finger for depth occlusion.
 * Geometry is unit-sized along Y; callers scale by finger width.
 */
export function createFingerOccluderGeometry(): THREE.BufferGeometry {
  return new THREE.CylinderGeometry(
    OCCLUSION_TIP_TAPER,
    1,
    1,
    64,
    1,
    false,
  );
}

/**
 * Bake signed (including mirrored) mesh scale into geometry so BVH builds
 * operate in local unit scale.
 */
export function bakeMeshScale(object: THREE.Mesh): void {
  const { x, y, z } = object.scale;
  if (
    Math.abs(x - 1) < 1e-6 &&
    Math.abs(y - 1) < 1e-6 &&
    Math.abs(z - 1) < 1e-6
  ) {
    return;
  }
  const bakedGeometry = object.geometry.clone();
  bakedGeometry.applyMatrix4(new THREE.Matrix4().makeScale(x, y, z));
  if (x * y * z < 0) {
    const index = bakedGeometry.getIndex();
    if (index) {
      for (let i = 0; i < index.count; i += 3) {
        const b = index.getX(i + 1);
        index.setX(i + 1, index.getX(i + 2));
        index.setX(i + 2, b);
      }
      index.needsUpdate = true;
    }
  }
  bakedGeometry.computeVertexNormals();
  object.geometry = bakedGeometry;
  object.scale.set(1, 1, 1);
}

/** Stable signature for sharing BVHs across identical gem geometries. */
export function geometrySignature(geometry: THREE.BufferGeometry): string {
  const position = geometry.getAttribute('position');
  if (!position) return `empty:${geometry.uuid}`;
  const count = position.count;
  const arr = position.array;
  const stride = position.itemSize;
  let hash = count * 73856093;
  const samples = [
    0,
    Math.floor(count / 4),
    Math.floor(count / 2),
    Math.floor((3 * count) / 4),
    count - 1,
  ];
  for (const index of samples) {
    if (index < 0 || index >= count) continue;
    const offset = index * stride;
    hash =
      (hash ^ (Math.fround(arr[offset] as number) * 19349663)) >>> 0;
    hash =
      (hash ^ (Math.fround(arr[offset + 1] as number) * 83492791)) >>> 0;
    hash =
      (hash ^ (Math.fround(arr[offset + 2] as number) * 2654435761)) >>> 0;
  }
  return `${count}:${hash}`;
}
