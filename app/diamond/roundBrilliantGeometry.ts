import * as THREE from 'three';

/** Modern-ideal / Tolkowsky-ish proportions (matches LR1844 ~62% depth). */
export const BRILLIANT_TABLE = 0.57;
export const BRILLIANT_CROWN_ANGLE = (34.5 * Math.PI) / 180;
export const BRILLIANT_PAVILION_ANGLE = (40.75 * Math.PI) / 180;
export const BRILLIANT_GIRDLE = 0.035;
export const BRILLIANT_STAR_LENGTH = 0.55;
export const BRILLIANT_LOWER_GIRDLE = 0.75;
export const BRILLIANT_DEPTH = 0.623;

export interface RoundBrilliantOptions {
  /** Girdle diameter. Default 1. */
  diameter?: number;
  /** Total depth as a fraction of diameter. Default 0.623. */
  depthRatio?: number;
}

function addTri(
  positions: number[],
  a: THREE.Vector3,
  b: THREE.Vector3,
  c: THREE.Vector3,
): void {
  positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
}

function addQuad(
  positions: number[],
  a: THREE.Vector3,
  b: THREE.Vector3,
  c: THREE.Vector3,
  d: THREE.Vector3,
): void {
  addTri(positions, a, b, c);
  addTri(positions, a, c, d);
}

function ringPoint(radius: number, angle: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, z);
}

function crownZ(r: number, girdleR: number, zGirdleTop: number, tanCrown: number): number {
  return zGirdleTop + (girdleR - r) * tanCrown;
}

function pavilionZ(r: number, girdleR: number, zGirdleBot: number, tanPav: number): number {
  return zGirdleBot - (girdleR - r) * tanPav;
}

function ensureOutwardAndCenter(pos: Float32Array): void {
  const count = pos.length / 3;
  const centroid = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    centroid.x += pos[i * 3];
    centroid.y += pos[i * 3 + 1];
    centroid.z += pos[i * 3 + 2];
  }
  centroid.multiplyScalar(1 / Math.max(count, 1));

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const n = new THREE.Vector3();
  const triC = new THREE.Vector3();

  for (let i = 0; i < count; i += 3) {
    a.fromArray(pos, i * 3);
    b.fromArray(pos, (i + 1) * 3);
    c.fromArray(pos, (i + 2) * 3);
    n.crossVectors(ab.subVectors(b, a), ac.subVectors(c, a));
    triC.copy(a).add(b).add(c).multiplyScalar(1 / 3).sub(centroid);
    if (n.dot(triC) < 0) {
      pos[(i + 1) * 3] = c.x;
      pos[(i + 1) * 3 + 1] = c.y;
      pos[(i + 1) * 3 + 2] = c.z;
      pos[(i + 2) * 3] = b.x;
      pos[(i + 2) * 3 + 1] = b.y;
      pos[(i + 2) * 3 + 2] = b.z;
    }
  }

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = pos[i * 3];
    const y = pos[i * 3 + 1];
    const z = pos[i * 3 + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  for (let i = 0; i < count; i++) {
    pos[i * 3] -= cx;
    pos[i * 3 + 1] -= cy;
    pos[i * 3 + 2] -= cz;
  }
}

/**
 * 57-facet round brilliant (plus a 16-segment planar girdle) with table on +Z
 * and culet on −Z. One plane per facet — two triangles for quads.
 */
export function createRoundBrilliantGeometry(
  options: RoundBrilliantOptions = {},
): THREE.BufferGeometry {
  const diameter = options.diameter ?? 1;
  const depthRatio = options.depthRatio ?? BRILLIANT_DEPTH;
  const R = diameter * 0.5;
  const tableR = BRILLIANT_TABLE * R;
  const tanCrown = Math.tan(BRILLIANT_CROWN_ANGLE);
  const tanPav = Math.tan(BRILLIANT_PAVILION_ANGLE);

  const naturalCrown = (R - tableR) * tanCrown;
  const naturalPav = R * tanPav;
  const naturalGirdle = BRILLIANT_GIRDLE * diameter;
  const naturalDepth = naturalCrown + naturalGirdle + naturalPav;
  const depthScale = (depthRatio * diameter) / Math.max(naturalDepth, 1e-6);

  const crownH = naturalCrown * depthScale;
  const girdleH = naturalGirdle * depthScale;
  const pavH = naturalPav * depthScale;
  const tanCrownScaled = crownH / Math.max(R - tableR, 1e-6);
  const tanPavScaled = pavH / R;

  const zGirdleTop = girdleH * 0.5;
  const zGirdleBot = -girdleH * 0.5;
  const zTable = crownZ(tableR, R, zGirdleTop, tanCrownScaled);
  const zCulet = pavilionZ(0, R, zGirdleBot, tanPavScaled);

  const tableEdgeMidR = tableR * Math.cos(Math.PI / 8);
  const starR = tableEdgeMidR + (R - tableEdgeMidR) * BRILLIANT_STAR_LENGTH;
  const zStar = crownZ(starR, R, zGirdleTop, tanCrownScaled);
  const pvR = R * (1 - BRILLIANT_LOWER_GIRDLE);
  const zPv = pavilionZ(pvR, R, zGirdleBot, tanPavScaled);

  const tableCenter = new THREE.Vector3(0, 0, zTable);
  const culet = new THREE.Vector3(0, 0, zCulet);
  const T: THREE.Vector3[] = [];
  const S: THREE.Vector3[] = [];
  const GMtop: THREE.Vector3[] = [];
  const GMbot: THREE.Vector3[] = [];
  const GHtop: THREE.Vector3[] = [];
  const GHbot: THREE.Vector3[] = [];
  const PV: THREE.Vector3[] = [];
  const GU: THREE.Vector3[] = [];
  const GL: THREE.Vector3[] = [];

  for (let i = 0; i < 8; i++) {
    const mainA = (i * Math.PI) / 4;
    const halfA = mainA + Math.PI / 8;
    T.push(ringPoint(tableR, mainA, zTable));
    S.push(ringPoint(starR, halfA, zStar));
    GMtop.push(ringPoint(R, mainA, zGirdleTop));
    GMbot.push(ringPoint(R, mainA, zGirdleBot));
    GHtop.push(ringPoint(R, halfA, zGirdleTop));
    GHbot.push(ringPoint(R, halfA, zGirdleBot));
    PV.push(ringPoint(pvR, halfA, zPv));
  }

  for (let i = 0; i < 16; i++) {
    const a = (i * Math.PI) / 8;
    GU.push(ringPoint(R, a, zGirdleTop));
    GL.push(ringPoint(R, a, zGirdleBot));
  }

  const positions: number[] = [];

  for (let i = 0; i < 8; i++) {
    addTri(positions, tableCenter, T[i], T[(i + 1) % 8]);
  }

  for (let i = 0; i < 8; i++) {
    const i1 = (i + 1) % 8;
    const iPrev = (i + 7) % 8;

    addTri(positions, T[i], T[i1], S[i]);
    addQuad(positions, T[i1], S[i], GMtop[i1], S[i1]);
    addTri(positions, S[i], GHtop[i], GMtop[i]);
    addTri(positions, S[i], GMtop[i1], GHtop[i]);

    addQuad(positions, culet, PV[iPrev], GMbot[i], PV[i]);
    addTri(positions, PV[i], GMbot[i], GHbot[i]);
    addTri(positions, PV[i], GHbot[i], GMbot[i1]);
  }

  for (let i = 0; i < 16; i++) {
    addQuad(positions, GU[i], GU[(i + 1) % 16], GL[(i + 1) % 16], GL[i]);
  }

  const pos = new Float32Array(positions);
  ensureOutwardAndCenter(pos);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}
