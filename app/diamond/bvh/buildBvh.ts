import * as THREE from 'three';

const LEAF_SIZE = 4;

export interface DiamondTriangle {
  a: THREE.Vector3;
  b: THREE.Vector3;
  c: THREE.Vector3;
  centroid: THREE.Vector3;
}

export interface PackedBvh {
  nodes: FlatBvhNode[];
  triangles: DiamondTriangle[];
  triangleCount: number;
  nodeCount: number;
}

export interface FlatBvhNode {
  min: THREE.Vector3;
  max: THREE.Vector3;
  isLeaf: boolean;
  left: number;
  right: number;
  triOffset: number;
  triCount: number;
}

interface BuildNode {
  bounds: THREE.Box3;
  isLeaf: boolean;
  triangles?: DiamondTriangle[];
  left?: BuildNode;
  right?: BuildNode;
}

type Axis = 'x' | 'y' | 'z';

export function extractTriangles(geometry: THREE.BufferGeometry): DiamondTriangle[] {
  const pos = geometry.attributes.position;
  if (!pos) {
    throw new Error('Diamond geometry is missing a position attribute');
  }
  const triangles: DiamondTriangle[] = [];

  const read = (index: number, target: THREE.Vector3): THREE.Vector3 => {
    target.fromBufferAttribute(pos, index);
    return target;
  };

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();

  const push = (ia: number, ib: number, ic: number): void => {
    const va = read(ia, a).clone();
    const vb = read(ib, b).clone();
    const vc = read(ic, c).clone();
    triangles.push({
      a: va,
      b: vb,
      c: vc,
      centroid: va.clone().add(vb).add(vc).multiplyScalar(1 / 3),
    });
  };

  if (geometry.index) {
    const idx = geometry.index;
    for (let i = 0; i < idx.count; i += 3) {
      push(idx.getX(i), idx.getX(i + 1), idx.getX(i + 2));
    }
  } else {
    for (let i = 0; i < pos.count; i += 3) {
      push(i, i + 1, i + 2);
    }
  }

  return triangles;
}

function boundsOf(tris: DiamondTriangle[]): THREE.Box3 {
  const box = new THREE.Box3();
  for (const t of tris) {
    box.expandByPoint(t.a);
    box.expandByPoint(t.b);
    box.expandByPoint(t.c);
  }
  return box;
}

function longestAxis(box: THREE.Box3): Axis {
  const size = box.getSize(new THREE.Vector3());
  if (size.x >= size.y && size.x >= size.z) return 'x';
  if (size.y >= size.z) return 'y';
  return 'z';
}

function buildNode(tris: DiamondTriangle[]): BuildNode {
  const bounds = boundsOf(tris);
  if (tris.length <= LEAF_SIZE) {
    return { bounds, isLeaf: true, triangles: tris };
  }

  const axis = longestAxis(bounds);
  const sorted = tris.slice().sort((p, q) => p.centroid[axis] - q.centroid[axis]);
  const mid = Math.floor(sorted.length / 2);

  if (mid === 0 || mid === sorted.length) {
    return { bounds, isLeaf: true, triangles: tris };
  }

  return {
    bounds,
    isLeaf: false,
    left: buildNode(sorted.slice(0, mid)),
    right: buildNode(sorted.slice(mid)),
  };
}

function flatten(node: BuildNode, nodes: FlatBvhNode[], triangles: DiamondTriangle[]): number {
  const index = nodes.length;
  const packed: FlatBvhNode = {
    min: node.bounds.min.clone(),
    max: node.bounds.max.clone(),
    isLeaf: node.isLeaf,
    left: -1,
    right: -1,
    triOffset: 0,
    triCount: 0,
  };
  nodes.push(packed);

  if (node.isLeaf && node.triangles) {
    packed.triOffset = triangles.length;
    packed.triCount = node.triangles.length;
    for (const t of node.triangles) triangles.push(t);
  } else if (node.left && node.right) {
    packed.left = flatten(node.left, nodes, triangles);
    packed.right = flatten(node.right, nodes, triangles);
  }

  return index;
}

/**
 * Build a binary BVH. Returns flat node + triangle arrays for GPU packing.
 */
export function buildBvh(geometry: THREE.BufferGeometry): PackedBvh {
  const soup = extractTriangles(geometry);
  if (soup.length === 0) {
    throw new Error('Diamond geometry has no triangles');
  }

  const root = buildNode(soup);
  const nodes: FlatBvhNode[] = [];
  const triangles: DiamondTriangle[] = [];
  flatten(root, nodes, triangles);

  return {
    nodes,
    triangles,
    triangleCount: triangles.length,
    nodeCount: nodes.length,
  };
}
