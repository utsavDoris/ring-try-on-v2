// GPU BVH for cut-gem meshes. Fast front-to-back traversal with early AABB culling.
export const BVH_GLSL = `
const int BVH_STACK_SIZE = 16;
const int BVH_STEPS = 48;
const int MAX_LEAF_TRIS = 4;

vec4 fetchIndex(sampler2D tex, vec2 texSize, int index) {
  int w = int(texSize.x);
  int h = int(texSize.y);
  int maxIndex = max(0, w * h - 1);
  int clampedIndex = clamp(index, 0, maxIndex);
  int x = clampedIndex - (clampedIndex / w) * w;
  int y = clampedIndex / w;
  return texelFetch(tex, ivec2(x, y), 0);
}

void fetchNode(
  sampler2D nodes,
  vec2 nodeSize,
  int nodeIndex,
  out vec3 bmin,
  out vec3 bmax,
  out int left,
  out int right
) {
  int base = nodeIndex * 2;
  vec4 a = fetchIndex(nodes, nodeSize, base);
  vec4 b = fetchIndex(nodes, nodeSize, base + 1);
  bmin = a.xyz;
  bmax = b.xyz;
  left = int(a.w);
  right = int(b.w);
}

void fetchTriangle(sampler2D tris, vec2 triSize, int triIndex, out vec3 v0, out vec3 v1, out vec3 v2) {
  int base = triIndex * 3;
  v0 = fetchIndex(tris, triSize, base).xyz;
  v1 = fetchIndex(tris, triSize, base + 1).xyz;
  v2 = fetchIndex(tris, triSize, base + 2).xyz;
}

bool intersectAABB(vec3 orig, vec3 invDir, vec3 bmin, vec3 bmax, float tMax) {
  vec3 t0 = (bmin - orig) * invDir;
  vec3 t1 = (bmax - orig) * invDir;
  vec3 tmin3 = min(t0, t1);
  vec3 tmax3 = max(t0, t1);
  float tNear = max(max(tmin3.x, tmin3.y), tmin3.z);
  float tFar = min(min(tmax3.x, tmax3.y), tmax3.z);
  return tFar >= max(tNear, 0.0) && tNear < tMax && tFar >= 0.0;
}

bool intersectTriangle(
  vec3 orig,
  vec3 dir,
  vec3 v0,
  vec3 v1,
  vec3 v2,
  float tMin,
  inout float tHit,
  out vec3 n
) {
  vec3 e1 = v1 - v0;
  vec3 e2 = v2 - v0;
  vec3 pvec = cross(dir, e2);
  float det = dot(e1, pvec);
  if (abs(det) < 1e-8) {
    n = vec3(0.0);
    return false;
  }
  float invDet = 1.0 / det;
  vec3 tvec = orig - v0;
  float u = dot(tvec, pvec) * invDet;
  if (u < 0.0 || u > 1.0) {
    n = vec3(0.0);
    return false;
  }
  vec3 qvec = cross(tvec, e1);
  float v = dot(dir, qvec) * invDet;
  if (v < 0.0 || u + v > 1.0) {
    n = vec3(0.0);
    return false;
  }
  float t = dot(e2, qvec) * invDet;
  if (t <= tMin || t >= tHit) {
    n = vec3(0.0);
    return false;
  }
  tHit = t;
  n = normalize(cross(e1, e2));
  return true;
}

bool intersectBvh(
  sampler2D nodes,
  vec2 nodeSize,
  sampler2D tris,
  vec2 triSize,
  vec3 orig,
  vec3 dir,
  float tMin,
  out float tHit,
  out vec3 hitN
) {
  // Safe inverse ray direction with zero division protection
  vec3 invDir = 1.0 / (dir + sign(dir + vec3(1e-16)) * 1e-10);
  int stack[BVH_STACK_SIZE];
  int sp = 0;
  stack[0] = 0;
  tHit = 1e20;
  hitN = vec3(0.0);
  bool hit = false;

  for (int step = 0; step < BVH_STEPS; step++) {
    if (sp < 0) break;
    int ni = stack[sp];
    sp--;

    vec3 bmin;
    vec3 bmax;
    int left;
    int right;
    fetchNode(nodes, nodeSize, ni, bmin, bmax, left, right);

    if (!intersectAABB(orig, invDir, bmin, bmax, tHit)) {
      continue;
    }

    if (left < 0) {
      int triOffset = -left - 1;
      int triCount = min(right, MAX_LEAF_TRIS);
      for (int t = 0; t < MAX_LEAF_TRIS; t++) {
        if (t >= triCount) break;
        vec3 v0;
        vec3 v1;
        vec3 v2;
        fetchTriangle(tris, triSize, triOffset + t, v0, v1, v2);
        vec3 n;
        if (intersectTriangle(orig, dir, v0, v1, v2, tMin, tHit, n)) {
          hit = true;
          hitN = n;
        }
      }
    } else {
      vec3 center = (bmin + bmax) * 0.5;
      int nearChild = left;
      int farChild = right;
      if (dot(dir, center - orig) <= 0.0) {
        nearChild = right;
        farChild = left;
      }
      if (sp + 1 < BVH_STACK_SIZE) {
        sp++; stack[sp] = farChild;
      }
      if (sp + 1 < BVH_STACK_SIZE) {
        sp++; stack[sp] = nearChild;
      }
    }
  }

  return hit;
}
`;
