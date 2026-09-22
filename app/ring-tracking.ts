import { Matrix4, Quaternion, Vector3, MathUtils } from 'three';
import { MIN_FINGER_EDGE_CONFIDENCE } from './finger-silhouette.ts';
import { Vector2OneEuroFilter, QuaternionOneEuroFilter, OneEuroFilter } from './one-euro-filter.ts';

export type HandPoint = {
  x: number; y: number; z: number;
  worldX?: number; worldY?: number; worldZ?: number;
};
export type ScreenPoint = { x: number; y: number };
export type RingPlacement = ScreenPoint & {
  width: number;
  orientation: Quaternion;
  surfaceFacing: number;
  anchors: ScreenPoint[];
  fingerLength: number;
  handScale: number;
  imageMotion?: { before: ScreenPoint[]; after: ScreenPoint[] };
};

const PALM_ANCHORS = [0, 5, 9, 13, 17];
// Distance must come from the palm: bending a finger is not a camera zoom.
const SCALE_BONES = [[0, 5], [0, 9], [0, 13], [0, 17], [5, 17]];
/** Max stored lateral offset as a fraction of finger width. */
export const MAX_FIT_OFFSET_RATIO = 0.1;
/**
 * Refuse a new size lock, and drop a locked slot, when the silhouette mid
 * drifts farther than this × width. Wide enough that a thumb-side bias on
 * either hand can still lock diameter. Placement stays tighter.
 */
export const MAX_LOCK_OFFSET_RATIO = 0.22;
/** Apply lateral offset in screen placement only when |offset| ≤ this × width. */
export const PLACEMENT_OFFSET_GATE = 0.08;
/** Consecutive off-bone edge frames before dropping a locked fit. */
const FIT_DRIFT_RESET_FRAMES = 5;

export type HandednessLabel = 'Left' | 'Right';

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
};
const alpha = (hz: number, dt: number) => 1 - Math.exp(-2 * Math.PI * hz * dt);

export function swapHandedness(hand: HandednessLabel): HandednessLabel {
  return hand === 'Left' ? 'Right' : 'Left';
}

/**
 * Map MediaPipe's label onto anatomical Left/Right for unmirrored detection.
 * Hands models assume selfie-mirrored input; rear camera frames are not mirrored.
 */
export function anatomicalHandedness(
  raw: string | null | undefined,
  facingMode: 'user' | 'environment',
): HandednessLabel | null {
  if (raw !== 'Left' && raw !== 'Right') return null;
  return facingMode === 'environment' ? swapHandedness(raw) : raw;
}

/**
 * Infer a 2D "right-shaped" silhouette (index left of pinky with radial thumb).
 * Left-dorsal and right-palm share this silhouette, so it is not anatomical
 * handedness by itself — only usable with a dorsal/palm facing estimate.
 */
export function rightShapedPalmSilhouette(landmarks: HandPoint[]): boolean | null {
  if (![0, 4, 5, 17].every((i) => landmarks[i])) return null;
  const wrist = landmarks[0];
  const index = landmarks[5];
  const pinky = landmarks[17];
  const thumb = landmarks[4];
  const cross =
    (index.x - wrist.x) * (pinky.y - wrist.y) -
    (index.y - wrist.y) * (pinky.x - wrist.x);
  if (Math.abs(cross) < 1e-8) return null;
  const midX = (index.x + pinky.x) * 0.5 - wrist.x;
  const midY = (index.y + pinky.y) * 0.5 - wrist.y;
  const thumbX = thumb.x - wrist.x;
  const thumbY = thumb.y - wrist.y;
  const thumbCross = midX * thumbY - midY * thumbX;
  if (Math.abs(thumbCross) < 1e-8) return null;
  // Opposite signs: thumb sits on the radial side of a right-dorsal / left-palm
  // silhouette in an unmirrored camera frame.
  return Math.sign(cross) !== Math.sign(thumbCross);
}

/**
 * @deprecated 2D winding alone cannot tell Left from Right (left-dorsal ≡ right-palm).
 * Prefer {@link anatomicalHandedness}. Kept for tests / callers that only need silhouette shape.
 */
export function handednessFromPalmWinding(
  landmarks: HandPoint[],
): HandednessLabel | null {
  const shaped = rightShapedPalmSilhouette(landmarks);
  if (shaped == null) return null;
  return shaped ? 'Right' : 'Left';
}

/**
 * Calculate 3D anatomical chirality invariant directly from skeletal landmarks.
 * In a human hand, the scalar triple product ((Index - Wrist) x (Pinky - Wrist)) . (Thumb - Wrist)
 * is an invariant anatomical property: it is strictly positive for one hand and negative for the other,
 * and invariant to 3D rotations (pitch, yaw, roll) and distance.
 */
export function anatomicalChiralityFromWorld(
  landmarks: HandPoint[],
): HandednessLabel | null {
  if (!landmarks[0] || !landmarks[5] || !landmarks[17]) return null;

  const w = landmarks[0];
  const i = landmarks[5];
  const p = landmarks[17];

  const hasWorld = (pt: HandPoint) =>
    Number.isFinite(pt.worldX) && Number.isFinite(pt.worldY) && Number.isFinite(pt.worldZ);

  if (!hasWorld(w) || !hasWorld(i) || !hasWorld(p)) return null;

  // Search for active thumb landmark with displacement from wrist
  const thumbIndices = [2, 1, 3, 4];
  let thumbPt: HandPoint | null = null;
  for (const idx of thumbIndices) {
    const pt = landmarks[idx];
    if (pt && hasWorld(pt)) {
      const dSq =
        (pt.worldX! - w.worldX!) ** 2 +
        (pt.worldY! - w.worldY!) ** 2 +
        (pt.worldZ! - w.worldZ!) ** 2;
      if (dSq > 1e-4) {
        thumbPt = pt;
        break;
      }
    }
  }

  if (!thumbPt) return null;

  const ux = i.worldX! - w.worldX!;
  const uy = i.worldY! - w.worldY!;
  const uz = i.worldZ! - w.worldZ!;

  const vx = p.worldX! - w.worldX!;
  const vy = p.worldY! - w.worldY!;
  const vz = p.worldZ! - w.worldZ!;

  const thx = thumbPt.worldX! - w.worldX!;
  const thy = thumbPt.worldY! - w.worldY!;
  const thz = thumbPt.worldZ! - w.worldZ!;

  // Cross product: (u x v)
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;

  // Scalar triple product: (u x v) . th
  const det = nx * thx + ny * thy + nz * thz;
  if (Math.abs(det) < 1e-7) return null;

  // Right hand: radial thumb makes scalar triple product positive in unmirrored world coordinates.
  // Left hand: mirror reflection makes scalar triple product negative.
  return det > 0 ? 'Right' : 'Left';
}

/**
 * Canonicalize MediaPipe's camera-dependent label using 3D anatomical chirality
 * when available, falling back to camera-canonical raw label.
 */
export class HandednessResolver {
  reset() {}

  resolve(
    raw: string | null | undefined,
    facingMode: 'user' | 'environment',
    landmarks?: HandPoint[],
  ): HandednessLabel | null {
    if (landmarks) {
      const chirality = anatomicalChiralityFromWorld(landmarks);
      if (chirality) return chirality;
    }
    return anatomicalHandedness(raw, facingMode);
  }
}

export function detectionSize(width: number, height: number, maximum = 640) {
  const scale = Math.min(1, maximum / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

export function solveRingPose(
  landmarks: HandPoint[],
  screen: ScreenPoint[],
  handedness: 'Left' | 'Right' | null,
  mirrored: boolean,
  previousOrientation?: Quaternion,
  joints: { mcp?: number; pip?: number; finger?: string } = {},
) {
  const mcp = joints.mcp ?? 13;
  const pip = joints.pip ?? 14;
  const isThumb = joints.finger === 'thumb' || mcp === 2;
  const required = [0, 5, 9, 13, 17, mcp, pip];
  if (!required.every((i) => landmarks[i] && screen[i])) return null;
  const hasWorld = required.every((i) =>
    [landmarks[i].worldX, landmarks[i].worldY, landmarks[i].worldZ].every(Number.isFinite));
  const mirror = mirrored ? -1 : 1;
  // The inference results remain unmirrored. Convert to display coordinates
  // exactly once, here. Image XY supplies alignment; world coordinates supply
  // the depth and surface orientation, not a second independently fitted XY.
  const cameraPoint = (i: number) => {
    const p = landmarks[i];
    return hasWorld
      ? new Vector3(p.worldX! * mirror, -p.worldY!, -p.worldZ!)
      : new Vector3(screen[i].x, -screen[i].y, 0);
  };
  const worldFinger = cameraPoint(pip).sub(cameraPoint(mcp));
  const dx = screen[pip].x - screen[mcp].x;
  const dy = screen[mcp].y - screen[pip].y;
  const fingerLength = Math.hypot(dx, dy);
  if (fingerLength < 2 || worldFinger.lengthSq() < 1e-10) return null;
  const index = cameraPoint(5).sub(cameraPoint(0));
  const pinky = cameraPoint(17).sub(cameraPoint(0));
  let normal = new Vector3();
  if (isThumb) {
    // Thumb dorsal normal:
    // Thumb bone runs from MCP(2) to IP(3).
    // Vector from thumb MCP(2) to index MCP(5) spans the palmar web space.
    const thumbBone = cameraPoint(pip).sub(cameraPoint(mcp));
    const web = cameraPoint(5).sub(cameraPoint(mcp));
    normal.crossVectors(web, thumbBone);
    if (normal.lengthSq() < 1e-10) {
      normal.crossVectors(index, pinky);
    }
  } else {
    normal.crossVectors(index, pinky);
  }
  if (normal.lengthSq() < 1e-10) return null;
  normal.normalize();
  // LR1844's stone lies along native -Z. Anatomical Left/Right fixes the sign
  // so both hands share the same mapping: dorsal → head toward camera, palm →
  // rear shank. Choosing a camera-facing sign each frame causes a half-turn
  // jump at an edge-on pose.
  if (handedness) normal.multiplyScalar((handedness === 'Right' ? 1 : -1) * mirror);
  else if (previousOrientation) {
    const previousNormal = new Vector3(0, 0, 1).applyQuaternion(previousOrientation);
    if (normal.dot(previousNormal) < 0) normal.negate();
  }
  let depth = MathUtils.clamp(worldFinger.clone().normalize().z, -0.92, 0.92);
  const facing = Math.abs(normal.z);
  // Palm vs dorsal share the same |facing| for fit lock. Only when the palm
  // faces the camera (normal.z > 0 after handedness) attenuate monocular depth
  // bias so the rear shank stays narrow — dorsal pitch stays free.
  const palmTowardCamera = MathUtils.smoothstep(normal.z, 0.45, 0.9);
  depth *= 1 - 0.55 * palmTowardCamera;
  const projected = Math.sqrt(1 - depth * depth);
  const yAxis = new Vector3(dx / fingerLength * projected, dy / fingerLength * projected, depth);
  normal.addScaledVector(yAxis, -normal.dot(yAxis));
  if (normal.lengthSq() < 1e-8) return null;
  normal.normalize();
  // Basis: Y = finger, Z = palm normal, X = across. Model −Z is the head, so
  // dorsal (normal away from camera) shows the setting; palm shows the shank.
  const xAxis = new Vector3().crossVectors(yAxis, normal).normalize();
  const zAxis = new Vector3().crossVectors(xAxis, yAxis).normalize();
  const orientation = new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(xAxis, yAxis, zAxis));

  // Estimate distance from several projected metric bones. A single screen
  // bone shortens on a tilt, which used to shrink/freeze the ring's diameter.
  const scales: number[] = [];
  if (hasWorld) for (const [a, b] of SCALE_BONES) {
    if (!landmarks[a] || !landmarks[b] || !screen[a] || !screen[b]) continue;
    const bone = cameraPoint(b).sub(cameraPoint(a));
    const projectedLength = Math.hypot(bone.x, bone.y);
    if (projectedLength < bone.length() * 0.45 || projectedLength < 0.008) continue;
    const pixels = Math.hypot(screen[b].x - screen[a].x, screen[b].y - screen[a].y);
    if (pixels > 4) scales.push(pixels / projectedLength * 0.05);
  }
  const handScale = scales.length > 0 ? median(scales) : fingerLength / Math.max(0.4, projected);
  return { orientation, facing, handScale, fingerLength, anchors: PALM_ANCHORS.map(i => screen[i]) };
}

export function projectedFingerAxis(orientation: Quaternion) {
  const axis = new Vector3(0, 1, 0).applyQuaternion(orientation);
  const length = Math.hypot(axis.x, axis.y);
  return { x: axis.x / Math.max(length, 1e-6), y: -axis.y / Math.max(length, 1e-6) };
}

// The 3D model can invent finger pitch/twist even when the image is flat.
// Bound that pose using the same front-facing reference that locked the fit.
// Palm area constrains surface tilt; proximal length constrains bone pitch.
export class RingSurfaceConstraint {
  /**
   * Frontal image lock: palm-area sign turns the surface from dorsal (head /
   * model −Z toward camera) to palm (rear shank; −Z behind the finger).
   * `palmNormalSide` is sign(normal.z) at lock — image facing alone flips it.
   */
  private reference: {
    area: number;
    length: number;
    width: number;
    palmNormalSide: number;
    handedness?: HandednessLabel | null;
  } | null = null;
  private smoothedFacing = 1;
  reset() { this.reference = null; this.smoothedFacing = 1; }
  update(
    orientation: Quaternion,
    anchors: ScreenPoint[],
    fingerLength: number,
    width: number,
    handedness?: HandednessLabel | null,
  ) {
    if (this.reference && handedness && this.reference.handedness && this.reference.handedness !== handedness) {
      // Different hand presented: reset reference so opposite area sign is not treated as palm
      this.reset();
    }
    const normal = new Vector3(0, 0, 1).applyQuaternion(orientation);
    const axis = new Vector3(0, 1, 0).applyQuaternion(orientation);
    if (anchors.length < 5 || !(width > 0) || !(fingerLength > 0)) return orientation.clone();
    const [wrist, index, , , pinky] = anchors;
    const area = (index.x - wrist.x) * (pinky.y - wrist.y) - (index.y - wrist.y) * (pinky.x - wrist.x);
    if (!Number.isFinite(area) || (!this.reference && Math.abs(area) < 1)) return orientation.clone();
    if (!this.reference) {
      this.reference = {
        area,
        length: fingerLength,
        width,
        palmNormalSide: Math.sign(normal.z) || 1,
        handedness: handedness ?? null,
      };
      this.smoothedFacing = MathUtils.clamp(area / this.reference.area, -1, 1);
    }
    const reference = this.reference;
    const zoom = width / reference.width;
    // Image palm-area ratio: +1 same view as lock, −1 opposite (dorsal↔palm).
    const rawFacing = MathUtils.clamp(area / reference.area / (zoom * zoom), -1, 1);
    const facingDiff = Math.abs(rawFacing - this.smoothedFacing);
    const facingAlpha = facingDiff > 0.12 ? 0.5 : 0.2;
    this.smoothedFacing = MathUtils.lerp(this.smoothedFacing, rawFacing, facingAlpha);
    const facing = this.smoothedFacing;
    // Flip palm-normal Z with image facing — never add a π bone twist (that
    // would billboard the head toward the camera on an open palm).
    const palmNormalZ = Math.sign(facing) * reference.palmNormalSide || reference.palmNormalSide;
    // A short dead zone around the frontal reference stops tiny apparent
    // shortening from creating a large square-root depth wobble.
    const projected = MathUtils.clamp(fingerLength / (reference.length * zoom), 0, 1);
    const maximumDepth = projected >= 0.93 ? 0 : Math.sqrt(1 - (projected / 0.93) ** 2);
    const depth = MathUtils.clamp(axis.z, -maximumDepth, maximumDepth);
    const xy = Math.hypot(axis.x, axis.y);
    if (xy < 1e-6) return orientation.clone();
    const projectedAxis = Math.sqrt(1 - depth * depth);
    axis.set(axis.x / xy * projectedAxis, axis.y / xy * projectedAxis, depth);
    const absoluteFacing = Math.abs(facing) >= 0.93 ? 1 : Math.abs(facing) / 0.93;
    const tilt = Math.sqrt(Math.max(0, 1 - absoluteFacing * absoluteFacing));
    const normalXY = Math.hypot(normal.x, normal.y);
    if (normalXY > 1e-6) {
      normal.set(normal.x / normalXY * tilt, normal.y / normalXY * tilt, palmNormalZ * absoluteFacing);
    } else {
      normal.set(0, 0, palmNormalZ * absoluteFacing || reference.palmNormalSide);
    }
    normal.addScaledVector(axis, -normal.dot(axis));
    if (normal.lengthSq() < 1e-8) return orientation.clone();
    normal.normalize();
    const x = new Vector3().crossVectors(axis, normal).normalize();
    return new Quaternion().setFromRotationMatrix(
      new Matrix4().makeBasis(x, axis, new Vector3().crossVectors(x, axis).normalize()),
    );
  }
}

// Least-squares screen similarity from the rigid palm, used to carry the last
// stable ring attachment with the current video frame before filtering noise.
export function fitPalmMotion(before: ScreenPoint[], after: ScreenPoint[]) {
  if (before.length !== after.length || before.length < 3) return null;
  const n = before.length;
  const center = (points: ScreenPoint[]) => points.reduce((a, p) => ({ x: a.x + p.x / n, y: a.y + p.y / n }), { x: 0, y: 0 });
  const p = center(before), q = center(after);
  let denom = 0, real = 0, imag = 0;
  for (let i = 0; i < n; i++) {
    const px = before[i].x - p.x, py = before[i].y - p.y;
    const qx = after[i].x - q.x, qy = after[i].y - q.y;
    denom += px * px + py * py;
    real += px * qx + py * qy;
    imag += px * qy - py * qx;
  }
  if (denom < 1e-12) return null;
  const a = real / denom, b = imag / denom;
  const map = (point: ScreenPoint) => ({
    x: q.x + a * (point.x - p.x) - b * (point.y - p.y),
    y: q.y + b * (point.x - p.x) + a * (point.y - p.y),
  });
  const residual = Math.sqrt(before.reduce((sum, point, i) => {
    const projected = map(point);
    return sum + (projected.x - after[i].x) ** 2 + (projected.y - after[i].y) ** 2;
  }, 0) / n);
  return { map, scale: Math.hypot(a, b), angle: Math.atan2(b, a), residual };
}

/** Ignore target flicker smaller than this once display has settled near it. */
const DISPLAY_WIDTH_DEADBAND = 0.02;
/** Snap the last fraction so ease-in / catch-up finish cleanly. */
const DISPLAY_WIDTH_SETTLE = 0.005;
/** First lock / reacquire: start slightly small then ease up (avoids a hard pop). */
const DISPLAY_WIDTH_EASE_IN = 0.85;

function smoothDisplayWidth(
  previousWidth: number,
  targetWidth: number,
  dt: number,
  previousTarget: number,
): { width: number; target: number } {
  if (!(targetWidth > 0) || !Number.isFinite(targetWidth)) {
    return { width: previousWidth, target: previousTarget };
  }
  if (!(previousWidth > 0) || !Number.isFinite(previousWidth)) {
    return {
      width: targetWidth * DISPLAY_WIDTH_EASE_IN,
      target: targetWidth,
    };
  }
  const stableTarget = previousTarget > 0 ? previousTarget : targetWidth;
  const targetFlicker = Math.abs(targetWidth / Math.max(stableTarget, 1e-6) - 1);
  const displaySettled = Math.abs(previousWidth / Math.max(stableTarget, 1e-6) - 1) < DISPLAY_WIDTH_SETTLE;
  // Once display has settled on a stable target, ignore tiny target noise.
  if (displaySettled && targetFlicker < DISPLAY_WIDTH_DEADBAND) {
    return { width: stableTarget, target: stableTarget };
  }

  const relative = Math.abs(targetWidth / previousWidth - 1);
  if (relative < DISPLAY_WIDTH_SETTLE) {
    return { width: targetWidth, target: targetWidth };
  }
  // Small zoom changes ease gently; large camera-distance jumps catch up faster.
  const hz = relative > 0.08
    ? 12
    : MathUtils.lerp(2, 10, MathUtils.clamp((relative - DISPLAY_WIDTH_SETTLE) / 0.06, 0, 1));
  return {
    width: MathUtils.lerp(previousWidth, targetWidth, alpha(hz, dt)),
    target: targetWidth,
  };
}

export class RingPoseFilter {
  private state: {
    raw: RingPlacement;
    value: RingPlacement;
    time: number;
    angularStep: Vector3;
    localStep: ScreenPoint;
    widthTarget: number;
  } | null = null;
  reset() { this.state = null; }
  current() { return this.state?.value; }
  update(raw: RingPlacement | null, time: number) {
    const previous = this.state;
    if (!raw) {
      if (!previous || time - previous.time > 400) this.reset();
      return this.state?.value ?? null;
    }
    if (!previous || time - previous.time > 150) {
      // Seed below target so first lock eases in instead of popping to full size.
      const seedWidth = Math.max(raw.width, 1e-6) * DISPLAY_WIDTH_EASE_IN;
      const value = { ...raw, width: seedWidth, orientation: raw.orientation.clone() };
      this.state = {
        raw,
        value,
        time,
        angularStep: new Vector3(),
        localStep: { x: 0, y: 0 },
        widthTarget: raw.width,
      };
      return this.state.value;
    }
    const dt = MathUtils.clamp((time - previous.time) / 1000, 0.001, 0.1);
    const usableMotion = (motion: ReturnType<typeof fitPalmMotion>) =>
      motion && motion.residual < Math.max(2, raw.width * 0.09) && motion.scale > 0.7 && motion.scale < 1.4 ? motion : null;
    const imageMotion = raw.imageMotion
      ? usableMotion(fitPalmMotion(raw.imageMotion.before, raw.imageMotion.after)) : null;
    const motion = imageMotion ?? usableMotion(fitPalmMotion(previous.raw.anchors, raw.anchors));
    const reliable = motion !== null;
    const rawTurn = previous.raw.orientation.angleTo(raw.orientation);
    const delta = raw.orientation.clone().multiply(previous.raw.orientation.clone().invert());
    const angularStep = new Vector3(delta.x, delta.y, delta.z).multiplyScalar(delta.w < 0 ? -1 : 1);
    const directionAgreement = angularStep.lengthSq() > 1e-10 && previous.angularStep.lengthSq() > 1e-10
      ? angularStep.clone().normalize().dot(previous.angularStep.clone().normalize()) : 0;
    const previousRaw = motion ? motion.map(previous.raw) : previous.raw;
    const localStep = { x: raw.x - previousRaw.x, y: raw.y - previousRaw.y };
    const stepLength = Math.hypot(localStep.x, localStep.y);
    const previousStepLength = Math.hypot(previous.localStep.x, previous.localStep.y);
    const agreement = stepLength > 0.35 && previousStepLength > 0.35
      ? (localStep.x * previous.localStep.x + localStep.y * previous.localStep.y) / (stepLength * previousStepLength) : 0;
    const carried = motion ? motion.map(previous.value) : previous.value;
    const error = Math.hypot(raw.x - carried.x, raw.y - carried.y);
    // Follow sustained finger motion relative to the palm quickly, while
    // keeping alternating landmark jitter on the slow correction path.
    const fallbackRate = error > raw.width * 0.08 ? 16 : MathUtils.lerp(2.8, 16, Math.max(0, (error - raw.width * 0.03) / (raw.width * 0.05)));
    const targetHz = error > raw.width * 0.15 ? 25 : agreement > 0.6 ? 16 : imageMotion ? 1.2 : reliable ? 2.8 : fallbackRate;
    const localBlend = alpha(targetHz, dt);
    let x = carried.x + (raw.x - carried.x) * localBlend;
    let y = carried.y + (raw.y - carried.y) * localBlend;
    // Continuous subpixel damping eliminates micro-tremor without hard-threshold snapping
    const subpixelStep = Math.hypot(x - previous.value.x, y - previous.value.y);
    if (subpixelStep < 0.7) {
      const damp = MathUtils.smoothstep(subpixelStep, 0.1, 0.7);
      x = previous.value.x + (x - previous.value.x) * damp;
      y = previous.value.y + (y - previous.value.y) * damp;
    }
    const carriedOrientation = previous.value.orientation.clone();
    if (reliable) carriedOrientation.premultiply(new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), -motion.angle));
    const angularError = carriedOrientation.angleTo(raw.orientation);
    // Alternating depth noise used to trigger the fast-turn path every frame.
    // Require directionally consistent rotation, or a persistent new pose,
    // before releasing strong smoothing around the finger's depth axes.
    const moving = (rawTurn > 0.018 && directionAgreement > 0.35)
      || (angularError > 0.18 && rawTurn < 0.018);
    const orientation = carriedOrientation.slerp(raw.orientation, alpha(moving ? 13 : 1.6, dt));
    // Keep the image-space finger direction consistent with the rendered pose.
    // Suppress tiny angular noise to eliminate stationary roll shimmy.
    const rawAxis = projectedFingerAxis(raw.orientation);
    const stableAxis = projectedFingerAxis(orientation);
    const axisError = Math.atan2(rawAxis.y, rawAxis.x) - Math.atan2(stableAxis.y, stableAxis.x);
    const correction = Math.atan2(Math.sin(axisError), Math.cos(axisError));
    const absCorrection = Math.abs(correction);
    const correctionWeight = moving ? 1 : MathUtils.smoothstep(absCorrection, 0.016, 0.05);
    if (correctionWeight > 0.001) {
      const correctionHz = moving ? 18 : MathUtils.lerp(1.2, 3.8, correctionWeight);
      orientation.premultiply(new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), -correction * correctionWeight * alpha(correctionHz, dt)));
    }
    // FingerFitFilter owns the locked diameter target. Smooth only the display
    // width so scale eases with pose instead of stepping while XY lags.
    const smoothed = smoothDisplayWidth(
      previous.value.width,
      raw.width,
      dt,
      previous.widthTarget,
    );
    const value = {
      ...raw,
      x,
      y,
      width: smoothed.width,
      orientation,
      surfaceFacing: Math.abs(new Vector3(0, 0, 1).applyQuaternion(orientation).z),
    };
    this.state = { raw, value, time, angularStep, localStep, widthTarget: smoothed.target };
    return value;
  }
}

// Fit a 2D affine map to a fixed, nearly front-facing palm reference.
// Its largest singular value follows image magnification while allowing the
// other axis to shorten during a turn. No predicted metric bone length enters
// this estimate, and a fixed reference avoids accumulating frame-to-frame drift.
export function palmImageScale(reference: ScreenPoint[], current: ScreenPoint[]) {
  if (reference.length < 4 || current.length !== reference.length
    || [...reference, ...current].some(p => !Number.isFinite(p.x + p.y))) return null;
  const center = (points: ScreenPoint[]) => points.reduce((sum, p) => ({
    x: sum.x + p.x / points.length, y: sum.y + p.y / points.length,
  }), { x: 0, y: 0 });
  const p = center(reference), q = center(current);
  let xx = 0, xy = 0, yy = 0, ux = 0, uy = 0, vx = 0, vy = 0;
  for (let i = 0; i < reference.length; i++) {
    const x = reference[i].x - p.x, y = reference[i].y - p.y;
    const u = current[i].x - q.x, v = current[i].y - q.y;
    xx += x * x; xy += x * y; yy += y * y;
    ux += u * x; uy += u * y; vx += v * x; vy += v * y;
  }
  const determinant = xx * yy - xy * xy;
  if (determinant < 1e-4 * (xx + yy) ** 2) return null;
  const a = (ux * yy - uy * xy) / determinant, b = (uy * xx - ux * xy) / determinant;
  const c = (vx * yy - vy * xy) / determinant, d = (vy * xx - vx * xy) / determinant;
  const trace = a * a + b * b + c * c + d * d;
  const scale = Math.sqrt((trace + Math.sqrt(Math.max(0, trace * trace - 4 * (a * d - b * c) ** 2))) / 2);
  let error = 0;
  for (let i = 0; i < reference.length; i++) {
    const x = reference[i].x - p.x, y = reference[i].y - p.y;
    error += (q.x + a * x + b * y - current[i].x) ** 2 + (q.y + c * x + d * y - current[i].y) ** 2;
  }
  const relativeError = Math.sqrt(error / (xx + yy)) / Math.max(scale, 1e-6);
  return scale > 0.25 && scale < 4 && relativeError < 0.035 ? scale : null;
}

class PalmScaleLock {
  private reference: ScreenPoint[] | null = null;
  private referenceScale = 0;
  private scale = 1;
  private observations: number[] = [];
  reset() { this.reference = null; this.referenceScale = 0; this.scale = 1; this.observations = []; }
  update(anchors: ScreenPoint[], initialScale: number, canInitialize: boolean) {
    if (!this.reference) {
      if (!canInitialize || palmImageScale(anchors, anchors) === null) return null;
      this.reference = anchors.map(p => ({ ...p }));
      this.referenceScale = initialScale;
    }
    const measured = palmImageScale(this.reference, anchors);
    if (measured === null) {
      this.observations = [];
      return null;
    }
    this.observations.push(measured);
    if (this.observations.length > 5) this.observations.shift();
    // Require a majority of recent frames; small size noise is held exactly.
    if (this.observations.length === 5) {
      const target = median(this.observations);
      const relative = Math.abs(target / this.scale - 1);
      if (relative > 0.025) {
        // Sustained camera-distance steps catch up faster; tiny noise stays tight.
        const step = relative > 0.08 ? { lo: 0.88, hi: 1.12 } : { lo: 0.94, hi: 1.06 };
        this.scale *= MathUtils.clamp(target / this.scale, step.lo, step.hi);
      }
    }
    return this.referenceScale * this.scale;
  }
}

/** Post-lock: edges within this relative band may update offset / drift; ratio stays frozen. */
const POST_LOCK_WIDTH_RECONFIRM = 0.2;
/** Distinct edge revisions must arrive within this many ms to confirm. */
const CONFIRM_REVISION_GAP_MS = 180;
const CONFIRM_FRAMES = 3;
/** After a finger or hand change: fewer frames, and a gap that still chains on a slow camera. */
const RELOCK_CONFIRM_FRAMES = 2;
const RELOCK_REVISION_GAP_MS = 480;
const RELOCK_RATIO_BAND = 0.08;
const RELOCK_MIN_FACING = 0.65;
/** Agreeing frames before a Left/Right label change switches slots. */
const HAND_SLOT_SWAP_FRAMES = 2;

type FitEvidenceStatus = 'waiting' | 'uncertain' | 'side view' | 'confirming' | 'accepted';

type FitSlot = {
  distance: PalmScaleLock;
  ratio: number | null;
  width: number;
  candidateRatio: number;
  candidateOffset: number;
  candidateFrames: number;
  offset: number;
  revision: number;
  candidateRevision: number;
  driftFrames: number;
  lastHandScale: number;
  reacquire: boolean;
  previewWidth: number;
  previewOffset: number;
  evidenceStatus: FitEvidenceStatus;
};

function blankSlot(reacquire: boolean): FitSlot {
  return {
    distance: new PalmScaleLock(),
    ratio: null,
    width: 0,
    candidateRatio: 0,
    candidateOffset: 0,
    candidateFrames: 0,
    offset: 0,
    revision: -1,
    candidateRevision: -Infinity,
    driftFrames: 0,
    lastHandScale: 0,
    reacquire,
    previewWidth: 0,
    previewOffset: 0,
    evidenceStatus: 'waiting',
  };
}

function slotKey(finger?: string | null, hand?: HandednessLabel | null) {
  return `${hand ?? ''}:${finger ?? ''}`;
}

export class FingerFitFilter {
  private slots = new Map<string, FitSlot>();
  private activeKey: string | null = null;
  private distance = new PalmScaleLock();
  private ratio: number | null = null;
  private width = 0;
  private candidateRatio = 0;
  private candidateOffset = 0;
  private candidateFrames = 0;
  private offset = 0;
  private revision = -1;
  private candidateRevision = -Infinity;
  private driftFrames = 0;
  private lastHandScale = 0;
  private reacquire = false;
  private previewWidth = 0;
  private previewOffset = 0;
  private seenFinger: string | null = null;
  private seenHand: HandednessLabel | null = null;
  private pendingHand: HandednessLabel | null = null;
  private handSwapFrames = 0;
  private evidenceStatus: FitEvidenceStatus = 'waiting';
  get status() { return this.evidenceStatus; }
  get confirmationFrames() {
    return Math.min(this.confirmationNeeded, this.candidateFrames);
  }
  get confirmationNeeded() {
    return this.reacquire ? RELOCK_CONFIRM_FRAMES : CONFIRM_FRAMES;
  }
  /** True while a finger/hand change is measuring a new diameter. */
  get relocking() { return this.reacquire && this.ratio === null; }
  get calibrated() { return this.ratio !== null; }
  /** Anatomical hand that owns the active slot, once a hand has been selected. */
  get lockedHand() { return this.seenHand; }
  get lockedFinger() { return this.seenFinger; }
  reset() {
    this.slots.clear();
    this.activeKey = null;
    this.distance = new PalmScaleLock();
    this.ratio = null;
    this.width = 0;
    this.candidateRatio = 0;
    this.candidateOffset = 0;
    this.candidateFrames = 0;
    this.offset = 0;
    this.revision = -1;
    this.candidateRevision = -Infinity;
    this.driftFrames = 0;
    this.lastHandScale = 0;
    this.reacquire = false;
    this.previewWidth = 0;
    this.previewOffset = 0;
    this.seenFinger = null;
    this.seenHand = null;
    this.pendingHand = null;
    this.handSwapFrames = 0;
    this.evidenceStatus = 'waiting';
  }

  /**
   * Drop the active slot's locked diameter and measure it again.
   * Other hand/finger slots stay cached. A hand-scoped relock also clears
   * that slot's palm-distance reference.
   */
  relock(scope: 'finger' | 'hand') {
    this.clearCalibration();
    this.previewWidth = 0;
    this.previewOffset = 0;
    this.reacquire = true;
    this.evidenceStatus = 'waiting';
    if (scope === 'hand') {
      this.distance.reset();
      this.lastHandScale = 0;
      this.handSwapFrames = 0;
      this.pendingHand = null;
    }
    this.storeActive();
  }

  /**
   * Switch to the slot for this hand and finger.
   * A known lock is restored on this call. A new slot measures on the fast path.
   */
  select(finger?: string | null, hand?: HandednessLabel | null) {
    const nextFinger = finger ?? this.seenFinger;
    const nextHand = hand ?? this.seenHand;
    if (!nextFinger && !nextHand) return;
    const key = slotKey(nextFinger, nextHand);
    this.seenFinger = nextFinger;
    this.seenHand = nextHand;
    this.handSwapFrames = 0;
    this.pendingHand = null;
    if (key === this.activeKey) return;
    this.storeActive();
    let slot = this.slots.get(key);
    if (!slot) {
      slot = blankSlot(true);
      this.slots.set(key, slot);
    }
    this.apply(slot);
    this.activeKey = key;
    if (this.ratio !== null) {
      this.reacquire = false;
      this.previewWidth = 0;
      this.previewOffset = 0;
      this.evidenceStatus = 'accepted';
    }
  }

  private capture(): FitSlot {
    return {
      distance: this.distance,
      ratio: this.ratio,
      width: this.width,
      candidateRatio: this.candidateRatio,
      candidateOffset: this.candidateOffset,
      candidateFrames: this.candidateFrames,
      offset: this.offset,
      revision: this.revision,
      candidateRevision: this.candidateRevision,
      driftFrames: this.driftFrames,
      lastHandScale: this.lastHandScale,
      reacquire: this.reacquire,
      previewWidth: this.previewWidth,
      previewOffset: this.previewOffset,
      evidenceStatus: this.evidenceStatus,
    };
  }

  private apply(slot: FitSlot) {
    this.distance = slot.distance;
    this.ratio = slot.ratio;
    this.width = slot.width;
    this.candidateRatio = slot.candidateRatio;
    this.candidateOffset = slot.candidateOffset;
    this.candidateFrames = slot.candidateFrames;
    this.offset = slot.offset;
    this.revision = slot.revision;
    this.candidateRevision = slot.candidateRevision;
    this.driftFrames = slot.driftFrames;
    this.lastHandScale = slot.lastHandScale;
    this.reacquire = slot.reacquire;
    this.previewWidth = slot.previewWidth;
    this.previewOffset = slot.previewOffset;
    this.evidenceStatus = slot.evidenceStatus;
  }

  private storeActive() {
    if (!this.activeKey) return;
    this.slots.set(this.activeKey, this.capture());
  }

  private observeTarget(finger?: string, hand?: HandednessLabel | null) {
    if (finger && this.seenFinger && finger !== this.seenFinger) {
      this.select(finger, hand ?? this.seenHand);
      return;
    }

    if (hand && this.seenHand && hand !== this.seenHand) {
      // One-frame label flicker stays on the current slot. A second agreeing
      // frame switches slots and keeps the previous hand's lock.
      if (this.ratio !== null || this.reacquire) {
        if (this.pendingHand !== hand) {
          this.pendingHand = hand;
          this.handSwapFrames = 1;
        } else {
          this.handSwapFrames += 1;
          if (this.handSwapFrames >= HAND_SLOT_SWAP_FRAMES) {
            this.select(finger ?? this.seenFinger, hand);
          }
        }
      } else {
        this.select(finger ?? this.seenFinger, hand);
      }
      return;
    }

    if (!this.activeKey && (finger || hand)) {
      this.select(finger ?? null, hand ?? null);
      return;
    }

    if (finger) this.seenFinger = finger;
    if (hand) {
      this.seenHand = hand;
      this.handSwapFrames = 0;
      this.pendingHand = null;
    }
    if ((finger || hand) && this.activeKey !== slotKey(this.seenFinger, this.seenHand)) {
      this.select(this.seenFinger, this.seenHand);
    }
  }

  private resolveHandScale(
    handScale: number,
    facing: number,
    edge: { confidence: number } | undefined,
    anchors: ScreenPoint[] | undefined,
  ): number | null {
    if (!anchors) {
      this.lastHandScale = handScale;
      return handScale;
    }
    const minInitFacing = this.reacquire ? RELOCK_MIN_FACING : 0.75;
    const canInit = facing >= minInitFacing
      && !!edge
      && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE;
    const distance = this.distance.update(anchors, handScale, canInit);
    if (distance === null) {
      // After lock, keep the last good distance so side/palm views still show
      // the band instead of dropping the ring for one bad palm frame.
      if (this.ratio !== null && this.lastHandScale > 0) {
        this.evidenceStatus = 'side view';
        return this.lastHandScale;
      }
      this.evidenceStatus = 'side view';
      return null;
    }
    this.lastHandScale = distance;
    return distance;
  }

  private clearCalibration() {
    this.ratio = null;
    this.width = 0;
    this.offset = 0;
    this.candidateFrames = 0;
    this.candidateRatio = 0;
    this.candidateOffset = 0;
    this.candidateRevision = -Infinity;
    this.driftFrames = 0;
    this.previewWidth = 0;
    this.previewOffset = 0;
    this.evidenceStatus = 'uncertain';
  }

  private ingestEdge(
    handScale: number,
    facing: number,
    edge: { width: number; offset: number; revision: number; confidence: number },
    hasAnchors: boolean,
  ) {
    if (edge.revision === this.revision) return;
    const elapsed = MathUtils.clamp((edge.revision - this.revision) / 1000, 0.001, 0.1);
    this.revision = edge.revision;
    const ratio = edge.width / handScale;
    const absOffsetRatio = Math.abs(edge.offset) / Math.max(edge.width, 1e-6);
    const offsetCentered = absOffsetRatio <= MAX_LOCK_OFFSET_RATIO;
    const maxOffset = edge.width * MAX_FIT_OFFSET_RATIO;
    const clampedOffset = MathUtils.clamp(edge.offset, -maxOffset, maxOffset);
    const widthTrustworthy = Number.isFinite(edge.confidence)
      && edge.confidence >= MIN_FINGER_EDGE_CONFIDENCE
      && Number.isFinite(edge.revision)
      && ratio > 0.1
      && ratio < 1.2
      && (this.ratio === null
        || Math.abs(edge.width / (this.ratio * handScale) - 1) < POST_LOCK_WIDTH_RECONFIRM);

    // After lock, sustained off-bone edges drop calibration instead of drifting.
    if (this.ratio !== null && widthTrustworthy && !offsetCentered) {
      this.driftFrames += 1;
      if (this.driftFrames >= FIT_DRIFT_RESET_FRAMES) {
        this.clearCalibration();
        this.reacquire = true;
        this.evidenceStatus = 'waiting';
        this.storeActive();
        return;
      }
    } else if (offsetCentered) {
      this.driftFrames = 0;
    }

    // Same front-facing gate for palm or dorsal (|normal.z|). Init with
    // anchors needs a clearer front; post-lock and relock accept a slightly lower facing.
    const minFacing = this.ratio === null && hasAnchors && !this.reacquire ? 0.75 : 0.65;
    if (!widthTrustworthy || !offsetCentered || facing < minFacing) {
      this.candidateFrames = 0;
      this.evidenceStatus = !widthTrustworthy || !offsetCentered ? 'uncertain' : 'side view';
      return;
    }

    // Confirm distinct, nearby observations in hand-relative units so
    // approaching the camera does not look like a change of finger shape.
    const framesNeeded = this.reacquire ? RELOCK_CONFIRM_FRAMES : CONFIRM_FRAMES;
    const gapMs = this.reacquire ? RELOCK_REVISION_GAP_MS : CONFIRM_REVISION_GAP_MS;
    const ratioBand = this.reacquire ? RELOCK_RATIO_BAND : 0.05;
    const offsetNorm = clampedOffset / handScale;
    const consistent = edge.revision > this.candidateRevision
      && edge.revision - this.candidateRevision <= gapMs
      && Math.abs(ratio / Math.max(this.candidateRatio, 1e-6) - 1) < ratioBand
      && Math.abs(offsetNorm - this.candidateOffset) < ratio * (this.reacquire ? 0.28 : 0.15);
    this.candidateFrames = consistent ? this.candidateFrames + 1 : 1;
    this.candidateRatio = ratio;
    this.candidateOffset = offsetNorm;
    this.candidateRevision = edge.revision;
    this.evidenceStatus = this.candidateFrames >= framesNeeded ? 'accepted' : 'confirming';
    if (this.reacquire && this.ratio === null) {
      // Show the new finger's measured span immediately; freeze it once confirmed.
      this.previewWidth = edge.width;
      this.previewOffset = clampedOffset;
    }
    if (this.candidateFrames < framesNeeded) return;

    if (this.ratio === null) {
      // Lock edge-to-edge skin span on the first confirmed clear sequence.
      this.width = edge.width;
      this.offset = offsetNorm;
      this.ratio = this.width / handScale;
      this.reacquire = false;
      this.previewWidth = 0;
      this.previewOffset = 0;
      return;
    }

    // Post-lock: ratio stays frozen (display size still tracks palm distance).
    // Only recenter laterally on clear, nearby edges.
    const offsetDiff = Math.abs(offsetNorm - this.offset);
    if (offsetDiff > 0.018) {
      this.offset = MathUtils.lerp(
        this.offset,
        offsetNorm,
        alpha(offsetDiff > 0.04 ? 5 : 3, elapsed),
      );
    }
  }

  update(
    handScale: number,
    facing: number,
    edge?: { width: number; offset: number; revision: number; confidence: number },
    anchors?: ScreenPoint[],
    target?: { finger?: string; hand?: HandednessLabel | null },
  ) {
    if (target) this.observeTarget(target.finger, target.hand);
    if (!(handScale > 0) || !Number.isFinite(handScale)) return null;
    const resolved = this.resolveHandScale(handScale, facing, edge, anchors);
    if (resolved === null) return null;
    handScale = resolved;

    if (edge) this.ingestEdge(handScale, facing, edge, !!anchors);

    // Uncertain/missing edges may not change width calibration or lateral
    // attachment. Only the image-based distance estimate changes display size.
    // Once calibrated, keep returning a size through side and palm turns so
    // the ring can show band profile / rear shank instead of disappearing.
    if (this.ratio === null) {
      if (this.reacquire && this.previewWidth > 0) {
        return { width: this.previewWidth, offset: this.previewOffset };
      }
      return null;
    }
    this.width = this.ratio * handScale;
    return { width: this.width, offset: this.offset * handScale };
  }
}
