/**
 * Take Photo pose gate — scores a live hand for auto-capture,
 * waits for a full open stable track, then freezes the video frame.
 * Adapted from v1 for the Next.js MediaPipe + FingerFitFilter stack.
 */

import type { Quaternion } from 'three';
import type { HandPoint, RingPlacement, ScreenPoint } from './ring-tracking';

export const HOLD_MS = 1800;
export const ARM_MS = 220;
/** Brief pose/motion drops do not wipe the hold bar. */
export const DROP_GRACE_MS = 480;
/** Normalized landmark travel per frame — ignore millimetre-level shake. */
export const CAPTURE_MOTION_MAX = 0.022;
export const SETTLE_FRAMES = 10;
export const SETTLE_MAX_MS = 2500;

export const POSE_HINTS = {
  noHand: 'Show your hand',
  fingers: 'Open all fingers',
  size: 'Move your hand closer',
  tooClose: 'Move your hand back',
  center: 'Center your full hand',
  partial: 'Show your full hand',
  track: 'Keep hand fully in view',
  ready: 'Capturing…',
  ringFit: 'Finger tracking…',
  manual: 'Tap Capture to take photo',
} as const;

export const DEFAULT_CAPTURE_HINT = POSE_HINTS.noHand;
export const MANUAL_CAPTURE_HINT = POSE_HINTS.manual;

const MAX_FLEXION = 1.12;
const MIN_CURL = 1.22;
const EDGE_MARGIN = 0.03;
const MIN_HAND_SPAN = 0.26;
const MAX_HAND_SPAN = 0.94;

const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_PIP = 6;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const MIDDLE_PIP = 10;
const MIDDLE_TIP = 12;
const RING_MCP = 13;
const RING_PIP = 14;
const RING_TIP = 16;
const PINKY_MCP = 17;
const PINKY_PIP = 18;
const PINKY_TIP = 20;

const CHECK_FINGERS = [
  { mcp: INDEX_MCP, pip: INDEX_PIP, tip: INDEX_TIP },
  { mcp: MIDDLE_MCP, pip: MIDDLE_PIP, tip: MIDDLE_TIP },
  { mcp: RING_MCP, pip: RING_PIP, tip: RING_TIP },
  { mcp: PINKY_MCP, pip: PINKY_PIP, tip: PINKY_TIP },
] as const;

export type CapturePoseEvaluation = {
  ok: boolean;
  reason: string;
  hint: string;
  guideScore: number;
};

export type CaptureGestureInfo = {
  isFist: boolean;
  isPinch: boolean;
  isDorsal: boolean;
  flexion: Record<string, number>;
  curlRatio: Record<string, number>;
};

export type PhotoSnapshot = {
  landmarks: HandPoint[];
  placement: RingPlacement | null;
  handedness: 'Left' | 'Right' | null;
  isMirrored: boolean;
  videoDimensions: { width: number; height: number };
  fitCalibrated: boolean;
};

function clonePoint(pt: HandPoint | null | undefined): HandPoint | null {
  if (!pt) return null;
  return {
    x: pt.x,
    y: pt.y,
    z: pt.z,
    worldX: pt.worldX,
    worldY: pt.worldY,
    worldZ: pt.worldZ,
  };
}

function cloneLandmarkList(list: HandPoint[] | null | undefined): HandPoint[] | null {
  if (!list) return null;
  return list.map((pt) => clonePoint(pt)!);
}

function isFinitePoint(pt: { x: number; y: number } | null | undefined): boolean {
  return !!pt && Number.isFinite(pt.x) && Number.isFinite(pt.y);
}

function hypot2(ax: number, ay: number, bx: number, by: number) {
  return Math.hypot(bx - ax, by - ay);
}

/**
 * True when every landmark is present and the hand sits fully in frame.
 */
export function evaluateFullHandTrack(
  landmarks: HandPoint[] | null | undefined,
): CapturePoseEvaluation & { span?: number } {
  if (!landmarks || landmarks.length < 21) {
    return { ok: false, reason: 'noHand', hint: POSE_HINTS.noHand, guideScore: 0 };
  }

  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;

  for (let i = 0; i < 21; i += 1) {
    const pt = landmarks[i];
    if (!isFinitePoint(pt)) {
      return { ok: false, reason: 'partial', hint: POSE_HINTS.partial, guideScore: 0 };
    }
    minX = Math.min(minX, pt.x);
    minY = Math.min(minY, pt.y);
    maxX = Math.max(maxX, pt.x);
    maxY = Math.max(maxY, pt.y);
  }

  if (
    minX < EDGE_MARGIN ||
    minY < EDGE_MARGIN ||
    maxX > 1 - EDGE_MARGIN ||
    maxY > 1 - EDGE_MARGIN
  ) {
    return { ok: false, reason: 'partial', hint: POSE_HINTS.partial, guideScore: 0.2 };
  }

  const wrist = landmarks[WRIST];
  const middleTip = landmarks[MIDDLE_TIP];
  const span = Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y);
  if (span < MIN_HAND_SPAN) {
    return { ok: false, reason: 'size', hint: POSE_HINTS.size, guideScore: 0.35 };
  }
  if (span > MAX_HAND_SPAN) {
    return { ok: false, reason: 'tooClose', hint: POSE_HINTS.tooClose, guideScore: 0.35 };
  }

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  if (cx < 0.22 || cx > 0.78 || cy < 0.2 || cy > 0.82) {
    return { ok: false, reason: 'center', hint: POSE_HINTS.center, guideScore: 0.4 };
  }

  return { ok: true, reason: 'track', hint: POSE_HINTS.track, guideScore: 0.6, span };
}

/** Lightweight open-hand / fist / pinch / curl from MediaPipe landmarks. */
export function evaluateOpenHandGesture(
  landmarks: HandPoint[] | null | undefined,
): CaptureGestureInfo {
  const empty: CaptureGestureInfo = {
    isFist: true,
    isPinch: false,
    isDorsal: false,
    flexion: {},
    curlRatio: {},
  };
  if (!landmarks || landmarks.length < 21) return empty;

  const curlRatio: Record<string, number> = {};
  const flexion: Record<string, number> = {};
  const names = ['index', 'middle', 'ring', 'pinky'] as const;

  for (let i = 0; i < CHECK_FINGERS.length; i += 1) {
    const { mcp, pip, tip } = CHECK_FINGERS[i];
    const a = landmarks[mcp];
    const b = landmarks[pip];
    const c = landmarks[tip];
    const dMcpTip = hypot2(a.x, a.y, c.x, c.y);
    const dMcpPip = hypot2(a.x, a.y, b.x, b.y) || 1e-4;
    curlRatio[names[i]] = dMcpTip / dMcpPip;

    const v1x = a.x - b.x;
    const v1y = a.y - b.y;
    const v1z = (a.z || 0) - (b.z || 0);
    const v2x = c.x - b.x;
    const v2y = c.y - b.y;
    const v2z = (c.z || 0) - (b.z || 0);
    const dot = v1x * v2x + v1y * v2y + v1z * v2z;
    const l1 = Math.hypot(v1x, v1y, v1z) || 1e-4;
    const l2 = Math.hypot(v2x, v2y, v2z) || 1e-4;
    const cosAngle = Math.max(-1, Math.min(1, dot / (l1 * l2)));
    flexion[names[i]] = Math.PI - Math.acos(cosAngle);
  }

  const ext = names.map((n) => curlRatio[n] > 1.35);
  const extCount = ext.filter(Boolean).length;
  const thumb = landmarks[THUMB_TIP];
  const indexTip = landmarks[INDEX_TIP];
  const indexMcp = landmarks[INDEX_MCP];
  const pinkyMcp = landmarks[PINKY_MCP];
  const pinchDist = hypot2(thumb.x, thumb.y, indexTip.x, indexTip.y);
  const handScale =
    hypot2(indexMcp.x, indexMcp.y, pinkyMcp.x, pinkyMcp.y) || 0.1;
  const isPinch = pinchDist / handScale < 0.35 && extCount >= 1;
  const isFist = extCount <= 1 && !isPinch;

  return { isFist, isPinch, isDorsal: false, flexion, curlRatio };
}

/** Average normalized landmark travel between two frames. */
export function landmarkMotion(
  previous: HandPoint[] | null | undefined,
  current: HandPoint[] | null | undefined,
): number {
  if (!previous || !current || previous.length < 21 || current.length < 21) {
    return 0;
  }
  let sum = 0;
  let count = 0;
  for (let i = 0; i < 21; i += 1) {
    const a = previous[i];
    const b = current[i];
    if (!isFinitePoint(a) || !isFinitePoint(b)) continue;
    sum += hypot2(a.x, a.y, b.x, b.y);
    count += 1;
  }
  return count > 0 ? sum / count : 0;
}

export function hasProperRingFit(options: {
  fitCalibrated?: boolean;
  placement?: RingPlacement | null;
  width?: number;
}): boolean {
  if (options.fitCalibrated === true) {
    const width = options.placement?.width ?? options.width ?? 0;
    return width > 0;
  }
  const width = options.placement?.width ?? options.width ?? 0;
  return width > 0 && options.fitCalibrated !== false;
}

/**
 * Capture only for a full open hand that is stably tracked with a locked finger fit.
 */
export function evaluateCapturePose(options: {
  hasHand: boolean;
  landmarks: HandPoint[] | null | undefined;
  gestureInfo?: CaptureGestureInfo | null;
  motion?: number;
  fitCalibrated?: boolean;
  placement?: RingPlacement | null;
}): CapturePoseEvaluation {
  const {
    hasHand,
    landmarks,
    gestureInfo: providedGesture,
    motion = 0,
    fitCalibrated,
    placement,
  } = options;

  if (!hasHand || !landmarks || landmarks.length < 21) {
    return { ok: false, reason: 'noHand', hint: POSE_HINTS.noHand, guideScore: 0 };
  }

  const full = evaluateFullHandTrack(landmarks);
  if (!full.ok) {
    return { ok: false, reason: full.reason, hint: full.hint, guideScore: full.guideScore };
  }

  const gestureInfo = providedGesture || evaluateOpenHandGesture(landmarks);
  if (gestureInfo.isFist || gestureInfo.isPinch) {
    return { ok: false, reason: 'fingers', hint: POSE_HINTS.fingers, guideScore: 0.5 };
  }

  const minCurl = gestureInfo.isDorsal ? MIN_CURL * 0.92 : MIN_CURL;
  const tooCurled = (['index', 'middle', 'ring', 'pinky'] as const).some((finger) => {
    return (
      (gestureInfo.flexion[finger] || 0) > MAX_FLEXION ||
      (typeof gestureInfo.curlRatio[finger] === 'number' &&
        gestureInfo.curlRatio[finger] < minCurl)
    );
  });
  if (tooCurled) {
    return { ok: false, reason: 'fingers', hint: POSE_HINTS.fingers, guideScore: 0.55 };
  }

  if (!hasProperRingFit({ fitCalibrated, placement })) {
    return { ok: false, reason: 'ringFit', hint: POSE_HINTS.ringFit, guideScore: 0.75 };
  }

  const travel = Number.isFinite(motion) ? motion : 0;
  if (travel > CAPTURE_MOTION_MAX) {
    return { ok: false, reason: 'track', hint: POSE_HINTS.track, guideScore: 0.9 };
  }

  return { ok: true, reason: 'ready', hint: POSE_HINTS.ready, guideScore: 1 };
}

/** Manual shutter: hand present + locked finger fit (no hold / full pose center gates). */
export function evaluateManualCaptureGate(options: {
  hasHand: boolean;
  fitCalibrated?: boolean;
  placement?: RingPlacement | null;
}): CapturePoseEvaluation {
  if (!options.hasHand) {
    return { ok: false, reason: 'noHand', hint: POSE_HINTS.noHand, guideScore: 0 };
  }
  if (!hasProperRingFit(options)) {
    return { ok: false, reason: 'ringFit', hint: POSE_HINTS.ringFit, guideScore: 0.75 };
  }
  return { ok: true, reason: 'ready', hint: MANUAL_CAPTURE_HINT, guideScore: 1 };
}

/**
 * Accumulates hold time only while the pose stays valid.
 * Brief dropouts pause the bar instead of resetting it.
 */
export class PoseHoldTimer {
  private _goodMs = 0;
  private _armedMs = 0;
  private _badMs = 0;
  private _last = 0;
  holdMs: number;
  armMs: number;
  dropGraceMs: number;

  constructor(
    holdMs = HOLD_MS,
    armMs = ARM_MS,
    dropGraceMs = DROP_GRACE_MS,
  ) {
    this.holdMs = holdMs;
    this.armMs = armMs;
    this.dropGraceMs = dropGraceMs;
  }

  reset() {
    this._goodMs = 0;
    this._armedMs = 0;
    this._badMs = 0;
    this._last = 0;
  }

  update(ok: boolean, now = performance.now()) {
    if (!this._last) this._last = now;
    const dt = Math.min(100, Math.max(0, now - this._last));
    this._last = now;

    if (ok) {
      this._badMs = 0;
      if (this._armedMs < this.armMs) {
        this._armedMs += dt;
        return { progress: 0, captured: false };
      }
      this._goodMs += dt;
      const progress = Math.min(1, this._goodMs / this.holdMs);
      return { progress, captured: progress >= 1 };
    }

    if (this._goodMs <= 0 && this._armedMs <= 0) {
      return { progress: 0, captured: false };
    }

    this._badMs += dt;
    if (this._badMs >= this.dropGraceMs) {
      this.reset();
      return { progress: 0, captured: false };
    }

    const progress = Math.min(1, this._goodMs / this.holdMs);
    return { progress, captured: false };
  }
}

/**
 * Draw the current video or preview canvas frame onto the photo canvas.
 */
export function captureVideoFrame(
  source: HTMLVideoElement | HTMLCanvasElement | null | undefined,
  canvas: HTMLCanvasElement | null | undefined,
): boolean {
  if (!source || !canvas) return false;
  const vw =
    source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const vh =
    source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  if (!vw || !vh) return false;

  canvas.width = vw;
  canvas.height = vh;
  const ctx = canvas.getContext('2d');
  if (!ctx) return false;
  ctx.drawImage(source, 0, 0, vw, vh);
  return true;
}

function clonePlacement(placement: RingPlacement | null): RingPlacement | null {
  if (!placement) return null;
  return {
    x: placement.x,
    y: placement.y,
    width: placement.width,
    orientation: placement.orientation.clone(),
    surfaceFacing: placement.surfaceFacing,
    anchors: placement.anchors.map((p: ScreenPoint) => ({ x: p.x, y: p.y })),
    fingerLength: placement.fingerLength,
    handScale: placement.handScale,
    imageMotion: placement.imageMotion
      ? {
          before: placement.imageMotion.before.map((p) => ({ x: p.x, y: p.y })),
          after: placement.imageMotion.after.map((p) => ({ x: p.x, y: p.y })),
        }
      : undefined,
  };
}

/**
 * Freeze landmarks and ring placement from the capture frame.
 */
export function buildPhotoSnapshot(options: {
  landmarks: HandPoint[] | null | undefined;
  placement: RingPlacement | null;
  handedness: 'Left' | 'Right' | null;
  isMirrored: boolean;
  videoDimensions: { width: number; height: number };
  fitCalibrated: boolean;
}): PhotoSnapshot {
  return {
    landmarks: cloneLandmarkList(options.landmarks) || [],
    placement: clonePlacement(options.placement),
    handedness: options.handedness,
    isMirrored: !!options.isMirrored,
    videoDimensions: {
      width: options.videoDimensions.width || 0,
      height: options.videoDimensions.height || 0,
    },
    fitCalibrated: !!options.fitCalibrated,
  };
}

/** Ensure Quaternion survives JSON round-trips in tests via a serializable check. */
export function snapshotHasOrientation(snapshot: PhotoSnapshot): boolean {
  const orientation = snapshot.placement?.orientation as Quaternion | undefined;
  return !!orientation && typeof orientation.clone === 'function';
}
