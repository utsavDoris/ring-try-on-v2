/**
 * MediaPipe hand landmark map for try-on fingers (ported from v1).
 * Left and Right hands share the same indices; handedness is handled in pose.
 */

export const LANDMARKS = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
} as const;

export type TryOnFinger = 'thumb' | 'index' | 'middle' | 'ring' | 'pinky';

export type FingerDef = {
  name: TryOnFinger;
  label: string;
  mcp: number;
  pip: number;
  dip: number;
  tip: number;
  widthRatio: number;
  defaultSeat: number;
};

/** UI try-on fingers */
export const TRY_ON_FINGERS: FingerDef[] = [
  {
    name: 'thumb',
    label: 'Thumb',
    mcp: LANDMARKS.THUMB_MCP,
    pip: LANDMARKS.THUMB_IP,
    dip: LANDMARKS.THUMB_IP, // Thumb only has IP, so we duplicate it for DIP if needed
    tip: LANDMARKS.THUMB_TIP,
    widthRatio: 1.18,
    defaultSeat: 0.46,
  },
  {
    name: 'index',
    label: 'Index',
    mcp: LANDMARKS.INDEX_MCP,
    pip: LANDMARKS.INDEX_PIP,
    dip: LANDMARKS.INDEX_DIP,
    tip: LANDMARKS.INDEX_TIP,
    widthRatio: 1.02,
    defaultSeat: 0.42,
  },
  {
    name: 'middle',
    label: 'Middle',
    mcp: LANDMARKS.MIDDLE_MCP,
    pip: LANDMARKS.MIDDLE_PIP,
    dip: LANDMARKS.MIDDLE_DIP,
    tip: LANDMARKS.MIDDLE_TIP,
    widthRatio: 1.07,
    defaultSeat: 0.42,
  },
  {
    name: 'ring',
    label: 'Ring',
    mcp: LANDMARKS.RING_MCP,
    pip: LANDMARKS.RING_PIP,
    dip: LANDMARKS.RING_DIP,
    tip: LANDMARKS.RING_TIP,
    widthRatio: 1.0,
    defaultSeat: 0.60,
  },
  {
    name: 'pinky',
    label: 'Pinky',
    mcp: LANDMARKS.PINKY_MCP,
    pip: LANDMARKS.PINKY_PIP,
    dip: LANDMARKS.PINKY_DIP,
    tip: LANDMARKS.PINKY_TIP,
    widthRatio: 0.8,
    defaultSeat: 0.38,
  },
];

export const DEFAULT_TRY_ON_FINGER: TryOnFinger = 'ring';

export function getFingerDef(name: TryOnFinger): FingerDef {
  return TRY_ON_FINGERS.find((f) => f.name === name) ?? TRY_ON_FINGERS[3];
}

/** Neighbor bones for silhouette clipping (same pairing as v1 fingerMetrics). */
export function neighborsForFinger(
  name: TryOnFinger,
): { mcp: number; pip: number }[] {
  switch (name) {
    case 'thumb':
      return [{ mcp: LANDMARKS.INDEX_MCP, pip: LANDMARKS.INDEX_PIP }];
    case 'index':
      return [{ mcp: LANDMARKS.MIDDLE_MCP, pip: LANDMARKS.MIDDLE_PIP }];
    case 'middle':
      return [
        { mcp: LANDMARKS.INDEX_MCP, pip: LANDMARKS.INDEX_PIP },
        { mcp: LANDMARKS.RING_MCP, pip: LANDMARKS.RING_PIP },
      ];
    case 'ring':
      return [
        { mcp: LANDMARKS.MIDDLE_MCP, pip: LANDMARKS.MIDDLE_PIP },
        { mcp: LANDMARKS.PINKY_MCP, pip: LANDMARKS.PINKY_PIP },
      ];
    case 'pinky':
    default:
      return [{ mcp: LANDMARKS.RING_MCP, pip: LANDMARKS.RING_PIP }];
  }
}
