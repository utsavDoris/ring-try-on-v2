/**
 * Shared ring fit catalog — ported from v1/src/rings/ringCatalog.js.
 * Tune fitScale / wearClearance / offset per SKU only when a model
 * still looks slightly tight or loose after unit-hole normalize.
 */

export type RingSku =
  | "lr1844"
  | "lr1025"
  | "lr1032"
  | "lr1035"
  | "lr1101"
  | "lr1157";

export type RingFitConfig = {
  cadInnerDiameterMm: number;
  fitScale: number;
  wearClearance: number;
  offset: { x: number; y: number; z: number };
  innerDiameterHint?: number;
};

export type RingStyleOption = {
  id: RingSku;
  name: string;
  url: string;
};

export const DEFAULT_RING_FIT: RingFitConfig = {
  cadInnerDiameterMm: 17.585,
  fitScale: 1.0,
  wearClearance: 1.02,
  offset: { x: 0, y: 0, z: 0 },
};

export const RING_CATALOG: Record<RingSku, RingFitConfig> = {
  lr1844: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 1.91,
    wearClearance: 1.025,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1025: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 1.887,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1032: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 1.849,
    wearClearance: 1.02,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1035: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 2.229,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1101: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 1.893,
    wearClearance: 1.025,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1157: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 2.085,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
};

/** SKUs shown in the bottom carousel (same order as v1). */
export const RING_STYLES: RingStyleOption[] = [
  { id: "lr1844", name: "LR1844", url: "/models/LR1844.glb" },
  { id: "lr1025", name: "LR1025", url: "/models/LR1025.glb" },
  { id: "lr1032", name: "LR1032", url: "/models/LR1032.glb" },
  { id: "lr1035", name: "LR1035", url: "/models/LR1035.glb" },
  { id: "lr1101", name: "LR1101", url: "/models/LR1101.glb" },
  { id: "lr1157", name: "LR1157", url: "/models/LR1157.glb" },
];

export const DEFAULT_RING_SKU: RingSku = "lr1844";

export function isRingSku(value: string): value is RingSku {
  return value in RING_CATALOG;
}

export function ringModelUrl(sku: RingSku): string {
  return `/models/${sku.toUpperCase()}.glb`;
}

/**
 * Resolve fit config for a model URL or SKU id.
 * @param urlOrSku e.g. '/models/LR1032.glb' or 'lr1032'
 */
export function getRingFitConfig(urlOrSku: string): {
  sku: RingSku | string;
  config: RingFitConfig;
} {
  const base =
    String(urlOrSku || "")
      .split(/[/\\]/)
      .pop()
      ?.replace(/\.glb$/i, "")
      .toLowerCase() || DEFAULT_RING_SKU;

  const catalogEntry = isRingSku(base) ? RING_CATALOG[base] : null;
  const config: RingFitConfig = catalogEntry
    ? {
        ...DEFAULT_RING_FIT,
        ...catalogEntry,
        offset: {
          ...DEFAULT_RING_FIT.offset,
          ...(catalogEntry.offset || {}),
        },
      }
    : { ...DEFAULT_RING_FIT, offset: { ...DEFAULT_RING_FIT.offset } };

  return { sku: base, config };
}
