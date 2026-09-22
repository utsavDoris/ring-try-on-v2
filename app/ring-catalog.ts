/**
 * Shared ring fit catalog — ported from v1/src/rings/ringCatalog.js.
 * Every SKU is worn like LR1844: hole along the finger, head on −Z.
 * GLBs that are authored sideways are rotated at load; innerDiameterHint
 * is only a fallback when the metal hole cannot be measured.
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

export type MetalColor = 'rose' | 'yellow' | 'white';

export type MetalOption = {
  id: MetalColor;
  label: string;
  hex: string;
  color: number;
  specularColor: number;
  roughness: number;
  metalness: number;
  clearcoat: number;
  clearcoatRoughness: number;
  envMapIntensity: number;
  specularIntensity: number;
  swatch: string;
  glow: string;
};

export const METAL_OPTIONS: MetalOption[] = [
  {
    id: 'rose',
    label: 'Rose Gold',
    hex: '#e0a48c',
    color: 0xe0a48c,
    specularColor: 0xffd4c8,
    roughness: 0.11,
    metalness: 1.0,
    clearcoat: 0.30,
    clearcoatRoughness: 0.14,
    envMapIntensity: 1.30,
    specularIntensity: 1.0,
    swatch: 'radial-gradient(circle at 32% 28%, #fff1ea 0%, #e8b4a4 36%, #c47a6a 70%, #8e4b41 100%)',
    glow: 'rgba(224, 164, 140, 0.45)',
  },
  {
    id: 'yellow',
    label: 'Yellow Gold',
    hex: '#e4c15a',
    color: 0xe4c15a,
    specularColor: 0xffe7a8,
    roughness: 0.10,
    metalness: 1.0,
    clearcoat: 0.35,
    clearcoatRoughness: 0.12,
    envMapIntensity: 1.35,
    specularIntensity: 1.0,
    swatch: 'radial-gradient(circle at 32% 28%, #fff6d4 0%, #e8c56a 38%, #c9a227 72%, #8a6a12 100%)',
    glow: 'rgba(228, 193, 90, 0.55)',
  },
  {
    id: 'white',
    label: 'White Gold',
    hex: '#f2f0eb',
    color: 0xf2f0eb,
    specularColor: 0xffffff,
    roughness: 0.08,
    metalness: 1.0,
    clearcoat: 0.40,
    clearcoatRoughness: 0.08,
    envMapIntensity: 1.45,
    specularIntensity: 1.0,
    swatch: 'radial-gradient(circle at 32% 28%, #ffffff 0%, #f4f1ea 38%, #d8d4cc 72%, #b8b3aa 100%)',
    glow: 'rgba(242, 240, 235, 0.45)',
  },
];

export const DEFAULT_METAL_COLOR: MetalColor = 'yellow';

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
    innerDiameterHint: 1.52,
    wearClearance: 1.02,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1032: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 1.321,
    wearClearance: 1.02,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1035: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 2.258,
    wearClearance: 1.02,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1101: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 1.454,
    wearClearance: 1.025,
    offset: { ...DEFAULT_RING_FIT.offset },
  },
  lr1157: {
    ...DEFAULT_RING_FIT,
    innerDiameterHint: 1.799,
    wearClearance: 1.02,
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
