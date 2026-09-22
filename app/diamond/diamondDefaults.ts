import * as THREE from 'three';

/**
 * Realistic brilliant-cut white diamond — load-time preset for AR try-on.
 * IOR/dispersion match diamond (n≈2.417, Abbe≈55). Tuned for live camera overlay.
 */
export const DIAMOND_REALISTIC_PRESET = {
  ior: 2.417,
  /** Spectral fire — strong enough for clear RGB inner rays on camera. */
  dispersion: 0.058,
  /** Bright studio so the stone is not dark. */
  envIntensity: 1.32,
  /** Pin-light scintillation feeding colored exits. */
  sparkle: 1.25,
  /** Near-colorless D–F body — lower = whiter pavilion. */
  absorption: 0.00045,
  /** Live video ground — room light without crushing fire. */
  arEnvBlend: 0.85,
  /** Soft multi-exit brightness boost. */
  causticStrength: 0.35,
  /** Adamantine crown & table specular luster. */
  crownLuster: 1.45,
  /** Facet-edge RGB scintillation. */
  facetScintillation: 0.85,
  /** Facet structure — moderate, not charcoal crush. */
  contrastDepth: 0.62,
  /** Interior bounce count — enough TIR to form pavilion arrows. */
  bounces: 6,
  /** First-hit skip as a fraction of gem half-extent (keep small so melee pavilion hits). */
  minHitRatio: 0.022,
  /** Default body tint (White Diamond: 1, 1, 1). Can be set to fancy pink/yellow. */
  gemColor: new THREE.Vector3(1.0, 1.0, 1.0),
} as const;

export type DiamondLookDevDefaults = {
  ior: number;
  dispersion: number;
  envIntensity: number;
  sparkle: number;
  bounces: number;
  arEnvBlend: number;
  causticStrength: number;
  crownLuster: number;
  facetScintillation: number;
  contrastDepth: number;
  gemColor: THREE.Vector3;
};

export const DIAMOND_IOR = DIAMOND_REALISTIC_PRESET.ior;
export const DIAMOND_DISPERSION = DIAMOND_REALISTIC_PRESET.dispersion;
export const DIAMOND_ENV_INTENSITY = DIAMOND_REALISTIC_PRESET.envIntensity;
export const DIAMOND_SPARKLE = DIAMOND_REALISTIC_PRESET.sparkle;
export const DIAMOND_ABSORPTION = DIAMOND_REALISTIC_PRESET.absorption;
export const DIAMOND_AR_ENV_BLEND = DIAMOND_REALISTIC_PRESET.arEnvBlend;
export const DIAMOND_CAUSTIC_STRENGTH = DIAMOND_REALISTIC_PRESET.causticStrength;
export const DIAMOND_CROWN_LUSTER = DIAMOND_REALISTIC_PRESET.crownLuster;
export const DIAMOND_FACET_SCINTILLATION = DIAMOND_REALISTIC_PRESET.facetScintillation;
export const DIAMOND_CONTRAST_DEPTH = DIAMOND_REALISTIC_PRESET.contrastDepth;
export const DIAMOND_BOUNCES = DIAMOND_REALISTIC_PRESET.bounces;
export const DIAMOND_MIN_HIT_RATIO = DIAMOND_REALISTIC_PRESET.minHitRatio;
export const DIAMOND_GEM_COLOR = DIAMOND_REALISTIC_PRESET.gemColor;

export function createDiamondLookDevDefaults(
  defaultBounces: number,
  supportsInteriorDispersion = true,
): DiamondLookDevDefaults {
  return {
    ior: DIAMOND_IOR,
    dispersion: supportsInteriorDispersion ? DIAMOND_DISPERSION : 0,
    envIntensity: DIAMOND_ENV_INTENSITY,
    sparkle: DIAMOND_SPARKLE,
    bounces: defaultBounces || DIAMOND_BOUNCES,
    arEnvBlend: DIAMOND_AR_ENV_BLEND,
    causticStrength: DIAMOND_CAUSTIC_STRENGTH,
    crownLuster: DIAMOND_CROWN_LUSTER,
    facetScintillation: DIAMOND_FACET_SCINTILLATION,
    contrastDepth: DIAMOND_CONTRAST_DEPTH,
    gemColor: DIAMOND_GEM_COLOR.clone(),
  };
}

export function realisticDiamondMaterialParams(): {
  ior: number;
  dispersion: number;
  envIntensity: number;
  sparkle: number;
  absorption: number;
  arEnvBlend: number;
  causticStrength: number;
  crownLuster: number;
  facetScintillation: number;
  contrastDepth: number;
  bounces: number;
  gemColor: THREE.Vector3;
} {
  return {
    ior: DIAMOND_IOR,
    dispersion: DIAMOND_DISPERSION,
    envIntensity: DIAMOND_ENV_INTENSITY,
    sparkle: DIAMOND_SPARKLE,
    absorption: DIAMOND_ABSORPTION,
    arEnvBlend: DIAMOND_AR_ENV_BLEND,
    causticStrength: DIAMOND_CAUSTIC_STRENGTH,
    crownLuster: DIAMOND_CROWN_LUSTER,
    facetScintillation: DIAMOND_FACET_SCINTILLATION,
    contrastDepth: DIAMOND_CONTRAST_DEPTH,
    bounces: DIAMOND_BOUNCES,
    gemColor: DIAMOND_GEM_COLOR.clone(),
  };
}
