import * as THREE from 'three';
import { buildBvh } from './bvh/buildBvh.ts';
import { disposePackedBvhTextures, packBvhTextures } from './bvh/packTextures.ts';
import {
  DIAMOND_BOUNCES,
  DIAMOND_MIN_HIT_RATIO,
  realisticDiamondMaterialParams,
} from './diamondDefaults.ts';
import { BVH_GLSL } from './shaders/bvh-glsl.ts';
import { DIAMOND_VERT_GLSL } from './shaders/diamond-vert.ts';
import { DIAMOND_FRAG_GLSL } from './shaders/diamond-frag.ts';

export interface SoftDepthFadeState {
  depthTexture: THREE.DepthTexture | null;
  screenResolution?: { width: number; height: number };
  fadeDepth: number;
  enabled: boolean;
  showFade?: boolean;
}

export interface EdgeFadeState {
  wearInverse: THREE.Matrix4;
  fadeStart: number;
  fadeEnd: number;
  enabled: boolean;
}

export interface DiamondMaterialOptions {
  geometry: THREE.BufferGeometry;
  envMap: THREE.CubeTexture;
  defaultBounces?: number;
}

export interface DiamondMaterialParams {
  ior?: number;
  bounces?: number;
  dispersion?: number;
  envIntensity?: number;
  sparkle?: number;
  absorption?: number;
  sceneLuma?: number;
  arEnvBlend?: number;
  sceneColor?: THREE.Vector3;
  causticStrength?: number;
  crownLuster?: number;
  facetScintillation?: number;
  contrastDepth?: number;
  gemColor?: THREE.Vector3;
}

interface DiamondUniforms {
  uIor: { value: number };
  uBounces: { value: number };
  uDispersion: { value: number };
  uEnvIntensity: { value: number };
  uEpsilon: { value: number };
  uMinHit: { value: number };
  uAbsorption: { value: number };
  uSparkle: { value: number };
  uSceneLuma: { value: number };
  uArEnvBlend: { value: number };
  uSceneColor: { value: THREE.Vector3 };
  uCausticStrength: { value: number };
  uCrownLuster: { value: number };
  uFacetScintillation: { value: number };
  uContrastDepth: { value: number };
  uGemColor: { value: THREE.Vector3 };
  uEnvMap: { value: THREE.CubeTexture };
  uBvhNodes: { value: THREE.DataTexture };
  uBvhTriangles: { value: THREE.DataTexture };
  uNodeTexSize: { value: THREE.Vector2 };
  uTriTexSize: { value: THREE.Vector2 };
  uModelMatrix: { value: THREE.Matrix4 };
  uInverseModelMatrix: { value: THREE.Matrix4 };
  fingerOcclusionDepth: { value: THREE.DepthTexture | null };
  fingerOcclusionResolution: { value: THREE.Vector2 };
  fingerFadeDepth: { value: number };
  fingerOcclusionEnabled: { value: number };
  showFingerOcclusionFade: { value: number };
  uOpacity: { value: number };
  uFadeStart: { value: number };
  uFadeEnd: { value: number };
  uEdgeFadeEnabled: { value: number };
  uWearInverse: { value: THREE.Matrix4 };
}

function geometrySizeScale(geometry: THREE.BufferGeometry): number {
  geometry.computeBoundingBox();
  const size = new THREE.Vector3();
  geometry.boundingBox?.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  return maxDim / 2;
}

/**
 * Split shared vertices and write one geometric normal per triangle
 * so every cut stays hard.
 */
export function flattenGemFacets(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  const geo = geometry.index ? geometry.toNonIndexed() : geometry.clone();
  const pos = geo.attributes.position;
  if (!pos) return geo;

  const count = pos.count;
  const normals = new Float32Array(count * 3);
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const cb = new THREE.Vector3();
  const ab = new THREE.Vector3();

  for (let i = 0; i < count; i += 3) {
    a.fromBufferAttribute(pos, i);
    b.fromBufferAttribute(pos, i + 1);
    c.fromBufferAttribute(pos, i + 2);
    cb.subVectors(c, b);
    ab.subVectors(a, b);
    cb.cross(ab).normalize();

    normals[i * 3] = cb.x;
    normals[i * 3 + 1] = cb.y;
    normals[i * 3 + 2] = cb.z;
    normals[(i + 1) * 3] = cb.x;
    normals[(i + 1) * 3 + 1] = cb.y;
    normals[(i + 1) * 3 + 2] = cb.z;
    normals[(i + 2) * 3] = cb.x;
    normals[(i + 2) * 3 + 1] = cb.y;
    normals[(i + 2) * 3 + 2] = cb.z;
  }

  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  return geo;
}

export class DiamondMaterial extends THREE.ShaderMaterial {
  readonly bvhStats: { triangleCount: number; nodeCount: number };
  private bvhTexturesDisposed = false;
  private edgeFadeActive = true;
  private readonly bvhTextures: {
    nodeTexture: THREE.DataTexture;
    triangleTexture: THREE.DataTexture;
  };

  constructor({ geometry, envMap, defaultBounces = DIAMOND_BOUNCES }: DiamondMaterialOptions) {
    const bvh = buildBvh(geometry);
    const packed = packBvhTextures(bvh);
    const scale = geometrySizeScale(geometry);
    const look = realisticDiamondMaterialParams();

    const estimatedDepth = Math.ceil(Math.log2((bvh.nodes?.length ?? 1) + 1)) + 2;
    const adaptiveStack = Math.min(Math.max(estimatedDepth, 12), 24);
    const adaptiveSteps = Math.min(Math.max(adaptiveStack * 3, 48), 96);
    const bvhGlslPatched = BVH_GLSL
      .replace(/const int BVH_STACK_SIZE = \d+;/, '')
      .replace(/const int BVH_STEPS = \d+;/, '');
    const defines = [
      `#define BVH_STACK_SIZE ${adaptiveStack}`,
      `#define BVH_STEPS ${adaptiveSteps}`,
    ].join('\n') + '\n';
    const patchedBvhGlsl = defines + bvhGlslPatched;

    super({
      name: 'DiamondMaterial',
      uniforms: {
        uIor: { value: look.ior },
        uBounces: { value: defaultBounces },
        uDispersion: { value: look.dispersion },
        uEnvIntensity: { value: look.envIntensity },
        uEpsilon: { value: 0.0007 * scale },
        uMinHit: { value: DIAMOND_MIN_HIT_RATIO * scale },
        uAbsorption: { value: look.absorption },
        uSparkle: { value: look.sparkle },
        uSceneLuma: { value: 0.45 },
        uArEnvBlend: { value: look.arEnvBlend },
        uSceneColor: { value: new THREE.Vector3(0.45, 0.45, 0.45) },
        uCausticStrength: { value: look.causticStrength },
        uCrownLuster: { value: look.crownLuster },
        uFacetScintillation: { value: look.facetScintillation },
        uContrastDepth: { value: look.contrastDepth },
        uGemColor: { value: look.gemColor.clone() },
        uEnvMap: { value: envMap },
        uBvhNodes: { value: packed.nodeTexture },
        uBvhTriangles: { value: packed.triangleTexture },
        uNodeTexSize: { value: packed.nodeTexSize },
        uTriTexSize: { value: packed.triTexSize },
        uModelMatrix: { value: new THREE.Matrix4() },
        uInverseModelMatrix: { value: new THREE.Matrix4() },
        fingerOcclusionDepth: { value: null },
        fingerOcclusionResolution: { value: new THREE.Vector2(1, 1) },
        fingerFadeDepth: { value: 0.0001 },
        fingerOcclusionEnabled: { value: 0 },
        showFingerOcclusionFade: { value: 0 },
        uOpacity: { value: 1 },
        uFadeStart: { value: 0.20 },
        uFadeEnd: { value: 0.0 },
        uEdgeFadeEnabled: { value: 1 },
        uWearInverse: { value: new THREE.Matrix4() },
      },
      vertexShader: DIAMOND_VERT_GLSL,
      fragmentShader: DIAMOND_FRAG_GLSL.replace('#include <bvh>', patchedBvhGlsl),
      side: THREE.FrontSide,
      transparent: true,
      depthWrite: true,
      depthTest: true,
      toneMapped: true,
    });

    this.bvhStats = {
      triangleCount: packed.triangleCount,
      nodeCount: packed.nodeCount,
    };
    this.bvhTextures = {
      nodeTexture: packed.nodeTexture,
      triangleTexture: packed.triangleTexture,
    };
  }

  updateFromMesh(mesh: THREE.Object3D): void {
    const uniforms = this.uniforms as unknown as DiamondUniforms;
    uniforms.uModelMatrix.value.copy(mesh.matrixWorld);
    uniforms.uInverseModelMatrix.value.copy(mesh.matrixWorld).invert();
  }

  updateSoftDepth(state: SoftDepthFadeState): void {
    const uniforms = this.uniforms as unknown as DiamondUniforms;
    uniforms.fingerOcclusionDepth.value = state.depthTexture;
    uniforms.fingerOcclusionResolution.value.set(
      state.screenResolution?.width ?? 1,
      state.screenResolution?.height ?? 1,
    );
    uniforms.fingerFadeDepth.value = Math.max(0.000001, state.fadeDepth);
    uniforms.fingerOcclusionEnabled.value = state.enabled && state.depthTexture ? 1 : 0;
    uniforms.showFingerOcclusionFade.value = state.enabled && state.showFade ? 1 : 0;
  }

  updateEdgeFade(state: EdgeFadeState): void {
    const uniforms = this.uniforms as unknown as DiamondUniforms;
    uniforms.uWearInverse.value.copy(state.wearInverse);
    uniforms.uFadeStart.value = state.fadeStart;
    uniforms.uFadeEnd.value = state.fadeEnd;
    uniforms.uEdgeFadeEnabled.value = state.enabled ? 1 : 0;
    this.edgeFadeActive = state.enabled;
    this.transparent = true;
    this.depthWrite = true;
  }

  setParams({
    ior,
    bounces,
    dispersion,
    envIntensity,
    sparkle,
    absorption,
    sceneLuma,
    arEnvBlend,
    sceneColor,
    causticStrength,
    crownLuster,
    facetScintillation,
    contrastDepth,
    gemColor,
  }: DiamondMaterialParams): void {
    const uniforms = this.uniforms as unknown as DiamondUniforms;
    if (ior !== undefined) uniforms.uIor.value = ior;
    if (bounces !== undefined) uniforms.uBounces.value = bounces;
    if (dispersion !== undefined) uniforms.uDispersion.value = dispersion;
    if (envIntensity !== undefined) uniforms.uEnvIntensity.value = envIntensity;
    if (sparkle !== undefined) uniforms.uSparkle.value = sparkle;
    if (absorption !== undefined) uniforms.uAbsorption.value = absorption;
    if (sceneLuma !== undefined) uniforms.uSceneLuma.value = sceneLuma;
    if (arEnvBlend !== undefined) uniforms.uArEnvBlend.value = arEnvBlend;
    if (sceneColor !== undefined) uniforms.uSceneColor.value.copy(sceneColor);
    if (causticStrength !== undefined) uniforms.uCausticStrength.value = causticStrength;
    if (crownLuster !== undefined) uniforms.uCrownLuster.value = crownLuster;
    if (facetScintillation !== undefined) uniforms.uFacetScintillation.value = facetScintillation;
    if (contrastDepth !== undefined) uniforms.uContrastDepth.value = contrastDepth;
    if (gemColor !== undefined) uniforms.uGemColor.value.copy(gemColor);
  }

  getParams(): Required<Pick<DiamondMaterialParams, 'ior' | 'bounces' | 'dispersion' | 'envIntensity' | 'sparkle' | 'arEnvBlend'>> {
    const uniforms = this.uniforms as unknown as DiamondUniforms;
    return {
      ior: uniforms.uIor.value,
      bounces: uniforms.uBounces.value,
      dispersion: uniforms.uDispersion.value,
      envIntensity: uniforms.uEnvIntensity.value,
      sparkle: uniforms.uSparkle.value,
      arEnvBlend: uniforms.uArEnvBlend.value,
    };
  }

  setOpacity(opacity: number): void {
    const clamped = Math.min(1, Math.max(0, opacity));
    const uniforms = this.uniforms as unknown as DiamondUniforms;
    uniforms.uOpacity.value = clamped;
    this.transparent = this.edgeFadeActive || clamped < 0.999;
    this.depthWrite = true;
    this.needsUpdate = true;
  }

  override dispose(): void {
    if (!this.bvhTexturesDisposed) {
      this.bvhTexturesDisposed = true;
      disposePackedBvhTextures(this.bvhTextures);
    }
    super.dispose();
  }
}
