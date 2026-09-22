export const DIAMOND_FRAG_GLSL = `
precision highp float;
precision highp sampler2D;
precision highp samplerCube;

varying vec3 vWorldPosition;
varying vec3 vWorldNormal;

uniform float uIor;
uniform int uBounces;
uniform float uDispersion;
uniform float uEnvIntensity;
uniform float uEpsilon;
uniform float uMinHit;
uniform float uAbsorption;
uniform float uSparkle;
uniform float uSceneLuma;
uniform float uArEnvBlend;
uniform vec3 uSceneColor;
uniform float uCausticStrength;
uniform float uCrownLuster;
uniform float uFacetScintillation;
uniform float uContrastDepth;
uniform vec3 uGemColor;
uniform samplerCube uEnvMap;
uniform sampler2D uBvhNodes;
uniform sampler2D uBvhTriangles;
uniform vec2 uNodeTexSize;
uniform vec2 uTriTexSize;
uniform mat4 uModelMatrix;
uniform mat4 uInverseModelMatrix;

// Finger soft-depth (bound from the scene — SoftDepth never injects into this shader).
uniform sampler2D fingerOcclusionDepth;
uniform vec2 fingerOcclusionResolution;
uniform float fingerFadeDepth;
uniform float fingerOcclusionEnabled;
uniform float showFingerOcclusionFade;
uniform float uOpacity;

// Wear-local angular edge fade (shared with metal — finger axis = local Y).
uniform float uFadeStart;
uniform float uFadeEnd;
uniform float uEdgeFadeEnabled;
uniform mat4 uWearInverse;

#include <bvh>

const int MAX_BOUNCES = 6;
const int MAX_EXITS = 2;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

/** Soft knee — keeps sparkle without milking to white or crushing to black. */
vec3 compressStudio(vec3 c) {
  return c / (vec3(1.0) + c * 0.16);
}

vec3 groundStudioEnv(vec3 studio) {
  float blend = clamp(uArEnvBlend * 0.28, 0.0, 0.28);
  vec3 sceneHue = mix(vec3(uSceneLuma), uSceneColor, 0.40);
  return mix(studio, sceneHue * max(uSceneLuma, 0.01) * 1.25, blend);
}

/**
 * Bright studio with moderate contrast cards + RGB pin lights for diamond fire.
 * Dark cards define structure without crushing the stone to black.
 */
vec3 sampleEnv(vec3 dir) {
  vec3 d = normalize(dir);
  // Lifted fill — bright D–F body, not charcoal
  vec3 fill = mix(vec3(0.05), vec3(0.14), d.y * 0.5 + 0.5);

  // Moderate dark cards — structure without killing brightness
  float darkCard1 = smoothstep(0.62, 0.94, abs(d.x * 0.866 + d.z * 0.5));
  float darkCard2 = smoothstep(0.62, 0.94, abs(-d.x * 0.5 + d.z * 0.866));
  float darkCard3 = smoothstep(0.68, 0.97, abs(d.y));
  float darkMask = max(max(darkCard1, darkCard2), darkCard3);
  float darkCutout = 1.0 - 0.48 * darkMask;

  // Pin lights with mild RGB tint — feeds spectral inner reflections
  float l1 = pow(max(dot(d, normalize(vec3(0.45, 0.72, 0.42))), 0.0), 180.0);
  float l2 = pow(max(dot(d, normalize(vec3(-0.62, 0.55, 0.28))), 0.0), 210.0);
  float l3 = pow(max(dot(d, normalize(vec3(0.08, 0.18, -0.92))), 0.0), 150.0);
  float l4 = pow(max(dot(d, normalize(vec3(0.75, 0.12, -0.18))), 0.0), 230.0);
  float l5 = pow(max(dot(d, normalize(vec3(-0.22, -0.15, 0.85))), 0.0), 170.0);
  float lTable = pow(max(dot(d, normalize(vec3(0.0, -0.98, 0.20))), 0.0), 480.0);
  float lKicker = pow(max(dot(d, normalize(vec3(0.10, 0.95, 0.15))), 0.0), 360.0);

  vec3 lamps =
    vec3(7.2, 6.6, 5.8) * l1 +
    vec3(5.2, 6.0, 7.0) * l2 +
    vec3(5.5, 5.2, 6.2) * l3 +
    vec3(6.8, 5.8, 5.2) * l4 +
    vec3(5.0, 5.6, 6.4) * l5 +
    vec3(9.5, 9.2, 8.6) * lTable +
    vec3(4.2, 4.6, 5.2) * lKicker;

  vec3 hdr = textureCube(uEnvMap, d).rgb;
  float hdrLum = max(dot(hdr, LUMA), 0.0);
  // HDR always contributes; dark cards only soft-attenuate it
  float hdrWeight = mix(0.65, 1.20, smoothstep(0.0, 0.35, hdrLum)) * mix(0.55, 1.0, darkCutout);

  vec3 studio = compressStudio((fill * darkCutout + lamps + hdr * hdrWeight) * uEnvIntensity);
  studio += compressStudio(lamps * lamps * 0.022 * uSparkle);
  return groundStudioEnv(studio);
}

vec3 toObjectPoint(vec3 worldP) {
  return (uInverseModelMatrix * vec4(worldP, 1.0)).xyz;
}

vec3 toObjectDir(vec3 worldD) {
  return normalize(mat3(uInverseModelMatrix) * worldD);
}

vec3 toWorldDir(vec3 objectD) {
  return normalize(mat3(uModelMatrix) * objectD);
}

float schlickFresnel(float cosTheta, float ior) {
  float r0 = (ior - 1.0) / (ior + 1.0);
  r0 *= r0;
  float m = 1.0 - cosTheta;
  return r0 + (1.0 - r0) * m * m * m * m * m;
}

vec3 beer(float pathLen) {
  return exp(-vec3(uAbsorption) * pathLen);
}

vec3 refractEnv(vec3 dir, vec3 n, vec3 refl, float eta) {
  vec3 r = refract(dir, n, eta);
  return sampleEnv(toWorldDir(dot(r, r) > 1e-6 ? r : refl));
}

/** Five-wavelength RGB spectral fire on exit — Cauchy dispersion for inner colored rays. */
vec3 fireExit(vec3 dir, vec3 n, float ior, float disp) {
  vec3 refl = reflect(dir, n);
  if (disp < 1e-4) {
    vec3 r = refract(dir, n, ior);
    return sampleEnv(toWorldDir(dot(r, r) > 1e-6 ? r : refl));
  }

  // Wider spectral split so RGB fire reads clearly on camera-sized gems
  float d = disp * 1.35;
  vec3 c0 = refractEnv(dir, n, refl, clamp(ior - d * 1.45, 1.01, 2.75));
  vec3 c1 = refractEnv(dir, n, refl, clamp(ior - d * 0.55, 1.01, 2.75));
  vec3 c2 = refractEnv(dir, n, refl, clamp(ior, 1.01, 2.75));
  vec3 c3 = refractEnv(dir, n, refl, clamp(ior + d * 0.65, 1.01, 2.75));
  vec3 c4 = refractEnv(dir, n, refl, clamp(ior + d * 1.50, 1.01, 2.75));

  vec3 redBand    = c0 * vec3(0.55, 0.02, 0.00);
  vec3 yellowBand = c1 * vec3(0.32, 0.28, 0.00);
  vec3 greenBand  = c2 * vec3(0.02, 0.48, 0.04);
  vec3 cyanBand   = c3 * vec3(0.00, 0.24, 0.48);
  vec3 violetBand = c4 * vec3(0.11, 0.02, 0.55);

  vec3 spectralColor = redBand + yellowBand + greenBand + cyanBand + violetBand;
  vec3 neutralColor = (c0 + c1 + c2 + c3 + c4) * 0.20;

  // Prefer saturated RGB fire while keeping a neutral body anchor
  return mix(neutralColor, spectralColor * 1.22, 0.88);
}

/**
 * Skip entry facet, then honest TIR / Fresnel-weighted multi-exit (max 2).
 * Inner rays come from real exit paths — no fake fireExit on TIR.
 * Returns RGB in xyz and accumulated facet contrast in w.
 */
vec4 traceInterior(vec3 worldOrigin, vec3 worldRefractDir, float ior) {
  vec3 dir = toObjectDir(worldRefractDir);
  vec3 orig = toObjectPoint(worldOrigin) + dir * (uEpsilon * 8.0);
  float pathLen = 0.0;
  float facetContrast = 0.0;
  float firstSkip = max(uMinHit, uEpsilon * 8.0);
  vec3 exitAccum = vec3(0.0);
  float exitWeight = 0.0;
  float pathThroughput = 1.0;
  int exitsTaken = 0;

  for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
    if (bounce >= uBounces) break;

    float tHit;
    vec3 hitN;
    float tMin = bounce == 0 ? firstSkip : uEpsilon;
    if (!intersectBvh(
      uBvhNodes, uNodeTexSize, uBvhTriangles, uTriTexSize,
      orig, dir, tMin, tHit, hitN
    )) {
      // Missed mesh — sample along current direction (open path)
      vec3 missColor = sampleEnv(toWorldDir(dir)) * beer(pathLen) * pathThroughput;
      if (exitWeight < 1e-4) {
        return vec4(missColor, facetContrast);
      }
      float remain = max(1.0 - exitWeight, 0.0);
      return vec4(exitAccum + missColor * remain, facetContrast);
    }

    if (bounce == 0 && tHit < firstSkip) {
      orig += dir * (tHit + firstSkip * 0.35);
      continue;
    }

    pathLen += tHit;
    vec3 hitPos = orig + dir * tHit;
    vec3 n = dot(hitN, dir) > 0.0 ? -hitN : hitN;
    float facing = pow(abs(dot(n, -dir)), 0.6);
    facetContrast = max(facetContrast, facing);
    float cosOut = clamp(dot(-dir, n), 0.0, 1.0);
    float F = schlickFresnel(cosOut, ior);

    // Refract OUT from diamond (n1 = ior) to air (n2 = 1.0) -> eta = ior
    vec3 refrOut = refract(dir, n, ior);
    bool last = bounce >= uBounces - 1;
    bool canExit = dot(refrOut, refrOut) > 1e-6;

    if (canExit) {
      vec3 exitColor = fireExit(dir, n, ior, uDispersion) * beer(pathLen);
      // Boost spectral energy so RGB inner rays read on small AR gems
      float exitPeak = max(dot(exitColor, LUMA), 0.0);
      float chroma = length(exitColor - vec3(exitPeak));
      exitColor += exitColor * (exitPeak * uSparkle * 0.05 + chroma * uDispersion * 2.5);

      float transmitW = (1.0 - F) * pathThroughput;
      exitAccum += exitColor * transmitW;
      exitWeight += transmitW;
      exitsTaken += 1;

      // Fresnel-weighted multi-exit: continue reflecting for layered RGB inner rays
      if (exitsTaken < MAX_EXITS && !last && F > 0.04) {
        pathThroughput *= F;
        dir = reflect(dir, n);
        orig = hitPos + dir * uEpsilon;
        continue;
      }

      float multiBoost = 1.0 + max(uCausticStrength, 0.35) * 0.25 * float(exitsTaken - 1);
      return vec4(exitAccum * multiBoost, facetContrast);
    }

    // TIR: continue path; soft chromatic reflect adds inner RGB without fake fireExit paint
    if (last) {
      vec3 reflected = fireExit(dir, n, ior, uDispersion * 0.55) * beer(pathLen) * pathThroughput;
      if (exitWeight > 1e-4) {
        float remain = max(1.0 - exitWeight, 0.0);
        return vec4(exitAccum + reflected * remain, facetContrast);
      }
      return vec4(reflected, facetContrast);
    }

    dir = reflect(dir, n);
    orig = hitPos + dir * uEpsilon;
  }

  vec3 fallback = sampleEnv(toWorldDir(dir)) * beer(pathLen) * pathThroughput;
  if (exitWeight > 1e-4) {
    float remain = max(1.0 - exitWeight, 0.0);
    return vec4(exitAccum + fallback * remain, facetContrast);
  }
  return vec4(fallback, facetContrast);
}

void main() {
  vec3 N = normalize(vWorldNormal);
  if (!gl_FrontFacing) N = -N;

  vec3 V = normalize(cameraPosition - vWorldPosition);
  if (length(cameraPosition) < 1e-4) {
    V = vec3(0.0, 0.0, 1.0);
  }
  vec3 I = -V;
  float F = schlickFresnel(clamp(dot(N, V), 0.0, 1.0), uIor);
  vec3 reflected = sampleEnv(reflect(I, N));

  // Ray entering diamond from air (n1 = 1.0, n2 = uIor) -> eta = 1.0 / uIor
  vec3 refractDir = refract(I, N, 1.0 / max(uIor, 1.0001));
  vec4 traced = dot(refractDir, refractDir) < 1e-6
    ? vec4(reflected, 0.0)
    : traceInterior(vWorldPosition, refractDir, uIor);
  vec3 interior = traced.rgb;
  float facetContrast = traced.a;

  float fresnel = pow(F, 0.90);
  vec3 color = mix(interior, reflected, fresnel);

  // Body color tinting (for white or fancy colored diamonds)
  color *= uGemColor;

  // Facet contrast — structure without crushing to black
  float contrastFactor = mix(1.0, 0.55 + 0.75 * facetContrast, clamp(uContrastDepth, 0.0, 1.0));
  color *= contrastFactor;

  // Thin spectral rim fire — visible RGB at grazing angles
  float crownFacing = clamp(dot(N, V), 0.0, 1.0);
  float pavilionFacing = clamp(dot(-N, V), 0.0, 1.0);
  float rim = pow(1.0 - crownFacing, 3.2);
  float rimStrength = rim * uDispersion * uSparkle * 0.42;
  color += vec3(0.95, 0.55, 0.35) * rimStrength * 0.55;
  color += vec3(0.35, 0.55, 1.0) * rimStrength * 0.45;
  color += vec3(1.0, 0.92, 0.75) * pow(pavilionFacing, 2.8) * uDispersion * uSparkle * 0.22;

  // Dual crown table micro-specular highlights for adamantine luster
  vec3 L_table1 = normalize(vec3(0.12, 0.96, 0.22));
  vec3 L_table2 = normalize(vec3(-0.35, 0.88, -0.28));
  vec3 H1 = normalize(V + L_table1);
  vec3 H2 = normalize(V + L_table2);
  float specAngle1 = max(dot(N, H1), 0.0);
  float specAngle2 = max(dot(N, H2), 0.0);
  vec3 specHighlight1 = vec3(1.6, 1.55, 1.45) * pow(specAngle1, 380.0) * F * uSparkle * uCrownLuster * 1.0;
  vec3 specHighlight2 = vec3(1.15, 1.25, 1.45) * pow(specAngle2, 260.0) * F * uSparkle * uCrownLuster * 0.65;
  color += specHighlight1 + specHighlight2;

  // Facet-edge RGB scintillation — clear colored flecks
  float facetHash = fract(sin(dot(N, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
  float edgeSparkle = smoothstep(0.42, 0.92, facetHash) * uDispersion * uSparkle * uFacetScintillation;
  vec3 edgeFire = vec3(
    0.85 + 0.35 * sin(facetHash * 22.0),
    0.75 + 0.40 * cos(facetHash * 17.0),
    0.90 + 0.35 * sin(facetHash * 27.0 + 1.5)
  );
  color += edgeFire * edgeSparkle * 0.22;

  // Soft tone — keep body bright and RGB intact
  color = pow(max(color, vec3(0.0)), vec3(0.94));

  float bodyLuma = dot(color, LUMA);
  if (bodyLuma < 0.05) {
    color = mix(color, normalize(max(color, vec3(0.001))) * 0.07, smoothstep(0.0, 0.05, 0.05 - bodyLuma) * 0.5);
  }

  // Finger proxy: hard discard only when clearly behind the finger — never dim RGB.
  float fingerOcclusionVisibility = 1.0;
  if (fingerOcclusionEnabled > 0.5) {
    vec2 fingerOcclusionUv = gl_FragCoord.xy / max(fingerOcclusionResolution, vec2(1.0));
    float fingerDepth = texture2D(fingerOcclusionDepth, fingerOcclusionUv).x;
    if (fingerDepth < 0.999999) {
      float behindFinger = max(0.0, gl_FragCoord.z - fingerDepth);
      fingerOcclusionVisibility = 1.0 - smoothstep(0.0, max(fingerFadeDepth, 0.000001), behindFinger);
    }
  }
  float discardCutoff = crownFacing > 0.28 ? 0.08 : 0.03;
  if (fingerOcclusionVisibility < discardCutoff) discard;
  if (showFingerOcclusionFade > 0.5) {
    float fadeBand = 4.0 * fingerOcclusionVisibility * (1.0 - fingerOcclusionVisibility);
    color = mix(color, vec3(1.0, 0.12, 0.45), clamp(fadeBand, 0.0, 1.0) * 0.72);
  }

  // Curved edge fade along finger cylinder (alpha only — keep stone bright)
  float edgeFade = 1.0;
  if (uEdgeFadeEnabled > 0.5) {
    vec3 localPos = (uWearInverse * vec4(vWorldPosition, 1.0)).xyz;
    vec3 localCam = (uWearInverse * vec4(cameraPosition, 1.0)).xyz;
    vec2 vF = localPos.xz;
    vec2 vC = localCam.xz;
    float dF = length(vF);
    float dC = length(vC);
    if (dF > 0.0001 && dC > 0.0001) {
      float cosA = clamp(dot(vF, vC) / (dF * dC), -1.0, 1.0);
      float ang = acos(cosA);
      float angStart = acos(clamp(uFadeStart, -1.0, 1.0));
      float angEnd = acos(clamp(uFadeEnd, -1.0, 1.0));
      edgeFade = 1.0 - smoothstep(angStart, angEnd, ang);
      edgeFade = pow(clamp(edgeFade, 0.0, 1.0), 0.85);
    }
  }

  float outAlpha = clamp(uOpacity * edgeFade, 0.0, 1.0);
  if (outAlpha < 0.005) discard;
  gl_FragColor = vec4(color, outAlpha);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;
