import * as THREE from 'three';
import { shaderStructs, shaderIntersectFunction } from 'three-mesh-bvh';

export function createDiamondMaterial(envMapTexture, bvhStruct) {

    const vertexShader = /* glsl */`
        varying vec3 vWorldPos;
        varying vec3 vWorldNormal;
        varying vec3 vLocalPos;
        varying vec3 vLocalNormal;
        varying vec3 vLocalCameraPos;
        varying vec3 vLocalViewDirection;
        varying mat3 vModelRotation;
        varying vec2 vUv;

        uniform vec3 cameraPos;

        void main() {
            vUv = uv;
            vLocalPos = position;
            vLocalNormal = normal;
            vec4 wp = modelMatrix * vec4(position, 1.0);
            vWorldPos = wp.xyz;
            vWorldNormal = normalize(mat3(modelMatrix) * normal);
            
            // Pass model rotation and local camera position for local-space BVH ray tracing
            vModelRotation = mat3(modelMatrix);
            vLocalCameraPos = (inverse(modelMatrix) * vec4(cameraPos, 1.0)).xyz;
            // The try-on overlay has parallel camera rays. Transform a
            // direction (w=0), so ring translation cannot change its lighting.
            vLocalViewDirection = (inverse(modelMatrix) * vec4(0.0, 0.0, 1.0, 0.0)).xyz;
            
            gl_Position = projectionMatrix * viewMatrix * wp;
        }
    `;

    const fragmentShader = /* glsl */`
        precision highp float;
        precision highp int;
        precision highp usampler2D;
        precision highp isampler2D;
        out highp vec4 pc_fragColor;
        #define gl_FragColor pc_fragColor
        
        uniform sampler2D envMap;
        uniform mat3      envRotationMatrix;
        
        uniform vec3      cameraPos;
        uniform float     ior;
        uniform float     dispersion;
        uniform float     absorption;
        uniform float     envIntensity;
        uniform float     ambientMin;
        uniform float     gamma;
        uniform vec3      color;
        uniform vec3      rgbBoost;
        uniform float     transmission;
        uniform float     reflectivity;
        uniform int       rayBounces;

        ${shaderStructs}
        ${shaderIntersectFunction}
        uniform BVH bvh;

        varying vec3 vWorldPos;
        varying vec3 vWorldNormal;
        varying vec3 vLocalPos;
        varying vec3 vLocalNormal;
        varying vec3 vLocalCameraPos;
        varying vec3 vLocalViewDirection;
        varying mat3 vModelRotation;
        varying vec2 vUv;

        #define PI 3.14159265359

        // Symmetrical Studio Environment Sample:
        // Mirrors left/right studio softboxes to guarantee 100% balanced lighting across the oval center
        // and eliminates left-right brightness separation.
        vec3 sampleEnv(vec3 dir) {
            vec3 rotatedDir = envRotationMatrix * dir;
            
            float safeY = clamp(rotatedDir.y, -0.9999, 0.9999);
            float v = asin(safeY) / PI + 0.5;

            // Primary horizontal ray
            float phi = atan(rotatedDir.z, rotatedDir.x);
            float u = fract(phi / (2.0 * PI) + 0.5);
            vec3 col = textureLod(envMap, vec2(u, v), 0.0).rgb;

            // Symmetrical mirrored horizontal ray (twin studio softbox balance)
            float phiMir = atan(rotatedDir.z, -rotatedDir.x);
            float uMir = fract(phiMir / (2.0 * PI) + 0.5);
            vec3 colMir = textureLod(envMap, vec2(uMir, v), 0.0).rgb;
            
            // Perfectly balanced bilateral studio radiance
            col = (col + colMir) * 0.5;

            // Subtle studio floor bounce (prevents total blackness while maintaining crisp crystal contrast)
            if (rotatedDir.y < 0.0) {
                float upperV = abs(asin(safeY)) / PI + 0.5;
                vec3 upperCol = textureLod(envMap, vec2(u, upperV), 0.0).rgb;
                vec3 upperColMir = textureLod(envMap, vec2(uMir, upperV), 0.0).rgb;
                vec3 balancedUpper = (upperCol + upperColMir) * 0.5;
                col = mix(col, balancedUpper, 0.18);
            }
            
            return col;
        }

        // Physically Energy-Conserving BVH Ray Tracer:
        // Ray bounce count strictly matches iJewel (bounces = 1 exits directly on the first back facet with 0 internal reflections).
        vec3 traceDiamondBVH(vec3 localIncident, vec3 localNormal, float baseIor, float disp, int bounces) {
            vec3 inRay = refract(localIncident, localNormal, 1.0 / baseIor);
            if (length(inRay) < 0.001) {
                inRay = reflect(localIncident, localNormal);
            }
            inRay = normalize(inRay);

            vec3 currRay = inRay;
            vec3 currPos = vLocalPos + currRay * 0.0001; // Push inside mesh
            
            vec3 accumulatedEnv = vec3(0.0);
            float weight = 1.0;
            float totalDist = 0.0;

            float iorR = baseIor - disp * 1.5;
            float iorG = baseIor;
            float iorB = baseIor + disp * 1.5;

            for (int i = 0; i < 8; i++) {
                if (i >= bounces) break;
                
                uvec4 faceIndices;
                vec3 faceNormal;
                vec3 barycoord;
                float side;
                float dist;
                
                bool hit = bvhIntersectFirstHit(bvh, currPos, currRay, faceIndices, faceNormal, barycoord, side, dist);
                
                if (hit) {
                    vec3 hitPos = currPos + currRay * dist;
                    totalDist += dist;
                    
                    if (dot(faceNormal, currRay) > 0.0) {
                        faceNormal = -faceNormal;
                    }

                    // Multi-spectral exit refractions
                    vec3 exitRayR = refract(currRay, faceNormal, iorR);
                    vec3 exitRayG = refract(currRay, faceNormal, iorG);
                    vec3 exitRayB = refract(currRay, faceNormal, iorB);

                    // If this is the final requested bounce, ray exits directly into the background environment
                    if (i == bounces - 1) {
                        vec3 wExitR = (length(exitRayR) > 0.001) ? exitRayR : currRay;
                        vec3 wExitG = (length(exitRayG) > 0.001) ? exitRayG : currRay;
                        vec3 wExitB = (length(exitRayB) > 0.001) ? exitRayB : currRay;
                        
                        vec3 spectralEnv = vec3(
                            sampleEnv(normalize(vModelRotation * wExitR)).r,
                            sampleEnv(normalize(vModelRotation * wExitG)).g,
                            sampleEnv(normalize(vModelRotation * wExitB)).b
                        );
                        accumulatedEnv += spectralEnv * weight;
                        weight = 0.0;
                        break;
                    }

                    // Internal Fresnel calculations: inside diamond (IOR) -> air (1.0)
                    float cosI = clamp(dot(-currRay, faceNormal), 0.0, 1.0);
                    float sinT2 = (baseIor * baseIor) * (1.0 - cosI * cosI);

                    if (sinT2 >= 1.0) {
                        // Total Internal Reflection (TIR): 100% reflection, 0% transmission
                        currRay = reflect(currRay, faceNormal);
                        currPos = hitPos + currRay * 0.0001;
                    } else {
                        // Partial internal refraction exit & internal bounce
                        float cosT = sqrt(max(1.0 - sinT2, 0.0));
                        float rParl = (1.0 * cosI - baseIor * cosT) / (1.0 * cosI + baseIor * cosT);
                        float rPerp = (baseIor * cosI - 1.0 * cosT) / (baseIor * cosI + 1.0 * cosT);
                        float R = clamp(0.5 * (rParl * rParl + rPerp * rPerp), 0.0, 0.95);
                        float T = 1.0 - R;

                        if (length(exitRayG) > 0.001) {
                            vec3 wExitR = normalize(vModelRotation * ((length(exitRayR) > 0.001) ? exitRayR : exitRayG));
                            vec3 wExitG = normalize(vModelRotation * exitRayG);
                            vec3 wExitB = normalize(vModelRotation * ((length(exitRayB) > 0.001) ? exitRayB : exitRayG));
                            
                            vec3 spectralEnv = vec3(
                                sampleEnv(wExitR).r,
                                sampleEnv(wExitG).g,
                                sampleEnv(wExitB).b
                            );
                            
                            accumulatedEnv += spectralEnv * (T * weight);
                        }

                        // Internal ray continues with R fraction of remaining energy
                        currRay = reflect(currRay, faceNormal);
                        currPos = hitPos + currRay * 0.0001;
                        weight *= R;
                    }
                    
                    if (weight < 0.02) break;
                } else {
                    break;
                }
            }

            vec3 result = accumulatedEnv;
            if (weight > 0.001) {
                vec3 finalWorldExit = normalize(vModelRotation * currRay);
                result += sampleEnv(finalWorldExit) * weight;
            }

            // Physical Beer-Lambert Absorption
            if (absorption > 1.001) {
                result *= exp(-vec3(0.04) * (absorption - 1.0) * totalDist);
            }

            return result;
        }

        void main() {
            // Precise geometric facet normals: WORLD space (eliminates derivative seam spikes)
            vec3 N = normalize(vWorldNormal);
            
            vec3 V = vec3(0.0, 0.0, 1.0);
            vec3 I = -V;
            if (dot(N, I) > 0.0) N = -N;

            // Precise geometric facet normals: LOCAL space
            vec3 localN = normalize(vLocalNormal);

            vec3 localV = normalize(vLocalViewDirection);
            vec3 localI = -localV;
            if (dot(localN, localI) > 0.0) localN = -localN;

            // Exterior Fresnel
            float cosTheta = clamp(dot(V, N), 0.0, 1.0);
            float R0 = pow((ior - 1.0) / (ior + 1.0), 2.0); // ~0.170 for diamond at 2.40
            float F = R0 + (1.0 - R0) * pow(1.0 - cosTheta, 5.0);
            F = clamp(F, 0.0, 0.95);

            vec3 extReflectDir = reflect(I, N);
            vec3 extReflection = sampleEnv(extReflectDir);

            // Orthographic view directions are constant across each mesh, so
            // dFdx/dFdy(localI) are zero. The old two paths traced identical
            // rays, doubling BVH work without adding any antialiasing.
            vec3 internalTransmission = traceDiamondBVH(localI, localN, ior, dispersion, rayBounces);

            // iJewel Gamma Contrast Power Curve:
            // Applied ONLY to internal depth/transmission reflections:
            // Increasing gamma deepens the contrast of inner secondary bounces (dark/darker contrast)
            // while preserving pure white crown table gleams, external Fresnel, and primary highlights!
            internalTransmission = pow(max(internalTransmission, vec3(0.0)), vec3(max(gamma, 0.01)));

            // Color tint, RGB Boost [2.00, 2.00, 2.00], and Environment Intensity
            internalTransmission *= color * rgbBoost * envIntensity;

            // Additive crown specular reflection on top of internal brilliance (iJewel signature look)
            vec3 hdrColor = internalTransmission * (1.0 - F * reflectivity * 0.5) + extReflection * (F * reflectivity * envIntensity * 1.2);

            // Crisp specular gleam on table and bezel crown facets
            float specCrown = pow(max(dot(reflect(I, N), V), 0.0), 24.0);
            hdrColor += extReflection * specCrown * 0.30 * envIntensity;

            // Transmission (direct see-through blend)
            if (transmission > 0.001) {
                vec3 passThrough = sampleEnv(refract(I, N, 1.0 / ior));
                hdrColor = mix(hdrColor, passThrough * envIntensity, transmission);
            }

            // Output linear HDR; the renderer applies ACESFilmicToneMapping once.
            gl_FragColor = vec4(hdrColor, 1.0);
        }
    `;

    return new THREE.ShaderMaterial({
        uniforms: {
            envMap: { value: envMapTexture },
            envRotationMatrix: { value: new THREE.Matrix3() },
            cameraPos: { value: new THREE.Vector3() },
            ior: { value: 2.40 },
            dispersion: { value: 0.0040 },
            absorption: { value: 1.00 },
            envIntensity: { value: 1.50 },
            ambientMin: { value: 0.18 },
            gamma: { value: 1.09 },
            color: { value: new THREE.Color(0xececec) },
            rgbBoost: { value: new THREE.Vector3(2.0, 2.0, 2.0) },
            transmission: { value: 0.00 },
            reflectivity: { value: 0.46 },
            rayBounces: { value: 4 },
            bvh: { value: bvhStruct }
        },
        vertexShader,
        fragmentShader,
        side: THREE.FrontSide,
        transparent: false,
        depthWrite: true,
        glslVersion: THREE.GLSL3
    });
}
