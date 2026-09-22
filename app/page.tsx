'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { HandLandmarker } from '@mediapipe/tasks-vision';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  createDiamondBVH,
  loadDiamondEnvironment,
} from 'realistic-diamond-shader';
import { createDiamondMaterial as createProjectDiamondMaterial } from './project-diamond-material.js';
import {
  detectionSize, solveRingPose, projectedFingerAxis, RingPoseFilter, FingerFitFilter, RingSurfaceConstraint,
  HandednessResolver,
  PLACEMENT_OFFSET_GATE,
  type HandPoint, type RingPlacement as RingModelPlacement,
} from './ring-tracking';
import { SkinMotionTracker, type SkinMotion, type FlowCv } from './skin-motion';
import { findFingerSilhouette, MIN_FINGER_EDGE_CONFIDENCE, sectionMedianWidth, type FingerSilhouette } from './finger-silhouette';
import {
  ringPixelRatio, limitEnvironmentSize, loadOptionalEnvironment,
  createFallbackDiamondMaterial, createFallbackEnvironment,
  fetchWithProgressAndCache, updateDiamondEnvironment,
} from './ring-runtime';
import {
  DEFAULT_INNER_TO_OUTER_DIAMETER,
  OCCLUSION_LENGTH_RATIO,
  OCCLUSION_RADIUS_RATIO,
  bakeMeshScale,
  centerRingOnMetalHole,
  applyStoneAxisBasis,
  createFingerOccluderGeometry,
  geometrySignature,
  isGemMesh,
  isMetalMesh,
  measureRingInnerDiameter,
  normalizeRingToUnitHole,
  readRingModelExtras,
  resolveFingerWidthPx,
  ringWorldScale,
} from './ring-model';
import {
  DEFAULT_RING_SKU,
  RING_STYLES,
  getRingFitConfig,
  ringModelUrl,
  type RingSku,
} from './ring-catalog';
import { PatchMotionTracker, usesLightweightTracking } from './patch-motion';
import { AppIcon, icons } from './ui-icons';
import {
  DEFAULT_CAPTURE_HINT,
  MANUAL_CAPTURE_HINT,
  PoseHoldTimer,
  SETTLE_FRAMES,
  SETTLE_MAX_MS,
  buildPhotoSnapshot,
  captureVideoFrame,
  evaluateCapturePose,
  evaluateManualCaptureGate,
  hasProperRingFit,
  landmarkMotion,
  type PhotoSnapshot,
} from './pose-capture';
import {
  DEFAULT_TRY_ON_FINGER,
  TRY_ON_FINGERS,
  getFingerDef,
  neighborsForFinger,
  type TryOnFinger,
} from './hand-landmarks';

type CameraState = 'idle' | 'requesting' | 'active' | 'error';
type FacingMode = 'environment' | 'user';
type ScreenshotState = 'idle' | 'saving' | 'saved' | 'error';
type UiMode = 'tryon' | 'photo';
type PhotoPhase = 'preview' | 'settling' | 'still';
type PhotoCaptureMode = 'auto' | 'manual';
type ThemeMode = 'dark' | 'light';
type Handedness = 'Left' | 'Right';
type DetectedHand = {
  landmarks: HandPoint[];
  handedness: Handedness | null;
};
type CameraPoint = { x: number; y: number };
type FingerEdgePair = Omit<FingerSilhouette, 'width'> & {
  revision: number;
  /** Along-bone seat t used for this silhouette sample (matches placement). */
  seatT: number;
};
type RingEnvironmentAssets = {
  diamond: THREE.Texture;
  metal: THREE.Texture;
};
type OpenCvApi = FlowCv;
type OpenCvRuntime = OpenCvApi & {
  calledRun?: boolean;
  then?: unknown;
};

declare global {
  interface Window {
    cv?: OpenCvRuntime;
  }
}

const errorMessages: Record<string, string> = {
  NotAllowedError: 'Camera permission was denied.',
  NotFoundError: 'No camera was found on this device.',
  NotReadableError: 'The camera is already being used by another app.',
  SecurityError: 'Camera access requires a secure HTTPS connection.',
  AbortError: 'The camera could not start. Close other camera apps and try again.',
  InvalidStateError: 'Open this page directly in Safari and try again.',
};
const HAND_MODEL_INPUT_SIZE = 640;
// Seat along MCP→PIP (0 = knuckle / palm-ward, 1 = toward tip).
const SEAT_ALONG_MIN = 0.22;
const SEAT_ALONG_MAX = 0.58;
const FINGER_EDGE_CROSS_POSITION = 0.38;
const RING_CAMERA_DISTANCE = 4;
const RING_WORLD_HEIGHT = 2;
const DRACO_DECODER_PATH = '/draco/gltf/';

function resizeRingCamera(camera: THREE.OrthographicCamera, aspect: number) {
  camera.left = -RING_WORLD_HEIGHT * aspect / 2;
  camera.right = RING_WORLD_HEIGHT * aspect / 2;
  camera.top = RING_WORLD_HEIGHT / 2;
  camera.bottom = -RING_WORLD_HEIGHT / 2;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
}

function disposeModelResources(model: THREE.Object3D | null) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  model?.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    geometries.add(object.geometry);
    for (const material of Array.isArray(object.material)
      ? object.material
      : [object.material]) {
      materials.add(material);
      for (const value of Object.values(material)) {
        if (value instanceof THREE.Texture) textures.add(value);
      }
    }
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
  for (const texture of textures) texture.dispose();
}

function placeRingOccluder(
  occluder: THREE.Mesh,
  x: number,
  y: number,
  orientation: THREE.Quaternion,
  width: number,
) {
  occluder.visible = true;
  occluder.position.set(x, y, 0);
  // The mask shares the exact rigid hand pose with the ring. This is essential
  // on a side view: independently composed pitch/roll values can turn the
  // mask and band about different axes.
  occluder.quaternion.copy(orientation);
  // Finger silhouette width does not contract just because its surface turns.
  const occlusionRadius = OCCLUSION_RADIUS_RATIO;
  occluder.scale.set(
    width * occlusionRadius,
    width * OCCLUSION_LENGTH_RATIO,
    width * occlusionRadius,
  );
}

function placeRingContactShadow(
  shadow: THREE.Mesh,
  x: number,
  y: number,
  orientation: THREE.Quaternion,
  width: number,
  surfaceFacing: number,
) {
  shadow.visible = true;
  shadow.position.set(x, y, -0.02);
  shadow.quaternion.copy(orientation);
  // PlaneGeometry is two units wide; this yields a compact soft contact patch
  // under the band and setting rather than a broad artificial dark area.
  shadow.scale.set(width * 0.56, width * 0.23, 1);

  const material = shadow.material;
  if (material instanceof THREE.ShaderMaterial) {
    material.uniforms.opacity.value = THREE.MathUtils.lerp(
      0.065,
      0.16,
      surfaceFacing,
    );
  }
}


export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const photoCanvasRef = useRef<HTMLCanvasElement>(null);
  const ringCanvasRef = useRef<HTMLCanvasElement>(null);
  const pointsCanvasRef = useRef<HTMLCanvasElement>(null);
  const ringRendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const ringSceneRef = useRef<THREE.Scene | null>(null);
  const ringCameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const ringGroupRef = useRef<THREE.Group | null>(null);
  const ringOccluderRef = useRef<THREE.Mesh | null>(null);
  const ringContactShadowRef = useRef<THREE.Mesh | null>(null);
  const ringModelPromiseRef = useRef<Promise<THREE.Group> | null>(null);
  const ringModelRef = useRef<THREE.Group | null>(null);
  const ringLoadFailedRef = useRef(false);
  const ringContextLostRef = useRef(false);
  const ringShaderFailedRef = useRef(false);
  const resourceGenerationRef = useRef(0);
  const ringEnvironmentRef = useRef<THREE.WebGLRenderTarget | null>(null);
  const ringEnvironmentAssetsRef = useRef<RingEnvironmentAssets | null>(null);
  const ringEnvironmentAssetsPromiseRef =
    useRef<Promise<RingEnvironmentAssets> | null>(null);
  const ringDiamondMaterialsRef = useRef<THREE.ShaderMaterial[]>([]);
  const ringDiamondBvhsRef = useRef<Set<{ dispose: () => void }>>(new Set());
  const modelEnabledRef = useRef(true);
  const guidesEnabledRef = useRef(false);
  const ringPoseFilterRef = useRef(new RingPoseFilter());
  const fingerFitFilterRef = useRef(new FingerFitFilter());
  const ringSurfaceRef = useRef(new RingSurfaceConstraint());
  const handednessResolverRef = useRef(new HandednessResolver());
  const skinMotionRef = useRef(new SkinMotionTracker());
  const patchMotionRef = useRef(new PatchMotionTracker());
  const flowCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraRequestRef = useRef(0);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const handLandmarkerPromiseRef = useRef<Promise<HandLandmarker> | null>(null);
  const openCvRef = useRef<OpenCvApi | null>(null);
  const openCvPromiseRef = useRef<Promise<OpenCvApi> | null>(null);
  const handDetectionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sobelFingerEdgesRef = useRef<FingerEdgePair | null>(null);
  const seatCrossPositionRef = useRef(FINGER_EDGE_CROSS_POSITION);
  const [ringSeatAlong, setRingSeatAlong] = useState(FINGER_EDGE_CROSS_POSITION);
  const ringSeatAlongRef = useRef(FINGER_EDGE_CROSS_POSITION);
  const smoothedFingerLandmarksRef = useRef<{ mcp: CameraPoint; pip: CameraPoint; time: number } | null>(null);
  const [targetFinger, setTargetFinger] = useState<TryOnFinger>(DEFAULT_TRY_ON_FINGER);
  const targetFingerRef = useRef<TryOnFinger>(DEFAULT_TRY_ON_FINGER);
  const animationFrameRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastDetectionTimeRef = useRef(0);
  const [cameraState, setCameraState] = useState<CameraState>('requesting');
  const [facingMode, setFacingMode] = useState<FacingMode>('environment');
  const [screenshotState, setScreenshotState] =
    useState<ScreenshotState>('idle');
  const [modelEnabled] = useState(true);
  const [guidesEnabled, setGuidesEnabled] = useState(false);
  const [uiMode, setUiMode] = useState<UiMode>('tryon');
  const uiModeRef = useRef<UiMode>('tryon');
  const [photoPhase, setPhotoPhase] = useState<PhotoPhase>('preview');
  const photoPhaseRef = useRef<PhotoPhase>('preview');
  const [photoCaptureMode, setPhotoCaptureMode] =
    useState<PhotoCaptureMode>('auto');
  const photoCaptureModeRef = useRef<PhotoCaptureMode>('auto');
  const [photoHoldProgress, setPhotoHoldProgress] = useState(0);
  const [photoHint, setPhotoHint] = useState(DEFAULT_CAPTURE_HINT);
  const [photoManualReady, setPhotoManualReady] = useState(false);
  const [photoFlash, setPhotoFlash] = useState(false);
  const photoSnapshotRef = useRef<PhotoSnapshot | null>(null);
  const poseHoldTimerRef = useRef(new PoseHoldTimer());
  const prevCaptureLandmarksRef = useRef<HandPoint[] | null>(null);
  const latestCaptureFrameRef = useRef<{
    landmarks: HandPoint[];
    placement: RingModelPlacement | null;
    handedness: Handedness | null;
    fitCalibrated: boolean;
  } | null>(null);
  const settleTokenRef = useRef(0);
  const settleRafRef = useRef<number | null>(null);
  const lastPhotoHudAtRef = useRef(0);
  const beginPhotoCaptureRef = useRef<() => boolean>(() => false);
  const [theme, setTheme] = useState<ThemeMode>('dark');
  const [ringFlipped, setRingFlipped] = useState(false);
  const ringFlippedRef = useRef(false);
  const [ringStyle, setRingStyle] = useState<RingSku>(DEFAULT_RING_SKU);
  const ringStyleRef = useRef<RingSku>(DEFAULT_RING_SKU);
  const [message, setMessage] = useState('');

  const fallbackDiamondEnvRef = useRef<THREE.Texture | null>(null);
  const [modelLoadingState, setModelLoadingState] = useState<{
    status: 'idle' | 'loading' | 'ready' | 'error';
    progress: number;
    message: string;
  }>({
    status: 'idle',
    progress: 0,
    message: '',
  });

  const zoomRef = useRef(1);
  const hasHardwareZoomRef = useRef(false);
  const zoomRangeRef = useRef<{ min: number; max: number; step: number }>({ min: 1, max: 3, step: 0.1 });
  const pinchStartDistRef = useRef<number | null>(null);
  const pinchStartZoomRef = useRef<number>(1);

  const applyZoom = useCallback(async (targetZoom: number) => {
    const clamped = Math.max(1, Math.min(3, Math.round(targetZoom * 10) / 10));
    zoomRef.current = clamped;
    const track = streamRef.current?.getVideoTracks()[0];
    if (track && hasHardwareZoomRef.current) {
      try {
        await (track as any).applyConstraints({
          advanced: [{ zoom: clamped }],
        });
      } catch (err) {
        console.warn('Hardware zoom could not be applied; using digital crop zoom.', err);
        hasHardwareZoomRef.current = false;
      }
    }
  }, []);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      pinchStartDistRef.current = Math.hypot(dx, dy);
      pinchStartZoomRef.current = zoomRef.current;
    }
  }, []);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (e.touches.length === 2 && pinchStartDistRef.current !== null) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.hypot(dx, dy);
      const ratio = dist / Math.max(pinchStartDistRef.current, 1);
      const targetZoom = Math.max(1, Math.min(3, pinchStartZoomRef.current * ratio));
      void applyZoom(targetZoom);
    }
  }, [applyZoom]);

  const handleTouchEnd = useCallback(() => {
    pinchStartDistRef.current = null;
  }, []);


  const initRingScene = useCallback(() => {
    const canvas = ringCanvasRef.current;
    if (!canvas || ringContextLostRef.current) return null;

    let renderer = ringRendererRef.current;
    let scene = ringSceneRef.current;
    let camera = ringCameraRef.current;
    let occluder = ringOccluderRef.current;
    let contactShadow = ringContactShadowRef.current;

    if (renderer && scene && camera && occluder && contactShadow) {
      return { renderer, scene, camera, occluder, contactShadow };
    }

    const width = canvas.clientWidth || window.innerWidth || 640;
    const height = canvas.clientHeight || window.innerHeight || 480;

    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: navigator.maxTouchPoints === 0,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.debug.onShaderError = (gl, program, vertex, fragment) => {
      console.error('Ring shader failed.', gl.getProgramInfoLog(program),
        gl.getShaderInfoLog(vertex), gl.getShaderInfoLog(fragment));
      ringShaderFailedRef.current = true;
    };

    const pixelRatio = ringPixelRatio(width, height, window.devicePixelRatio, navigator.maxTouchPoints > 0);
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height, false);

    scene = new THREE.Scene();
    camera = new THREE.OrthographicCamera(
      -width / height,
      width / height,
      1,
      -1,
      0.1,
      100,
    );
    camera.position.set(0, 0, RING_CAMERA_DISTANCE);
    camera.lookAt(0, 0, 0);
    resizeRingCamera(camera, width / height);

    const environmentGenerator = new THREE.PMREMGenerator(renderer);
    const roomEnvironment = new RoomEnvironment();
    const environment = environmentGenerator.fromScene(roomEnvironment, 0.04);
    scene.environment = environment.texture;
    roomEnvironment.dispose();
    environmentGenerator.dispose();
    ringEnvironmentRef.current = environment;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x4d2d10, 2.5));
    const keyLight = new THREE.DirectionalLight(0xfff2d5, 4.5);
    keyLight.position.set(-1, -2, 4);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0xffd38a, 2.4);
    rimLight.position.set(2, 1, 2);
    scene.add(rimLight);

    const occlusionMaterial = new THREE.MeshBasicMaterial({
      colorWrite: false,
      depthTest: true,
      depthWrite: true,
    });
    occluder = new THREE.Mesh(
      createFingerOccluderGeometry(),
      occlusionMaterial,
    );
    occluder.renderOrder = -100;
    occluder.frustumCulled = false;
    occluder.visible = false;
    scene.add(occluder);

    const contactShadowMaterial = new THREE.ShaderMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: {
        opacity: { value: 0.12 },
      },
      vertexShader: `
        varying vec2 vUv;

        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        precision highp float;
        uniform float opacity;
        varying vec2 vUv;

        void main() {
          vec2 point = vUv * 2.0 - 1.0;
          float radius = dot(point, point);
          float core = exp(-4.8 * radius);
          float edge = 1.0 - smoothstep(0.6, 1.0, radius);
          gl_FragColor = vec4(0.0, 0.0, 0.0, opacity * core * edge);
        }
      `,
    });
    contactShadow = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      contactShadowMaterial,
    );
    contactShadow.renderOrder = 0;
    contactShadow.frustumCulled = false;
    contactShadow.visible = false;
    scene.add(contactShadow);

    ringRendererRef.current = renderer;
    ringSceneRef.current = scene;
    ringCameraRef.current = camera;
    ringOccluderRef.current = occluder;
    ringContactShadowRef.current = contactShadow;

    return { renderer, scene, camera, occluder, contactShadow };
  }, []);

  const loadRingEnvironments = useCallback(() => {
    if (!ringEnvironmentAssetsPromiseRef.current) {
      const generation = resourceGenerationRef.current;
      ringEnvironmentAssetsPromiseRef.current = Promise.allSettled([
        loadOptionalEnvironment(() => loadDiamondEnvironment('/diamond_project.exr', { type: 'exr' })),
        loadOptionalEnvironment(() => loadDiamondEnvironment('/metal_env.exr', { type: 'exr' })),
      ])
        .then(([diamondResult, metalResult]) => {
          if (
            generation !== resourceGenerationRef.current ||
            diamondResult.status === 'rejected' ||
            metalResult.status === 'rejected'
          ) {
            if (diamondResult.status === 'fulfilled') diamondResult.value.dispose();
            if (metalResult.status === 'fulfilled') metalResult.value.dispose();
            if (diamondResult.status === 'rejected') throw diamondResult.reason;
            if (metalResult.status === 'rejected') throw metalResult.reason;
            throw new DOMException('Environment load was cancelled.', 'AbortError');
          }
          const diamond = diamondResult.value;
          const metal = metalResult.value;
          if (navigator.maxTouchPoints > 0) {
            limitEnvironmentSize(diamond, 1024);
            limitEnvironmentSize(metal, 1024);
          }
          diamond.mapping = THREE.EquirectangularReflectionMapping;
          diamond.wrapS = THREE.RepeatWrapping;
          diamond.wrapT = THREE.ClampToEdgeWrapping;
          if (diamond.type === THREE.HalfFloatType) {
            diamond.minFilter = THREE.LinearFilter;
            diamond.magFilter = THREE.LinearFilter;
          }
          diamond.generateMipmaps = false;
          diamond.needsUpdate = true;

          const assets = { diamond, metal };
          ringEnvironmentAssetsRef.current = assets;

          // Upgrade diamond shaders dynamically
          if (ringDiamondMaterialsRef.current.length > 0) {
            updateDiamondEnvironment(ringDiamondMaterialsRef.current, diamond);
            if (fallbackDiamondEnvRef.current) {
              fallbackDiamondEnvRef.current.dispose();
              fallbackDiamondEnvRef.current = null;
            }
          }

          // Upgrade metal environment on the scene
          const renderer = ringRendererRef.current;
          const scene = ringSceneRef.current;
          const camera = ringCameraRef.current;
          if (renderer && scene) {
            const metalEnvironmentGenerator = new THREE.PMREMGenerator(renderer);
            metalEnvironmentGenerator.compileEquirectangularShader();
            const metalEnvironment = metalEnvironmentGenerator.fromEquirectangular(metal);
            metalEnvironmentGenerator.dispose();

            const previousEnvironment = ringEnvironmentRef.current;
            scene.environment = metalEnvironment.texture;
            ringEnvironmentRef.current = metalEnvironment;
            previousEnvironment?.dispose();

            for (const material of ringDiamondMaterialsRef.current) {
              const cameraPosition = material.uniforms.cameraPos?.value;
              if (cameraPosition instanceof THREE.Vector3 && camera) {
                cameraPosition.copy(camera.position);
              }
            }
            if (ringGroupRef.current?.visible && camera) {
              renderer.render(scene, camera);
            }
          }

          return assets;
        })
        .catch((error) => {
          if (generation === resourceGenerationRef.current) {
            ringEnvironmentAssetsPromiseRef.current = null;
          }
          throw error;
        });
    }

    return ringEnvironmentAssetsPromiseRef.current;
  }, []);

  const loadRingModel = useCallback((sku: RingSku = ringStyleRef.current) => {
    const modelUrl = ringModelUrl(sku);
    const { config } = getRingFitConfig(sku);

    if (!ringModelPromiseRef.current) {
      const generation = resourceGenerationRef.current;
      setModelLoadingState({ status: 'loading', progress: 10, message: `Preparing ${sku.toUpperCase()}…` });

      // Trigger high-res HDR environment maps in background without blocking initial ring render
      void loadRingEnvironments().catch((error) => {
        console.warn('HDR environment load skipped; using studio lighting.', error);
      });

      const loader = new GLTFLoader();
      const dracoLoader = new DRACOLoader();
      dracoLoader.setDecoderPath(DRACO_DECODER_PATH);
      loader.setDRACOLoader(dracoLoader);

      const gltfPromise = fetchWithProgressAndCache(modelUrl, (ratio) => {
        if (generation !== resourceGenerationRef.current) return;
        const progress = Math.round(10 + ratio * 65);
        setModelLoadingState({
          status: 'loading',
          progress,
          message: ratio < 1
            ? `Downloading ${sku.toUpperCase()}… ${Math.round(ratio * 100)}%`
            : 'Parsing 3D geometry…',
        });
      })
        .then((buffer) => loader.parseAsync(buffer, ''))
        .finally(() => {
          dracoLoader.dispose();
        });

      ringModelPromiseRef.current = gltfPromise
        .then(async (gltf) => {
          if (generation !== resourceGenerationRef.current) {
            disposeModelResources(gltf.scene);
            throw new DOMException('Ring load was cancelled.', 'AbortError');
          }

          setModelLoadingState({ status: 'loading', progress: 78, message: 'Preparing metal band…' });

          const model = new THREE.Group();
          ringModelRef.current = model;
          model.add(gltf.scene);
          model.updateMatrixWorld(true);

          // Center on the metal band in XYZ so the unit hole shares the occluder axis.
          centerRingOnMetalHole(model, gltf.scene);

          const extras = readRingModelExtras(gltf.scene);
          const measuredInner = measureRingInnerDiameter(
            model,
            DEFAULT_INNER_TO_OUTER_DIAMETER,
          );
          const rawInnerDiameter =
            typeof config.innerDiameterHint === 'number' && config.innerDiameterHint > 0
              ? config.innerDiameterHint
              : typeof extras.innerDiameter === 'number'
                ? extras.innerDiameter
                : measuredInner;
          normalizeRingToUnitHole(model, rawInnerDiameter, config.fitScale);
          // Re-seat after scale so numerical drift / gem AABB cannot pull the hole off origin.
          centerRingOnMetalHole(model, model, config.offset || {});
          applyStoneAxisBasis(model, extras.stoneAxis);
          centerRingOnMetalHole(model, model);
          model.userData.wearClearance = config.wearClearance;
          model.userData.sku = sku;
          model.userData.stoneAxis = '-Z';
          if (extras.stoneAxis) model.userData.stoneAxisSource = extras.stoneAxis;

          const metalTint = extras.metalTint || '#c2c2c3';
          const metalMaterial = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(metalTint),
            metalness: 1,
            roughness: 0.08,
            clearcoat: 0.35,
            clearcoatRoughness: 0.12,
            envMapIntensity: 1.55,
            side: THREE.FrontSide,
            depthWrite: true,
          });
          const diamondMaterials: THREE.ShaderMaterial[] = [];
          const diamondBvhs = new Set<{ dispose: () => void }>();
          ringDiamondMaterialsRef.current = diamondMaterials;
          ringDiamondBvhsRef.current = diamondBvhs;
          const supersededMaterials = new Set<THREE.Material>();
          let usesMetalMaterial = false;
          const gemMeshes: THREE.Mesh[] = [];
          const placeholderStone = createFallbackDiamondMaterial();

          model.traverse((object) => {
            if (!(object instanceof THREE.Mesh)) return;

            const materials = Array.isArray(object.material)
              ? object.material
              : [object.material];

            if (isGemMesh(object)) {
              bakeMeshScale(object);
              for (const material of materials) {
                supersededMaterials.add(material);
              }
              object.material = placeholderStone;
              gemMeshes.push(object);
              return;
            }

            if (isMetalMesh(object)) {
              for (const material of materials) {
                supersededMaterials.add(material);
              }
              object.material = metalMaterial;
              usesMetalMaterial = true;
              return;
            }

            for (const material of materials) {
              material.side = THREE.FrontSide;
              material.depthWrite = true;
            }
          });

          const retainedMaterials = new Set<THREE.Material>();
          model.traverse((object) => {
            if (!(object instanceof THREE.Mesh)) return;
            const materials = Array.isArray(object.material)
              ? object.material
              : [object.material];
            for (const material of materials) retainedMaterials.add(material);
          });
          for (const material of supersededMaterials) {
            if (!retainedMaterials.has(material)) material.dispose();
          }
          if (!usesMetalMaterial) metalMaterial.dispose();

          // Attach metal band early so try-on can start before diamond BVHs finish.
          const scene = ringSceneRef.current;
          const renderer = ringRendererRef.current;
          const camera = ringCameraRef.current;
          if (scene && !ringGroupRef.current) {
            const group = new THREE.Group();
            group.renderOrder = 1;
            group.visible = false;
            group.add(model);
            scene.add(group);
            ringGroupRef.current = group;
          }
          if (renderer && scene && camera) {
            try {
              renderer.compile(scene, camera);
            } catch {
              // Ignore pre-warm compile errors
            }
          }

          setModelLoadingState({
            status: 'loading',
            progress: 88,
            message: gemMeshes.length
              ? 'Generating diamond optics…'
              : '3D ring ready',
          });

          let initialDiamondEnv = ringEnvironmentAssetsRef.current?.diamond;
          if (!initialDiamondEnv) {
            initialDiamondEnv = createFallbackEnvironment();
            fallbackDiamondEnvRef.current = initialDiamondEnv;
          }

          const touchDevice = navigator.maxTouchPoints > 0;
          const maxLeafTriangles = touchDevice ? 4 : 1;
          const sharedBvhBySignature = new Map<
            string,
            { bvh: { dispose: () => void }; material: THREE.ShaderMaterial }
          >();
          const yieldToMain = () =>
            new Promise<void>((resolve) => {
              if (typeof requestAnimationFrame === 'function') {
                requestAnimationFrame(() => resolve());
              } else {
                setTimeout(resolve, 0);
              }
            });

          for (let i = 0; i < gemMeshes.length; i++) {
            if (generation !== resourceGenerationRef.current) break;
            const object = gemMeshes[i];
            const signature = geometrySignature(object.geometry);
            let diamondMaterial: THREE.Material;
            try {
              const shared = sharedBvhBySignature.get(signature);
              if (shared) {
                diamondMaterial = shared.material;
              } else {
                const { bvh } = createDiamondBVH(object.geometry, {
                  maxLeafTriangles,
                });
                diamondBvhs.add(bvh);
                const shader = createProjectDiamondMaterial(
                  initialDiamondEnv,
                  bvh,
                ) as THREE.ShaderMaterial;
                diamondMaterials.push(shader);
                sharedBvhBySignature.set(signature, { bvh, material: shader });
                diamondMaterial = shader;
              }
            } catch (error) {
              console.warn(
                'Diamond ray tracing unavailable; using reflective stone material.',
                error,
              );
              diamondMaterial = createFallbackDiamondMaterial();
            }
            diamondMaterial.depthWrite = true;
            object.material = diamondMaterial;

            if (i % 4 === 3) {
              setModelLoadingState({
                status: 'loading',
                progress: Math.round(88 + ((i + 1) / gemMeshes.length) * 10),
                message: 'Generating diamond optics…',
              });
              await yieldToMain();
            }
          }

          if (generation !== resourceGenerationRef.current) {
            disposeModelResources(model);
            throw new DOMException('Ring load was cancelled.', 'AbortError');
          }

          // Drop the shared placeholder once every gem has a real material.
          let placeholderStillUsed = false;
          model.traverse((object) => {
            if (object instanceof THREE.Mesh && object.material === placeholderStone) {
              placeholderStillUsed = true;
            }
          });
          if (!placeholderStillUsed) placeholderStone.dispose();

          ringDiamondMaterialsRef.current = diamondMaterials;
          ringDiamondBvhsRef.current = diamondBvhs;

          if (renderer && scene && camera) {
            try {
              renderer.compile(scene, camera);
            } catch {
              // Ignore pre-warm compile errors
            }
          }

          setModelLoadingState({ status: 'ready', progress: 100, message: '3D ring ready' });
          setTimeout(() => {
            setModelLoadingState((prev) => (prev.status === 'ready' ? { ...prev, status: 'idle' } : prev));
          }, 1200);

          return model;
        })
        .catch((error) => {
          if (generation === resourceGenerationRef.current) {
            disposeModelResources(ringModelRef.current);
            ringModelRef.current = null;
            for (const bvh of ringDiamondBvhsRef.current) bvh.dispose();
            ringDiamondBvhsRef.current.clear();
            ringDiamondMaterialsRef.current = [];
            ringModelPromiseRef.current = null;
            ringLoadFailedRef.current = true;
            setModelLoadingState({ status: 'error', progress: 0, message: 'Unable to load ring model' });
          }
          throw error;
        });
    }

    return ringModelPromiseRef.current;
  }, [loadRingEnvironments]);

  const disposeRingModel = useCallback(() => {
    resourceGenerationRef.current += 1;
    // Preloaded models also own resources before they are added to the scene.
    disposeModelResources(ringModelRef.current);
    ringModelRef.current = null;
    for (const bvh of ringDiamondBvhsRef.current) bvh.dispose();
    ringDiamondBvhsRef.current.clear();
    ringDiamondMaterialsRef.current = [];
    fallbackDiamondEnvRef.current?.dispose();
    fallbackDiamondEnvRef.current = null;
    ringOccluderRef.current?.geometry.dispose();
    const occluderMaterial = ringOccluderRef.current?.material;
    if (Array.isArray(occluderMaterial)) {
      for (const material of occluderMaterial) material.dispose();
    } else {
      occluderMaterial?.dispose();
    }
    ringContactShadowRef.current?.geometry.dispose();
    const contactShadowMaterial = ringContactShadowRef.current?.material;
    if (Array.isArray(contactShadowMaterial)) {
      for (const material of contactShadowMaterial) material.dispose();
    } else {
      contactShadowMaterial?.dispose();
    }
    ringEnvironmentRef.current?.dispose();
    ringEnvironmentAssetsRef.current?.diamond.dispose();
    ringEnvironmentAssetsRef.current?.metal.dispose();
    ringRendererRef.current?.dispose();
    ringGroupRef.current = null;
    ringOccluderRef.current = null;
    ringContactShadowRef.current = null;
    ringRendererRef.current = null;
    ringSceneRef.current = null;
    ringCameraRef.current = null;
    ringEnvironmentRef.current = null;
    ringEnvironmentAssetsRef.current = null;
    ringEnvironmentAssetsPromiseRef.current = null;
    ringModelPromiseRef.current = null;
  }, []);

  const renderRingModel = useCallback(
    (placement: RingModelPlacement | null) => {
      const canvas = ringCanvasRef.current;
      if (!canvas || ringContextLostRef.current) return;

      const ringSceneCtx = initRingScene();
      if (!ringSceneCtx) return;
      const { renderer, scene, camera, occluder, contactShadow } = ringSceneCtx;
      const ring = ringGroupRef.current;

      if (!placement || !modelEnabledRef.current) {
        if (ring) ring.visible = false;
        if (occluder) occluder.visible = false;
        if (contactShadow) contactShadow.visible = false;
        renderer.render(scene, camera);
        return;
      }

      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (!width || !height) return;

      const pixelRatio = ringPixelRatio(width, height, window.devicePixelRatio, navigator.maxTouchPoints > 0);
      const targetWidth = Math.round(width * pixelRatio);
      const targetHeight = Math.round(height * pixelRatio);
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        renderer.setPixelRatio(pixelRatio);
        renderer.setSize(width, height, false);
        resizeRingCamera(camera, width / height);
      }

      if (!ring) {
        if (ringLoadFailedRef.current) return;
        const loadingScene = scene;
        void loadRingModel(ringStyleRef.current)
          .then((model) => {
            if (ringSceneRef.current !== loadingScene) return;

            let group = ringGroupRef.current;
            if (!group) {
              group = new THREE.Group();
              group.renderOrder = 1;
              group.visible = false;
              ringSceneRef.current.add(group);
              ringGroupRef.current = group;
            } else {
              while (group.children.length > 0) {
                group.remove(group.children[0]);
              }
            }
            group.add(model);
          })
          .catch((error) => {
            if (ringSceneRef.current !== loadingScene) return;
            console.error('Unable to load the ring model.', error);
            setMessage('Unable to load the ring. Stop and restart the camera to try again.');
          });
        return;
      }


      if (ringShaderFailedRef.current) {
        ringShaderFailedRef.current = false;
        const shaders = new Set(ringDiamondMaterialsRef.current);
        const fallback = createFallbackDiamondMaterial();
        let replaced = false;
        ring.traverse(object => {
          if (object instanceof THREE.Mesh && shaders.has(object.material)) {
            object.material = fallback;
            replaced = true;
          }
        });
        if (!replaced) fallback.dispose();
        for (const material of shaders) material.dispose();
        ringDiamondMaterialsRef.current = [];
        for (const bvh of ringDiamondBvhsRef.current) bvh.dispose();
        ringDiamondBvhsRef.current.clear();
      }

      const worldUnitsPerPixel = RING_WORLD_HEIGHT / height;
      const worldX = (placement.x - width / 2) * worldUnitsPerPixel;
      const worldY = (height / 2 - placement.y) * worldUnitsPerPixel;
      // Exact finger skin span in world units (no wear clearance).
      const fingerWorld = placement.width * worldUnitsPerPixel;

      ring.visible = true;
      if (occluder) {
        placeRingOccluder(
          occluder,
          worldX,
          worldY,
          placement.orientation,
          fingerWorld,
        );
      }
      if (contactShadow) {
        placeRingContactShadow(
          contactShadow,
          worldX,
          worldY,
          placement.orientation,
          fingerWorld,
          placement.surfaceFacing,
        );
      }
      ring.position.set(worldX, worldY, 0);
      ring.quaternion.copy(placement.orientation);
      const model = ring.children[0];
      // All pose rotation belongs to the ring group so the model, depth mask,
      // and shadow use one hand-locked coordinate frame. Optional Flip spins
      // 180° about the finger axis to swap head/shank as a user override only.
      model?.rotation.set(0, ringFlippedRef.current ? Math.PI : 0, 0, 'XYZ');
      // Unit-hole GLB: scale = finger diameter × wear clearance so metal kisses skin.
      const wearClearance =
        typeof model?.userData?.wearClearance === 'number'
          ? model.userData.wearClearance
          : undefined;
      ring.scale.setScalar(ringWorldScale(fingerWorld, wearClearance));
      for (const material of ringDiamondMaterialsRef.current) {
        const cameraPosition = material.uniforms.cameraPos?.value;
        if (cameraPosition instanceof THREE.Vector3) {
          cameraPosition.copy(camera.position);
        }
      }
      renderer.render(scene, camera);
    },
    [loadRingEnvironments, loadRingModel],
  );

  const clearPoints = useCallback(() => {
    const canvas = pointsCanvasRef.current;
    const context = canvas?.getContext('2d');

    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  const clearSobelOverlay = useCallback(() => {
    sobelFingerEdgesRef.current = null;
    seatCrossPositionRef.current = ringSeatAlongRef.current;
    smoothedFingerLandmarksRef.current = null;
  }, []);

  // Clear size lock / pose so a new hand (or Refit) can calibrate from scratch.
  const resetRingFit = useCallback(() => {
    fingerFitFilterRef.current.reset();
    ringSurfaceRef.current.reset();
    ringPoseFilterRef.current.reset();
    handednessResolverRef.current.reset();
    smoothedFingerLandmarksRef.current = null;
    clearSobelOverlay();
    renderRingModel(null);
  }, [clearSobelOverlay, renderRingModel]);

  const stopHandTracking = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    const preview = previewCanvasRef.current;
    preview?.getContext('2d')?.clearRect(0, 0, preview.width, preview.height);
    lastVideoTimeRef.current = -1;
    lastDetectionTimeRef.current = 0;
    ringPoseFilterRef.current.reset();
    fingerFitFilterRef.current.reset();
    ringSurfaceRef.current.reset();
    handednessResolverRef.current.reset();
    smoothedFingerLandmarksRef.current = null;
    skinMotionRef.current.reset();
    patchMotionRef.current.reset();
    clearPoints();
    clearSobelOverlay();
    renderRingModel(null);
  }, [clearPoints, clearSobelOverlay, renderRingModel]);

  const getHandLandmarker = useCallback(() => {
    if (!handLandmarkerPromiseRef.current) {
      const generation = resourceGenerationRef.current;
      handLandmarkerPromiseRef.current = (async () => {
        const { FilesetResolver, HandLandmarker } = await import(
          '@mediapipe/tasks-vision'
        );
        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/mediapipe/models/hand_landmarker.task',
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.5,
        });

        if (generation !== resourceGenerationRef.current) {
          landmarker.close();
          throw new DOMException('Hand tracking load was cancelled.', 'AbortError');
        }
        handLandmarkerRef.current = landmarker;
        return landmarker;
      })().catch((error) => {
        if (generation === resourceGenerationRef.current) {
          handLandmarkerPromiseRef.current = null;
        }
        throw error;
      });
    }

    return handLandmarkerPromiseRef.current;
  }, []);

  const getOpenCv = useCallback(() => {
    if (!openCvPromiseRef.current) {
      const generation = resourceGenerationRef.current;
      openCvPromiseRef.current = new Promise<OpenCvApi>((resolve, reject) => {
        const existingScript = document.querySelector<HTMLScriptElement>(
          'script[data-opencv-runtime]',
        );
        const script = existingScript ?? document.createElement('script');
        let timer: number | undefined;
        const cleanup = () => {
          window.clearTimeout(timer);
          script.removeEventListener('error', handleError);
        };
        const handleError = () => {
          cleanup();
          // A failed script must be removed so a later attempt can fetch it.
          script.remove();
          reject(new Error('Unable to load OpenCV.'));
        };
        const startedAt = performance.now();
        const checkRuntime = () => {
          const runtime = window.cv;
          if (generation !== resourceGenerationRef.current) {
            cleanup();
            reject(new DOMException('OpenCV load was cancelled.', 'AbortError'));
            return;
          }
          if (runtime?.calledRun && typeof runtime.Mat === 'function') {
            cleanup();
            delete runtime.then;
            resolve(runtime);
            return;
          }
          if (performance.now() - startedAt >= 30000) {
            handleError();
            return;
          }
          timer = window.setTimeout(checkRuntime, 25);
        };
        script.addEventListener('error', handleError, { once: true });
        if (!existingScript && !window.cv) {
          script.src = '/opencv/opencv.js';
          script.async = true;
          script.dataset.opencvRuntime = 'true';
          document.head.appendChild(script);
        }
        // Poll immediately: a reused script may already have fired its load event.
        checkRuntime();
      })
        .then((openCv) => {
          if (generation === resourceGenerationRef.current) openCvRef.current = openCv;
          return openCv;
        })
        .catch((error) => {
          if (generation === resourceGenerationRef.current) openCvPromiseRef.current = null;
          throw error;
        });
    }

    return openCvPromiseRef.current;
  }, []);

  const updateHorizontalSobel = useCallback(
    (source: HTMLCanvasElement, landmarks: HandPoint[] | null) => {
      const finger = getFingerDef(targetFingerRef.current);
      if (!landmarks?.[finger.pip] || !landmarks?.[finger.mcp] || !source.width || !source.height) {
        sobelFingerEdgesRef.current = null;
        return;
      }

      // Read the exact captured frame used for landmarks and optical flow.
      // Reading the live video again could measure a later, already moved hand.
      const size = { width: source.width, height: source.height };
      const context = source.getContext('2d');
      if (!context) return;
      const point = (i: number) => ({ x: landmarks[i].x * size.width, y: landmarks[i].y * size.height });
      const neighbors = neighborsForFinger(finger.name)
        .filter((n) => landmarks[n.mcp] && landmarks[n.pip])
        .map((n) => ({ base: point(n.mcp), tip: point(n.pip) }));
      const image = context.getImageData(0, 0, size.width, size.height);
      const base = point(finger.mcp);
      const tip = point(finger.pip);
      // Slider owns along-finger seat; measure edges at the same station.
      const measureT = Math.min(
        SEAT_ALONG_MAX,
        Math.max(SEAT_ALONG_MIN, ringSeatAlongRef.current),
      );
      let edges = findFingerSilhouette(image, base, tip, neighbors, measureT);
      if (edges) {
        seatCrossPositionRef.current = measureT;
        const normalize = (p: CameraPoint) => ({ x: p.x / size.width, y: p.y / size.height });
        sobelFingerEdgesRef.current = {
          ...edges,
          left: normalize(edges.left),
          right: normalize(edges.right),
          revision: performance.now(),
          seatT: measureT,
          sections: edges.sections.map(section => ({
            left: normalize(section.left),
            right: normalize(section.right),
          })),
        };
      } else {
        sobelFingerEdgesRef.current = null;
      }
    }, [],
  );

  const drawPoints = useCallback(
    (hands: DetectedHand[], mode: FacingMode, imageMotion?: SkinMotion | null) => {
      const canvas = pointsCanvasRef.current;
      const video = videoRef.current;

      if (!canvas || !video || !video.videoWidth || !video.videoHeight) {
        return;
      }

      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const pixelRatio = ringPixelRatio(width, height, window.devicePixelRatio, navigator.maxTouchPoints > 0);
      const targetWidth = Math.round(width * pixelRatio);
      const targetHeight = Math.round(height * pixelRatio);

      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }

      const context = canvas.getContext('2d');
      if (!context) return;

      if (video.style.transform) {
        video.style.removeProperty('transform');
      }

      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, width, height);
      const showGuides = guidesEnabledRef.current;

      const scale = Math.max(
        width / video.videoWidth,
        height / video.videoHeight,
      );
      const renderedWidth = video.videoWidth * scale;
      const renderedHeight = video.videoHeight * scale;
      const offsetX = (width - renderedWidth) / 2;
      const offsetY = (height - renderedHeight) / 2;
      let ringPlacement: RingModelPlacement | null = null;

      for (const hand of hands) {
        const landmarks = hand.landmarks;
        const finger = getFingerDef(targetFingerRef.current);
        const mcpIdx = finger.mcp;
        const pipIdx = finger.pip;

        const screenLandmarks = landmarks.map((point) => ({
          x:
            offsetX +
            (mode === 'user' ? 1 - point.x : point.x) * renderedWidth,
          y: offsetY + point.y * renderedHeight,
        }));
        const pointMcp = screenLandmarks[mcpIdx];
        const pointPip = screenLandmarks[pipIdx];

        if (!pointMcp || !pointPip) {
          smoothedFingerLandmarksRef.current = null;
          continue;
        }

        const now = performance.now();
        const prevSmoothed = smoothedFingerLandmarksRef.current;
        let pMcp = pointMcp;
        let pPip = pointPip;
        if (prevSmoothed && now - prevSmoothed.time < 160) {
          const dt = Math.max(0.001, Math.min(0.1, (now - prevSmoothed.time) / 1000));
          const stepMcp = Math.hypot(pointMcp.x - prevSmoothed.mcp.x, pointMcp.y - prevSmoothed.mcp.y);
          const stepPip = Math.hypot(pointPip.x - prevSmoothed.pip.x, pointPip.y - prevSmoothed.pip.y);
          const hzMcp = stepMcp > 5 ? 26 : THREE.MathUtils.lerp(4.5, 26, Math.max(0, stepMcp - 1) / 4);
          const hzPip = stepPip > 5 ? 26 : THREE.MathUtils.lerp(4.5, 26, Math.max(0, stepPip - 1) / 4);
          const aMcp = 1 - Math.exp(-2 * Math.PI * hzMcp * dt);
          const aPip = 1 - Math.exp(-2 * Math.PI * hzPip * dt);
          pMcp = {
            x: prevSmoothed.mcp.x + (pointMcp.x - prevSmoothed.mcp.x) * aMcp,
            y: prevSmoothed.mcp.y + (pointMcp.y - prevSmoothed.mcp.y) * aMcp,
          };
          pPip = {
            x: prevSmoothed.pip.x + (pointPip.x - prevSmoothed.pip.x) * aPip,
            y: prevSmoothed.pip.y + (pointPip.y - prevSmoothed.pip.y) * aPip,
          };
        }
        smoothedFingerLandmarksRef.current = { mcp: pMcp, pip: pPip, time: now };

        const fingerEdges = sobelFingerEdgesRef.current;
        const seatT = Math.min(
          SEAT_ALONG_MAX,
          Math.max(SEAT_ALONG_MIN, ringSeatAlongRef.current),
        );
        seatCrossPositionRef.current = seatT;
        const ringCenter = {
          x: pMcp.x + (pPip.x - pMcp.x) * seatT,
          y: pMcp.y + (pPip.y - pMcp.y) * seatT,
        };
        const fingerScreenX = pPip.x - pMcp.x;
        const fingerScreenY = pPip.y - pMcp.y;
        let perpendicularX = -fingerScreenY;
        let perpendicularY = fingerScreenX;
        const perpendicularLength = Math.hypot(
          perpendicularX,
          perpendicularY,
        );

        if (!perpendicularLength) continue;

        perpendicularX /= perpendicularLength;
        perpendicularY /= perpendicularLength;

        const stabilizedScreenLandmarks = [...screenLandmarks];
        stabilizedScreenLandmarks[mcpIdx] = pMcp;
        stabilizedScreenLandmarks[pipIdx] = pPip;

        const surfacePose = solveRingPose(
          landmarks, stabilizedScreenLandmarks, hand.handedness, mode === 'user',
          ringPoseFilterRef.current.current()?.orientation,
          { mcp: mcpIdx, pip: pipIdx },
        );
        if (!surfacePose) continue;
        let edgeSample: { width: number; offset: number; revision: number; confidence: number } | undefined;

        if (fingerEdges) {
          const toScreenPoint = (point: CameraPoint) => ({
            x:
              offsetX +
              (mode === 'user' ? 1 - point.x : point.x) * renderedWidth,
            y: offsetY + point.y * renderedHeight,
          });
          const detectedLeftEdge = toScreenPoint(fingerEdges.left);
          const detectedRightEdge = toScreenPoint(fingerEdges.right);
          const screenSilhouette = {
            left: detectedLeftEdge,
            right: detectedRightEdge,
            width: Math.hypot(
              detectedRightEdge.x - detectedLeftEdge.x,
              detectedRightEdge.y - detectedLeftEdge.y,
            ),
            support: fingerEdges.support,
            confidence: fingerEdges.confidence,
            reason: fingerEdges.reason,
            sections: fingerEdges.sections.map((section) => ({
              left: toScreenPoint(section.left),
              right: toScreenPoint(section.right),
            })),
          };
          // Edge-to-edge skin span: section-median width (not a single noisy pair).
          const detectedWidth = sectionMedianWidth(screenSilhouette);
          const boneLengthPx = Math.hypot(pPip.x - pMcp.x, pPip.y - pMcp.y);
          const clampedWidth = resolveFingerWidthPx(detectedWidth, boneLengthPx);

          if (clampedWidth && performance.now() - fingerEdges.revision < 120) {
            // Lateral offset of silhouette mid vs bone seat (gated later).
            edgeSample = {
              width: clampedWidth,
              offset: ((detectedLeftEdge.x + detectedRightEdge.x) / 2 - ringCenter.x) * perpendicularX
                + ((detectedLeftEdge.y + detectedRightEdge.y) / 2 - ringCenter.y) * perpendicularY,
              revision: fingerEdges.revision,
              confidence: fingerEdges.confidence,
            };
          }
        }
        const fit = fingerFitFilterRef.current.update(
          surfacePose.handScale, surfacePose.facing, edgeSample, surfacePose.anchors,
        );
        if (!fit) continue;
        const constrained = ringSurfaceRef.current.update(
          surfacePose.orientation, surfacePose.anchors, surfacePose.fingerLength, fit.width,
        );
        // Same for Left and Right: MediaPipe winding vs model −Z needs a 180°
        // spin about the finger so dorsal shows the setting and palm shows the
        // rear shank. Occluder uses this orientation too.
        const orientation = constrained.clone().multiply(
          new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI),
        );
        // Bone-centered by default; only a small silhouette nudge is allowed.
        const lateral =
          Math.abs(fit.offset) <= fit.width * PLACEMENT_OFFSET_GATE
            ? fit.offset
            : 0;
        ringPlacement = {
          x: ringCenter.x + perpendicularX * lateral,
          y: ringCenter.y + perpendicularY * lateral,
          width: fit.width,
          orientation,
          surfaceFacing: Math.abs(new THREE.Vector3(0, 0, 1).applyQuaternion(orientation).z),
          anchors: surfacePose.anchors,
          fingerLength: surfacePose.fingerLength,
          handScale: surfacePose.handScale,
          imageMotion: imageMotion ? {
            before: imageMotion.before.map(point => ({
              x: offsetX + (mode === 'user' ? 1 - point.x : point.x) * renderedWidth,
              y: offsetY + point.y * renderedHeight,
            })),
            after: imageMotion.after.map(point => ({
              x: offsetX + (mode === 'user' ? 1 - point.x : point.x) * renderedWidth,
              y: offsetY + point.y * renderedHeight,
            })),
          } : undefined,
        };
      }

      const stablePlacement = ringPoseFilterRef.current.update(ringPlacement, performance.now());
      if (showGuides) {
        const raw = sobelFingerEdgesRef.current;
        const age = raw ? performance.now() - raw.revision : Infinity;
        const fresh = raw && age < 120 && hands.length > 0;
        const clear = fresh && raw.confidence >= MIN_FINGER_EDGE_CONFIDENCE;
        const rawColor = clear ? '#54e7f7' : '#ff9b54';
        context.save();
        if (fresh) {
          const toScreen = (point: CameraPoint) => ({
            x: offsetX + (mode === 'user' ? 1 - point.x : point.x) * renderedWidth,
            y: offsetY + point.y * renderedHeight,
          });
          context.strokeStyle = rawColor;
          context.lineWidth = 1;
          context.globalAlpha = 0.6;
          for (const section of raw.sections) {
            const a = toScreen(section.left), b = toScreen(section.right);
            context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
          }
          context.globalAlpha = 1;
          context.setLineDash([5, 4]);
          context.lineWidth = 2;
          const a = toScreen(raw.left), b = toScreen(raw.right);
          context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
          context.setLineDash([]);
          context.fillStyle = rawColor;
          for (const p of [a, b]) context.fillRect(p.x - 4, p.y - 4, 8, 8);
        }
        const fitFilter = fingerFitFilterRef.current;
        const edgeLabel = fresh
          ? `Raw: ${raw.reason} | score ${raw.confidence.toFixed(2)} | ${raw.support}/6 sections`
          : 'Raw: no usable edges';
        const fitLabel = !hands.length ? 'Show your hand to measure'
          : fitFilter.calibrated ? 'Size locked - use Refit ring to measure again'
          : !fresh || fitFilter.status === 'uncertain' ? 'Waiting for clear edges to fit ring'
          : fitFilter.status === 'side view' ? 'Face your palm or back of hand toward the camera'
          : fitFilter.status === 'confirming' ? `Confirming edges ${fitFilter.confirmationFrames}/3`
          : 'Clear edges accepted';
        const lines = [edgeLabel, fitLabel, 'Dashed / squares: raw | Yellow: filtered'];
        context.font = '12px Arial, sans-serif';
        const panelWidth = Math.min(width - 24, Math.max(...lines.map(line => context.measureText(line).width)) + 20);
        context.fillStyle = 'rgba(0,0,0,0.72)';
        context.fillRect(12, 16, panelWidth, 68);
        context.textBaseline = 'top';
        lines.forEach((line, i) => {
          context.fillStyle = i === 0 ? rawColor : '#fff';
          context.fillText(line, 22, 24 + i * 19, Math.max(1, panelWidth - 20));
        });
        context.restore();
      }
      if (showGuides && stablePlacement) {
        const halfWidth = stablePlacement.width / 2;
        const axis = projectedFingerAxis(stablePlacement.orientation);
        const directionX = -axis.y;
        const directionY = axis.x;
        const guideSeatT = seatCrossPositionRef.current;
        context.strokeStyle = '#ffd45c';
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(
          stablePlacement.x - axis.x * stablePlacement.fingerLength * guideSeatT,
          stablePlacement.y - axis.y * stablePlacement.fingerLength * guideSeatT,
        );
        context.lineTo(
          stablePlacement.x + axis.x * stablePlacement.fingerLength * (1 - guideSeatT),
          stablePlacement.y + axis.y * stablePlacement.fingerLength * (1 - guideSeatT),
        );
        context.stroke();
        const edgePoints = [
          {
            x: stablePlacement.x - directionX * halfWidth,
            y: stablePlacement.y - directionY * halfWidth,
          },
          {
            x: stablePlacement.x + directionX * halfWidth,
            y: stablePlacement.y + directionY * halfWidth,
          },
        ];

        context.strokeStyle = '#ffd45c';
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(edgePoints[0].x, edgePoints[0].y);
        context.lineTo(edgePoints[1].x, edgePoints[1].y);
        context.stroke();

        context.fillStyle = '#ffd45c';
        context.strokeStyle = 'rgba(0, 0, 0, 0.7)';
        context.lineWidth = 1;
        for (const point of edgePoints) {
          context.beginPath();
          context.arc(point.x, point.y, 3, 0, Math.PI * 2);
          context.fill();
          context.stroke();
        }
      }

      const primary = hands[0];
      if (primary) {
        latestCaptureFrameRef.current = {
          landmarks: primary.landmarks,
          placement: stablePlacement,
          handedness: primary.handedness,
          fitCalibrated: fingerFitFilterRef.current.calibrated,
        };
      } else {
        latestCaptureFrameRef.current = null;
      }

      const inPhoto = uiModeRef.current === 'photo';
      const phase = photoPhaseRef.current;

      if (inPhoto && (phase === 'settling' || phase === 'still')) {
        const snap = photoSnapshotRef.current;
        renderRingModel(snap?.placement ?? null);
        return;
      }

      if (inPhoto && phase === 'preview') {
        // Guide-only preview: hide the live ring until capture freezes a frame.
        renderRingModel(null);

        const now = performance.now();
        const motion = landmarkMotion(
          prevCaptureLandmarksRef.current,
          primary?.landmarks,
        );
        prevCaptureLandmarksRef.current = primary?.landmarks
          ? primary.landmarks.map((pt) => ({ ...pt }))
          : null;

        const fitCalibrated = fingerFitFilterRef.current.calibrated;
        const captureEval = evaluateCapturePose({
          hasHand: !!primary,
          landmarks: primary?.landmarks,
          motion,
          fitCalibrated,
          placement: stablePlacement,
        });
        const manualGate = evaluateManualCaptureGate({
          hasHand: !!primary,
          fitCalibrated,
          placement: stablePlacement,
        });

        if (photoCaptureModeRef.current === 'auto') {
          const hold = poseHoldTimerRef.current.update(captureEval.ok, now);
          if (now - lastPhotoHudAtRef.current > 80) {
            lastPhotoHudAtRef.current = now;
            setPhotoHoldProgress(hold.progress);
            setPhotoHint(captureEval.hint);
            setPhotoManualReady(manualGate.ok);
          }
          if (hold.captured) {
            const committed = beginPhotoCaptureRef.current();
            if (!committed) poseHoldTimerRef.current.reset();
          }
        } else {
          poseHoldTimerRef.current.reset();
          if (now - lastPhotoHudAtRef.current > 80) {
            lastPhotoHudAtRef.current = now;
            setPhotoHoldProgress(0);
            setPhotoHint(manualGate.ok ? MANUAL_CAPTURE_HINT : manualGate.hint);
            setPhotoManualReady(manualGate.ok);
          }
        }
        return;
      }

      renderRingModel(stablePlacement);
    },
    [renderRingModel],
  );

  const startHandTracking = useCallback(
    async (
      stream: MediaStream,
      mode: FacingMode,
      onError: (error: unknown) => void,
    ) => {
      const handLandmarker = await getHandLandmarker();

      if (streamRef.current !== stream) {
        return;
      }

      // Avoid initializing two WASM heaps simultaneously on camera startup.
      const lightweightMotion = usesLightweightTracking(navigator.userAgent, navigator.maxTouchPoints);
      if (!lightweightMotion) {
        void getOpenCv().catch((error) => {
          if (streamRef.current === stream) console.warn('Skin motion tracking could not load; using landmarks.', error);
        });
      }

      // Hold Left/Right through brief classifier flicker, but accept a real
      // hand swap after several agreeing frames (or after the hand left).
      const HAND_SWAP_CONFIRM_FRAMES = 10;
      let trackedHandedness: Handedness | null = null;
      let lastHandSeenAt = 0;
      let pendingHandedness: Handedness | null = null;
      let pendingHandednessFrames = 0;
      const renderPoints = () => {
        const video = videoRef.current;

        if (streamRef.current !== stream || !video) {
          return;
        }

        try {
        const now = performance.now();
        const frozenPhoto =
          uiModeRef.current === 'photo' &&
          (photoPhaseRef.current === 'settling' ||
            photoPhaseRef.current === 'still');

        if (frozenPhoto) {
          const snap = photoSnapshotRef.current;
          if (snap?.placement && photoPhaseRef.current === 'still') {
            renderRingModel(snap.placement);
          }
          animationFrameRef.current = requestAnimationFrame(renderPoints);
          return;
        }

        const capturedVideoTime = video.currentTime;
        const hasNewFrame = capturedVideoTime !== lastVideoTimeRef.current;

        // Process the newest available frame. This prevents the 30 ms tracking
        // gap that made the model trail behind a moving camera or hand.
        if (video.readyState >= 2 && hasNewFrame) {
          const detectionCanvas =
            handDetectionCanvasRef.current ?? document.createElement('canvas');
          handDetectionCanvasRef.current = detectionCanvas;
          const inputSize = detectionSize(video.videoWidth, video.videoHeight, HAND_MODEL_INPUT_SIZE);
          if (detectionCanvas.width !== inputSize.width || detectionCanvas.height !== inputSize.height) {
            detectionCanvas.width = inputSize.width;
            detectionCanvas.height = inputSize.height;
          }
          const detectionContext = detectionCanvas.getContext('2d', { willReadFrequently: true });
          let detectedLandmarks: HandPoint[] | null = null;
          let detectedHandedness: Handedness | null = null;

          if (detectionContext) {
            // Preserve the whole video and its aspect ratio, including corners.
            const preview = previewCanvasRef.current;
            const previewContext = preview?.getContext('2d');
            if (!preview || !previewContext) throw new Error('Camera presentation is unavailable.');
            if (preview.width !== video.videoWidth || preview.height !== video.videoHeight) {
              preview.width = video.videoWidth; preview.height = video.videoHeight;
            }
            // Present the same camera frame used for the pose. A live <video>
            // can advance independently while inference is still running.
            const currentZoom = zoomRef.current;
            if (!hasHardwareZoomRef.current && currentZoom > 1.01) {
              const cropW = video.videoWidth / currentZoom;
              const cropH = video.videoHeight / currentZoom;
              const cropX = (video.videoWidth - cropW) / 2;
              const cropY = (video.videoHeight - cropH) / 2;
              previewContext.drawImage(
                video,
                cropX,
                cropY,
                cropW,
                cropH,
                0,
                0,
                preview.width,
                preview.height,
              );
            } else {
              previewContext.drawImage(video, 0, 0, preview.width, preview.height);
            }
            detectionContext.drawImage(preview, 0, 0, inputSize.width, inputSize.height);

            const result = handLandmarker.detectForVideo(detectionCanvas, now);
            const imageLandmarks = result.landmarks[0];
            const worldLandmarks = result.worldLandmarks[0];
            const categoryName = result.handedness[0]?.[0]?.categoryName;
            if (imageLandmarks) {
              const mappedLandmarks = imageLandmarks.map((point, index) => {
                const worldPoint = worldLandmarks?.[index];
                return {
                  x: point.x,
                  y: point.y,
                  z: point.z,
                  worldX: worldPoint?.x,
                  worldY: worldPoint?.y,
                  worldZ: worldPoint?.z,
                };
              });
              // Unmirrored detection: canonicalize MediaPipe labels per camera,
              // then let palm winding override sustained mislabels.
              const resolvedHandedness = handednessResolverRef.current.resolve(
                categoryName,
                mode,
                mappedLandmarks,
              );
              if (resolvedHandedness) {
                const handAbsent = !trackedHandedness || now - lastHandSeenAt > 300;
                if (handAbsent) {
                  if (trackedHandedness && trackedHandedness !== resolvedHandedness) {
                    resetRingFit();
                  }
                  trackedHandedness = resolvedHandedness;
                  pendingHandedness = null;
                  pendingHandednessFrames = 0;
                } else if (resolvedHandedness === trackedHandedness) {
                  pendingHandedness = null;
                  pendingHandednessFrames = 0;
                } else if (pendingHandedness === resolvedHandedness) {
                  pendingHandednessFrames += 1;
                  if (pendingHandednessFrames >= HAND_SWAP_CONFIRM_FRAMES) {
                    trackedHandedness = resolvedHandedness;
                    pendingHandedness = null;
                    pendingHandednessFrames = 0;
                    resetRingFit();
                  }
                } else {
                  pendingHandedness = resolvedHandedness;
                  pendingHandednessFrames = 1;
                }
              }
              detectedHandedness = trackedHandedness;
              lastHandSeenAt = now;
              detectedLandmarks = mappedLandmarks;
            }
          }

          const frameLandmarks = detectedLandmarks;
          let skinMotion: SkinMotion | null = null;
          if ((lightweightMotion || openCvRef.current) && frameLandmarks) {
            const flowCanvas = flowCanvasRef.current ?? document.createElement('canvas');
            flowCanvasRef.current = flowCanvas;
            const flowSize = detectionSize(video.videoWidth, video.videoHeight, 320);
            if (flowCanvas.width !== flowSize.width || flowCanvas.height !== flowSize.height) {
              flowCanvas.width = flowSize.width; flowCanvas.height = flowSize.height;
            }
            const flowContext = flowCanvas.getContext('2d', { willReadFrequently: true });
            if (flowContext) {
              flowContext.drawImage(detectionCanvas, 0, 0, flowSize.width, flowSize.height);
              try {
                const image = flowContext.getImageData(0, 0, flowSize.width, flowSize.height);
                const finger = getFingerDef(targetFingerRef.current);
                const region = [frameLandmarks[finger.mcp], frameLandmarks[finger.pip]];
                skinMotion = lightweightMotion
                  ? patchMotionRef.current.update(image, region, now)
                  : skinMotionRef.current.update(openCvRef.current!, image, region, now);
              } catch (error) {
                skinMotionRef.current.reset();
                patchMotionRef.current.reset();
                console.warn('Skin tracking unavailable for this frame.', error);
              }
            }
          } else {
            skinMotionRef.current.reset();
            patchMotionRef.current.reset();
          }

          if (!frameLandmarks) {
            sobelFingerEdgesRef.current = null;
          }

          // Measure this captured frame, never reuse an older edge midpoint
          // against newer landmarks: that produces a motion-dependent offset.
          {
            try {
              updateHorizontalSobel(
                detectionCanvas,
                frameLandmarks,
              );
            } catch (error) {
              clearSobelOverlay();
              console.error('Horizontal Sobel filtering failed.', error);
            }
          }

          drawPoints(
            frameLandmarks
              ? [
                  {
                    landmarks: frameLandmarks,
                    handedness: detectedHandedness,
                  },
                ]
              : [],
            mode,
            skinMotion,
          );

          lastVideoTimeRef.current = capturedVideoTime;
          lastDetectionTimeRef.current = now;
        }

        animationFrameRef.current = requestAnimationFrame(renderPoints);
        } catch (error) {
          onError(error);
        }
      };

      stopHandTracking();
      animationFrameRef.current = requestAnimationFrame(renderPoints);
    },
    [
      clearSobelOverlay,
      drawPoints,
      getHandLandmarker,
      getOpenCv,
      resetRingFit,
      renderRingModel,
      stopHandTracking,
      updateHorizontalSobel,
    ],
  );

  const cancelPhotoSettle = useCallback(() => {
    settleTokenRef.current += 1;
    if (settleRafRef.current !== null) {
      cancelAnimationFrame(settleRafRef.current);
      settleRafRef.current = null;
    }
  }, []);

  const runPhotoSettle = useCallback(
    (token: number) => {
      const start = performance.now();
      let frames = 0;

      const step = () => {
        if (token !== settleTokenRef.current) {
          settleRafRef.current = null;
          return;
        }

        const snap = photoSnapshotRef.current;
        if (snap?.placement) {
          renderRingModel(snap.placement);
        }

        frames += 1;
        const elapsed = performance.now() - start;
        const fitOk = hasProperRingFit({
          fitCalibrated: snap?.fitCalibrated,
          placement: snap?.placement ?? null,
        });
        const modelReady = !!ringModelRef.current && !ringLoadFailedRef.current;
        const warmed = frames >= SETTLE_FRAMES;

        if ((fitOk && modelReady && warmed) || elapsed >= SETTLE_MAX_MS) {
          settleRafRef.current = null;
          if (token !== settleTokenRef.current) return;
          photoPhaseRef.current = 'still';
          setPhotoPhase('still');
          if (snap?.placement) renderRingModel(snap.placement);
          return;
        }

        settleRafRef.current = requestAnimationFrame(step);
      };

      settleRafRef.current = requestAnimationFrame(step);
    },
    [renderRingModel],
  );

  const beginPhotoCapture = useCallback(() => {
    if (photoPhaseRef.current === 'settling') return false;
    if (uiModeRef.current !== 'photo') return false;

    const frame = latestCaptureFrameRef.current;
    const video = videoRef.current;
    const source = previewCanvasRef.current ?? video;
    if (!frame || !source || !photoCanvasRef.current) return false;
    if (!frame.placement && !frame.landmarks.length) return false;

    const ok = captureVideoFrame(source, photoCanvasRef.current);
    if (!ok) return false;

    const snapshot = buildPhotoSnapshot({
      landmarks: frame.landmarks,
      placement: frame.placement,
      handedness: frame.handedness,
      isMirrored: facingMode === 'user',
      videoDimensions: {
        width: video?.videoWidth || photoCanvasRef.current.width,
        height: video?.videoHeight || photoCanvasRef.current.height,
      },
      fitCalibrated: frame.fitCalibrated,
    });

    cancelPhotoSettle();
    const token = settleTokenRef.current;
    photoSnapshotRef.current = snapshot;
    poseHoldTimerRef.current.reset();
    setPhotoHoldProgress(0);
    setPhotoFlash(true);
    window.setTimeout(() => setPhotoFlash(false), 280);
    photoPhaseRef.current = 'settling';
    setPhotoPhase('settling');
    clearPoints();
    runPhotoSettle(token);
    return true;
  }, [cancelPhotoSettle, clearPoints, facingMode, runPhotoSettle]);

  useEffect(() => {
    beginPhotoCaptureRef.current = beginPhotoCapture;
  }, [beginPhotoCapture]);

  const retakePhoto = useCallback(() => {
    cancelPhotoSettle();
    photoSnapshotRef.current = null;
    prevCaptureLandmarksRef.current = null;
    poseHoldTimerRef.current.reset();
    setPhotoHoldProgress(0);
    setPhotoHint(
      photoCaptureModeRef.current === 'manual'
        ? MANUAL_CAPTURE_HINT
        : DEFAULT_CAPTURE_HINT,
    );
    setPhotoManualReady(false);
    photoPhaseRef.current = 'preview';
    setPhotoPhase('preview');
    renderRingModel(null);
  }, [cancelPhotoSettle, renderRingModel]);

  const selectUiMode = useCallback(
    (mode: UiMode) => {
      if (mode === uiModeRef.current) return;
      cancelPhotoSettle();
      uiModeRef.current = mode;
      setUiMode(mode);
      if (mode === 'photo') {
        photoSnapshotRef.current = null;
        poseHoldTimerRef.current.reset();
        setPhotoHoldProgress(0);
        setPhotoHint(
          photoCaptureModeRef.current === 'manual'
            ? MANUAL_CAPTURE_HINT
            : DEFAULT_CAPTURE_HINT,
        );
        setPhotoManualReady(false);
        photoPhaseRef.current = 'preview';
        setPhotoPhase('preview');
        renderRingModel(null);
      } else {
        photoSnapshotRef.current = null;
        photoPhaseRef.current = 'preview';
        setPhotoPhase('preview');
      }
    },
    [cancelPhotoSettle, renderRingModel],
  );

  const selectPhotoCaptureMode = useCallback((mode: PhotoCaptureMode) => {
    photoCaptureModeRef.current = mode;
    setPhotoCaptureMode(mode);
    poseHoldTimerRef.current.reset();
    setPhotoHoldProgress(0);
    setPhotoHint(mode === 'manual' ? MANUAL_CAPTURE_HINT : DEFAULT_CAPTURE_HINT);
  }, []);

  const releaseStream = useCallback(() => {
    // Invalidate pending permission/play requests before releasing this stream.
    cameraRequestRef.current += 1;
    stopHandTracking();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    hasHardwareZoomRef.current = false;
    zoomRef.current = 1;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [stopHandTracking]);

  const stopCamera = useCallback(() => {
    releaseStream();
    clearPoints();
    setCameraState('idle');
    setScreenshotState('idle');
    setMessage('');
  }, [clearPoints, releaseStream]);

  const startCamera = useCallback(
    async (mode: FacingMode) => {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        setCameraState('error');
        setMessage('Camera access requires HTTPS on phones and tablets.');
        return;
      }
      if (ringContextLostRef.current) {
        setCameraState('error');
        setMessage('Graphics are still recovering. Close this tab and reopen the try-on if this persists.');
        return;
      }

      releaseStream();
      const requestId = cameraRequestRef.current;
      ringLoadFailedRef.current = false;
      setCameraState('requesting');
      setMessage('');

      try {
        let stream: MediaStream;

        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
              facingMode: { ideal: mode },
              width: { ideal: 640 },
              height: { ideal: 480 },
              frameRate: { ideal: 30, max: 30 },
            },
          });
        } catch (error) {
          if (requestId !== cameraRequestRef.current) return;
          if (!(error instanceof DOMException) || error.name !== 'OverconstrainedError') {
            throw error;
          }

          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: true,
          });
        }

        if (requestId !== cameraRequestRef.current || !videoRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;

        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack) {
          try {
            const capabilities = (videoTrack as any).getCapabilities?.();
            if (capabilities && 'zoom' in capabilities) {
              hasHardwareZoomRef.current = true;
              zoomRangeRef.current = {
                min: capabilities.zoom.min ?? 1,
                max: capabilities.zoom.max ?? 3,
                step: capabilities.zoom.step ?? 0.1,
              };
            } else {
              hasHardwareZoomRef.current = false;
            }
          } catch {
            hasHardwareZoomRef.current = false;
          }
        }
        zoomRef.current = 1;

        if (videoRef.current) {
          videoRef.current.muted = true;
          videoRef.current.playsInline = true;
          videoRef.current.srcObject = stream;

          try {
            await videoRef.current.play();
          } catch {
            // iOS Safari can reject the explicit play call while still honoring
            // the muted autoplay attributes once camera metadata is available.
          }
        }

        if (requestId !== cameraRequestRef.current || streamRef.current !== stream) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        setFacingMode(mode);
        setCameraState('active');
        const handleTrackingError = (error: unknown) => {
          if (streamRef.current !== stream) return;
          console.error('Hand tracking failed.', error);
          releaseStream();
          setCameraState('error');
          setMessage('Hand tracking could not start or stopped unexpectedly. Start the camera to try again.');
        };
        void startHandTracking(stream, mode, handleTrackingError).catch(handleTrackingError);

        stream.getVideoTracks()[0]?.addEventListener(
          'ended',
          () => {
            if (streamRef.current !== stream) return;
            releaseStream();
            setCameraState('idle');
            setMessage('Camera disconnected. Start the camera to reconnect.');
          },
          { once: true },
        );
      } catch (error) {
        if (requestId !== cameraRequestRef.current) return;
        releaseStream();
        const name = error instanceof DOMException ? error.name : '';
        setCameraState('error');
        setMessage(errorMessages[name] ?? 'Unable to start the camera.');
      }
    },
    [releaseStream, startHandTracking],
  );

  const toggleCamera = () => {
    if (cameraState === 'active') {
      stopCamera();
      return;
    }

    void startCamera(facingMode);
  };

  const switchCamera = () => {
    const nextMode: FacingMode =
      facingMode === 'environment' ? 'user' : 'environment';

    if (cameraState === 'active') {
      void startCamera(nextMode);
      return;
    }

    setFacingMode(nextMode);
    setMessage('');
  };

  const toggleGuides = () => {
    setGuidesEnabled((enabled) => {
      const next = !enabled;
      guidesEnabledRef.current = next;
      if (!next) {
        clearPoints();
      }
      return next;
    });
  };

  const toggleRingFlip = () => {
    setRingFlipped((flipped) => {
      const next = !flipped;
      ringFlippedRef.current = next;
      return next;
    });
  };

  const selectTargetFinger = useCallback(
    (finger: TryOnFinger) => {
      if (finger === targetFingerRef.current) return;
      targetFingerRef.current = finger;
      setTargetFinger(finger);
      resetRingFit();
    },
    [resetRingFit],
  );

  const toggleTheme = () => {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'));
  };

  const clearLoadedRingModel = useCallback(() => {
    resourceGenerationRef.current += 1;
    const group = ringGroupRef.current;
    if (group) {
      ringSceneRef.current?.remove(group);
    }
    disposeModelResources(ringModelRef.current);
    ringModelRef.current = null;
    ringGroupRef.current = null;
    for (const bvh of ringDiamondBvhsRef.current) bvh.dispose();
    ringDiamondBvhsRef.current.clear();
    ringDiamondMaterialsRef.current = [];
    ringModelPromiseRef.current = null;
    ringLoadFailedRef.current = false;
  }, []);

  const selectRingStyle = useCallback(
    (sku: RingSku) => {
      if (sku === ringStyleRef.current) return;
      ringStyleRef.current = sku;
      setRingStyle(sku);
      clearLoadedRingModel();
      void loadRingModel(sku).catch((error) => {
        console.error(`Unable to load ring model ${sku}.`, error);
        setMessage(`Unable to load ${sku.toUpperCase()}. Try another style.`);
      });
    },
    [clearLoadedRingModel, loadRingModel],
  );

  const takeScreenshot = async () => {
    const video = videoRef.current;
    const ringCanvas = ringCanvasRef.current;
    const pointsCanvas = pointsCanvasRef.current;

    if (
      cameraState !== 'active' ||
      !video ||
      !pointsCanvas ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      return;
    }

    setScreenshotState('saving');

    try {
      const width = pointsCanvas.clientWidth;
      const height = pointsCanvas.clientHeight;
      const pixelRatio = ringPixelRatio(width, height, window.devicePixelRatio, navigator.maxTouchPoints > 0);
      const screenshotCanvas = document.createElement('canvas');
      screenshotCanvas.width = Math.max(1, Math.round(width * pixelRatio));
      screenshotCanvas.height = Math.max(1, Math.round(height * pixelRatio));
      const context = screenshotCanvas.getContext('2d');

      if (!context) {
        throw new Error('Screenshot canvas is unavailable.');
      }

      context.fillStyle = '#111111';
      context.fillRect(0, 0, screenshotCanvas.width, screenshotCanvas.height);

      const videoScale = Math.max(
        width / video.videoWidth,
        height / video.videoHeight,
      );
      const renderedWidth = video.videoWidth * videoScale;
      const renderedHeight = video.videoHeight * videoScale;
      const offsetX = (width - renderedWidth) / 2;
      const offsetY = (height - renderedHeight) / 2;
      const computedTransform = window.getComputedStyle(video).transform;
      const transform = new DOMMatrixReadOnly(
        computedTransform === 'none' ? undefined : computedTransform,
      );

      context.setTransform(
        pixelRatio * transform.a,
        pixelRatio * transform.b,
        pixelRatio * transform.c,
        pixelRatio * transform.d,
        pixelRatio * transform.e,
        pixelRatio * transform.f,
      );
      context.drawImage(
        previewCanvasRef.current ?? video,
        offsetX,
        offsetY,
        renderedWidth,
        renderedHeight,
      );

      context.setTransform(1, 0, 0, 1, 0, 0);
      if (ringCanvas && modelEnabledRef.current) {
        // The default framebuffer may have been discarded after presentation.
        if (ringRendererRef.current && ringSceneRef.current && ringCameraRef.current && !ringContextLostRef.current) {
          ringRendererRef.current.render(ringSceneRef.current, ringCameraRef.current);
        }
        context.drawImage(
          ringCanvas,
          0,
          0,
          screenshotCanvas.width,
          screenshotCanvas.height,
        );
      }
      if (guidesEnabled) {
        context.drawImage(
          pointsCanvas,
          0,
          0,
          screenshotCanvas.width,
          screenshotCanvas.height,
        );
      }

      const screenshot = await new Promise<Blob>((resolve, reject) => {
        screenshotCanvas.toBlob((blob) => {
          if (blob) {
            resolve(blob);
          } else {
            reject(new Error('Unable to encode the screenshot.'));
          }
        }, 'image/png');
      });
      const response = await fetch('/__save-screenshot', {
        method: 'POST',
        headers: { 'Content-Type': 'image/png' },
        body: screenshot,
      });

      if (!response.ok) {
        throw new Error('Unable to save the screenshot.');
      }

      setScreenshotState('saved');
      window.setTimeout(() => setScreenshotState('idle'), 1800);
    } catch (error) {
      console.error('Screenshot capture failed.', error);
      setScreenshotState('error');
      window.setTimeout(() => setScreenshotState('idle'), 2400);
    }
  };

  useEffect(() => {
    modelEnabledRef.current = modelEnabled;
    if (!modelEnabledRef.current) {
      renderRingModel(null);
    }
  }, [modelEnabled, renderRingModel]);

  useEffect(() => {
    guidesEnabledRef.current = guidesEnabled;
    if (!guidesEnabled) {
      clearPoints();
    }
  }, [clearPoints, guidesEnabled]);

  useEffect(() => {
    ringFlippedRef.current = ringFlipped;
  }, [ringFlipped]);

  useEffect(() => {
    ringSeatAlongRef.current = ringSeatAlong;
    seatCrossPositionRef.current = ringSeatAlong;
  }, [ringSeatAlong]);

  useEffect(() => {
    targetFingerRef.current = targetFinger;
  }, [targetFinger]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem('ring-try-theme');
      if (saved === 'light' || saved === 'dark') setTheme(saved);
    } catch {
      // ignore storage failures
    }
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    try {
      localStorage.setItem('ring-try-theme', theme);
    } catch {
      // ignore storage failures
    }
  }, [theme]);

  useEffect(() => {
    const generation = resourceGenerationRef.current;
    initRingScene();
    void loadRingModel(ringStyleRef.current).catch((error) => {
      if (generation !== resourceGenerationRef.current) return;
      console.error('Unable to preload the ring model.', error);
      setMessage('Unable to load the ring. Start the camera to try again.');
    });
    const canvas = ringCanvasRef.current;
    let mounted = true;
    const handleContextLost = (event: Event) => {
      event.preventDefault();
      ringContextLostRef.current = true;
      stopCamera();
      setCameraState('error');
      setMessage('The graphics session stopped. Waiting for recovery before restarting the camera.');
    };
    const handleContextRestored = () => {
      // Let Three finish its own context-restored event handler first.
      queueMicrotask(() => {
        if (!mounted) return;
        disposeRingModel();
        ringContextLostRef.current = false;
        ringLoadFailedRef.current = false;
        ringShaderFailedRef.current = false;
        setCameraState('idle');
        setMessage('Graphics recovered. Start the camera to try again.');
      });
    };
    canvas?.addEventListener('webglcontextlost', handleContextLost);
    canvas?.addEventListener('webglcontextrestored', handleContextRestored);
    const handlePageHide = () => stopCamera();
    window.addEventListener('pagehide', handlePageHide);

    return () => {
      mounted = false;
      canvas?.removeEventListener('webglcontextlost', handleContextLost);
      canvas?.removeEventListener('webglcontextrestored', handleContextRestored);
      window.removeEventListener('pagehide', handlePageHide);
      releaseStream();
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
      handLandmarkerPromiseRef.current = null;
      openCvPromiseRef.current = null;
      openCvRef.current = null;
      disposeRingModel();
    };
  }, [disposeRingModel, initRingScene, loadRingModel, releaseStream, stopCamera]);

  useEffect(() => {
    void startCamera(facingMode);
    // Start the camera once when the try-on screen mounts.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional one-shot auto-start
  }, []);

  return (
    <main
      className={`camera-screen camera-screen--${facingMode}`}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      <header className="app-header">
        <div className="header-left">
          <span className="logo-mark" aria-hidden="true">
            <AppIcon icon={icons.sparkles} size={16} />
          </span>
          <div className="logo-group">
            <h1 className="logo-text">Ring Try</h1>
            <span className="logo-badge">OpenCV Engine</span>
          </div>
        </div>

        <div className="mode-switch-container" role="tablist" aria-label="Try-on mode">
          <button
            type="button"
            className={`mode-btn ${uiMode === 'tryon' ? 'active' : ''}`}
            role="tab"
            aria-selected={uiMode === 'tryon'}
            onClick={() => selectUiMode('tryon')}
          >
            <AppIcon icon={icons.live} size={16} className="mode-btn-icon" />
            Live
          </button>
          <button
            type="button"
            className={`mode-btn ${uiMode === 'photo' ? 'active' : ''}`}
            role="tab"
            aria-selected={uiMode === 'photo'}
            onClick={() => selectUiMode('photo')}
          >
            <AppIcon icon={icons.photo} size={16} className="mode-btn-icon" />
            Take Photo
          </button>
        </div>

        <div className="header-right">
          <div
            className={`status-chip ${
              cameraState === 'active' ? 'status-chip--live' : 'status-chip--idle'
            }`}
          >
            {cameraState === 'active' ? 'Tracking Pose…' : 'Camera Off'}
          </div>
          <button
            type="button"
            className="theme-toggle-btn"
            title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
            aria-label={
              theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'
            }
            onClick={toggleTheme}
          >
            <AppIcon icon={theme === 'dark' ? icons.sun : icons.moon} size={18} />
          </button>
        </div>
      </header>

      <div className="camera-stage">
        {modelLoadingState.status === 'loading' && (
          <div className="model-loading-pill" role="status" aria-live="polite">
            <div className="model-loading-row">
              <div className="model-loading-spinner" aria-hidden="true" />
              <span>{modelLoadingState.message}</span>
            </div>
            <div className="model-loading-track" aria-hidden="true">
              <div
                className="model-loading-fill"
                style={{ width: `${Math.round(modelLoadingState.progress)}%` }}
              />
            </div>
          </div>
        )}
        <video
          ref={videoRef}
          className="camera-preview"
          muted
          playsInline
          autoPlay
          style={{ opacity: cameraState === 'active' ? 0 : 1 }}
          aria-hidden="true"
        />
        <canvas ref={previewCanvasRef} className="camera-preview" aria-label="Live camera preview" />
        <canvas
          ref={photoCanvasRef}
          className={`photo-canvas ${
            uiMode === 'photo' &&
            (photoPhase === 'settling' || photoPhase === 'still')
              ? 'photo-canvas--visible'
              : ''
          }`}
          aria-hidden={
            !(
              uiMode === 'photo' &&
              (photoPhase === 'settling' || photoPhase === 'still')
            )
          }
        />
        <canvas
          ref={ringCanvasRef}
          className="ring-model"
          aria-hidden="true"
        />
        <canvas
          ref={pointsCanvasRef}
          className="hand-points"
          aria-hidden="true"
        />

        {uiMode === 'photo' &&
          cameraState === 'active' &&
          photoPhase === 'preview' && (
            <div
              className={`photo-capture-hud ${photoHoldProgress >= 1 || photoManualReady ? 'ready' : ''}`}
            >
              <div className="photo-capture-modes" role="group" aria-label="Capture mode">
                <button
                  type="button"
                  className={`photo-mode-btn ${photoCaptureMode === 'auto' ? 'active' : ''}`}
                  aria-pressed={photoCaptureMode === 'auto'}
                  onClick={() => selectPhotoCaptureMode('auto')}
                >
                  Auto Capture
                </button>
                <button
                  type="button"
                  className={`photo-mode-btn ${photoCaptureMode === 'manual' ? 'active' : ''}`}
                  aria-pressed={photoCaptureMode === 'manual'}
                  onClick={() => selectPhotoCaptureMode('manual')}
                >
                  Click Photo
                </button>
              </div>
              <p className="photo-capture-status">{photoHint}</p>
              <div
                className={`photo-hold-track ${photoCaptureMode === 'auto' ? '' : 'hidden'}`}
                aria-hidden={photoCaptureMode !== 'auto'}
              >
                <div
                  className="photo-hold-bar"
                  style={{ width: `${Math.round(photoHoldProgress * 100)}%` }}
                />
              </div>
            </div>
          )}

        <div
          className={`capture-flash ${photoFlash ? 'flash' : ''}`}
          aria-hidden="true"
        />

        {uiMode === 'photo' && photoPhase === 'settling' && (
          <div className="capture-settle" aria-live="polite" aria-busy="true">
            <div className="capture-settle-card">
              <div className="capture-settle-spinner" aria-hidden="true" />
              <p className="capture-settle-text">Setting ring…</p>
            </div>
          </div>
        )}

        {(cameraState !== 'active' || message) && (
          <div className="camera-status" role="status" aria-live="polite">
            {cameraState === 'requesting'
              ? 'Starting camera…'
              : message || 'Camera is off'}
          </div>
        )}

        <div
          className={`tryon-bar ${
            uiMode === 'photo' && photoPhase === 'settling' ? 'tryon-bar--hidden' : ''
          }`}
        >
          {uiMode === 'tryon' ? (
            <div className="tryon-panel">
              <div className="tryon-panel-row" role="group" aria-label="Try-on finger">
                <span className="tryon-label">Finger</span>
                <div className="finger-pills">
                  {TRY_ON_FINGERS.map((finger) => (
                    <button
                      key={finger.name}
                      type="button"
                      className={`finger-pill finger-toggle ${targetFinger === finger.name ? 'active' : ''}`}
                      onClick={() => selectTargetFinger(finger.name)}
                      aria-pressed={targetFinger === finger.name}
                      title={`Place ring on ${finger.label.toLowerCase()} finger`}
                    >
                      {finger.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="tryon-panel-row tryon-panel-row--tools">
                <button
                  type="button"
                  className={`tryon-tool-btn ${cameraState === 'active' ? 'tryon-tool-btn--primary' : ''}`}
                  onClick={toggleCamera}
                  disabled={cameraState === 'requesting'}
                  title={cameraState === 'active' ? 'Stop camera' : 'Start camera'}
                >
                  <AppIcon
                    icon={cameraState === 'active' ? icons.cameraOff : icons.live}
                    size={16}
                  />
                  <span className="tryon-tool-label">
                    {cameraState === 'active' ? 'Stop' : 'Start'}
                  </span>
                </button>

                <button
                  type="button"
                  className={`tryon-tool-btn ${guidesEnabled ? 'active' : ''}`}
                  onClick={toggleGuides}
                  aria-pressed={guidesEnabled}
                  title="Toggle measurement guides"
                >
                  <AppIcon icon={icons.guides} size={16} />
                  <span className="tryon-tool-label">Guides</span>
                </button>

                <button
                  type="button"
                  className="tryon-tool-btn"
                  onClick={switchCamera}
                  disabled={cameraState === 'requesting'}
                  title="Switch front / back camera"
                >
                  <AppIcon icon={icons.camera} size={16} />
                  <span className="tryon-tool-label">Camera</span>
                </button>

                <button
                  type="button"
                  className={`tryon-tool-btn flip-btn ${ringFlipped ? 'active' : ''}`}
                  onClick={toggleRingFlip}
                  aria-pressed={ringFlipped}
                  title="Optional: flip head / shank 180°"
                >
                  <AppIcon icon={icons.flip} size={16} />
                  <span className="tryon-tool-label">Flip</span>
                </button>

                <button
                  type="button"
                  className="tryon-tool-btn"
                  onClick={() => void takeScreenshot()}
                  disabled={
                    cameraState !== 'active' || screenshotState === 'saving'
                  }
                  title="Save debug screenshot"
                >
                  <AppIcon icon={icons.photo} size={16} />
                  <span className="tryon-tool-label">
                    {screenshotState === 'saving'
                      ? 'Saving…'
                      : screenshotState === 'saved'
                        ? 'Saved'
                        : screenshotState === 'error'
                          ? 'Failed'
                          : 'SS'}
                  </span>
                </button>

                <button
                  type="button"
                  className="tryon-tool-btn"
                  onClick={resetRingFit}
                  disabled={cameraState !== 'active'}
                  title="Refit ring to finger"
                >
                  <AppIcon icon={icons.retake} size={16} />
                  <span className="tryon-tool-label">Refit</span>
                </button>
              </div>

              <div
                className="tryon-panel-row tryon-panel-row--seat"
                title="Move ring toward palm or tip"
              >
                <span className="tryon-label">Place</span>
                <span className="seat-slider-end">Palm</span>
                <input
                  type="range"
                  className="seat-slider"
                  min={Math.round(SEAT_ALONG_MIN * 100)}
                  max={Math.round(SEAT_ALONG_MAX * 100)}
                  step={1}
                  value={Math.round(ringSeatAlong * 100)}
                  aria-label="Ring placement along finger"
                  onChange={(event) => {
                    const next = Number(event.target.value) / 100;
                    ringSeatAlongRef.current = next;
                    seatCrossPositionRef.current = next;
                    setRingSeatAlong(next);
                  }}
                />
                <span className="seat-slider-end">Tip</span>
              </div>

              <div
                className="tryon-panel-row tryon-panel-row--styles"
                role="listbox"
                aria-label="Ring model"
              >
                {RING_STYLES.map((style) => (
                  <button
                    key={style.id}
                    type="button"
                    role="option"
                    aria-selected={ringStyle === style.id}
                    className={`ring-style-card ${ringStyle === style.id ? 'active' : ''}`}
                    data-style={style.id}
                    onClick={() => selectRingStyle(style.id)}
                    disabled={modelLoadingState.status === 'loading'}
                    title={`Try ${style.name}`}
                  >
                    <span className="style-icon" aria-hidden="true">
                      💍
                    </span>
                    <span className="style-label">{style.name}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {photoPhase === 'still' ? (
                <div className="tryon-panel tryon-panel--compact">
                  <div className="tryon-panel-row tryon-panel-row--tools">
                    <button
                      type="button"
                      className="tryon-tool-btn tryon-tool-btn--primary"
                      onClick={retakePhoto}
                      title="Retake photo"
                    >
                      <AppIcon icon={icons.retake} size={16} />
                      <span className="tryon-tool-label">Retake</span>
                    </button>
                  </div>
                </div>
              ) : photoPhase === 'preview' ? (
                <>
                  {photoCaptureMode === 'manual' && (
                    <button
                      type="button"
                      className="photo-shutter-btn"
                      onClick={() => beginPhotoCapture()}
                      disabled={cameraState !== 'active' || !photoManualReady}
                      title="Take photo"
                    >
                      <span className="photo-shutter-ring" />
                      <span className="photo-shutter-label">Capture</span>
                    </button>
                  )}
                  <div className="tryon-panel tryon-panel--compact">
                    <div className="tryon-panel-row tryon-panel-row--tools">
                      <button
                        type="button"
                        className="tryon-tool-btn"
                        onClick={switchCamera}
                        disabled={cameraState === 'requesting'}
                        title="Switch front / back camera"
                      >
                        <AppIcon icon={icons.camera} size={16} />
                        <span className="tryon-tool-label">Camera</span>
                      </button>
                      <button
                        type="button"
                        className="tryon-tool-btn"
                        onClick={toggleCamera}
                        disabled={cameraState === 'requesting'}
                        title={
                          cameraState === 'active' ? 'Stop camera' : 'Start camera'
                        }
                      >
                        <AppIcon
                          icon={
                            cameraState === 'active'
                              ? icons.cameraOff
                              : icons.live
                          }
                          size={16}
                        />
                        <span className="tryon-tool-label">
                          {cameraState === 'active' ? 'Stop' : 'Start'}
                        </span>
                      </button>
                    </div>
                  </div>
                </>
              ) : null}
            </>
          )}
        </div>
      </div>
    </main>
  );
}
