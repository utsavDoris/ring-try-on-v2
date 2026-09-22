import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, KHRDracoMeshCompression } from '@gltf-transform/extensions';
import { createDecoderModule } from 'draco3dgltf';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_MODEL = join(ROOT, 'public', 'models', 'LR1844.glb');

let decoderPromise = null;

function getDecoder() {
  if (!decoderPromise) decoderPromise = createDecoderModule();
  return decoderPromise;
}

async function createIo() {
  return new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      'draco3d.decoder': await getDecoder(),
    });
}

function stripDracoExtension(document) {
  for (const extension of document.getRoot().listExtensionsUsed()) {
    if (extension.extensionName === KHRDracoMeshCompression.EXTENSION_NAME) {
      extension.dispose();
    }
  }
  for (const extension of document.getRoot().listExtensionsRequired()) {
    if (extension.extensionName === KHRDracoMeshCompression.EXTENSION_NAME) {
      extension.dispose();
    }
  }
}

async function documentToGltf(document) {
  stripDracoExtension(document);
  const io = await createIo();
  const binary = await io.writeBinary(document);
  const loader = new GLTFLoader();
  return loader.parseAsync(
    binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength),
    '',
  );
}

/**
 * Load the try-on ring GLB under Node, expanding Draco compression first.
 */
export async function loadRingGltf(modelPath = DEFAULT_MODEL) {
  const io = await createIo();
  const document = await io.read(modelPath);
  return documentToGltf(document);
}

export async function loadRingGltfFromBytes(bytes) {
  const io = await createIo();
  const document = await io.readBinary(
    new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
  );
  return documentToGltf(document);
}

export async function readRingModelBytes() {
  return readFile(DEFAULT_MODEL);
}
