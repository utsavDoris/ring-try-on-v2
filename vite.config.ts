import { sites } from '@openai/sites-vite-plugin';
import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig, type Plugin } from 'vite';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import hostingConfig from './.openai/hosting.json';

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  '00000000-0000-4000-8000-000000000000';

const { d1, r2 } = hostingConfig;

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === 'seatbelt';
const localCertificateDirectory = resolve(process.cwd(), '.cert');
const localCertificatePath = resolve(
  localCertificateDirectory,
  'server.crt.pem',
);
const localKeyPath = resolve(localCertificateDirectory, 'server.key.pem');
const localHttps =
  existsSync(localCertificatePath) && existsSync(localKeyPath)
    ? {
        cert: readFileSync(localCertificatePath),
        key: readFileSync(localKeyPath),
      }
    : undefined;

const localScreenshotSaver = (): Plugin => ({
  name: 'local-screenshot-saver',
  configureServer(server) {
    server.middlewares.use(
      '/__save-screenshot',
      (request, response, next) => {
        if (request.method !== 'POST') {
          next();
          return;
        }

        const maximumBytes = 20 * 1024 * 1024;
        const chunks: Buffer[] = [];
        let receivedBytes = 0;

        request.on('data', (chunk: Buffer) => {
          receivedBytes += chunk.length;
          if (receivedBytes <= maximumBytes) {
            chunks.push(chunk);
          }
        });

        request.on('end', () => {
          void (async () => {
            if (receivedBytes > maximumBytes) {
              response.statusCode = 413;
              response.end('Screenshot is too large.');
              return;
            }

            const screenshot = Buffer.concat(chunks);
            const pngSignature = Buffer.from([
              0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
            ]);

            if (
              request.headers['content-type'] !== 'image/png' ||
              screenshot.length < pngSignature.length ||
              !screenshot.subarray(0, pngSignature.length).equals(pngSignature)
            ) {
              response.statusCode = 415;
              response.end('Only PNG screenshots are accepted.');
              return;
            }

            try {
              const screenshotDirectory = resolve(
                process.cwd(),
                'screenshots',
              );
              await mkdir(screenshotDirectory, { recursive: true });
              const timestamp = new Date()
                .toISOString()
                .replace(/[:.]/g, '-');
              const filename = `ring-tryon-${timestamp}.png`;
              await writeFile(
                resolve(screenshotDirectory, filename),
                screenshot,
              );

              response.statusCode = 201;
              response.setHeader('Content-Type', 'application/json');
              response.setHeader('Cache-Control', 'no-store');
              response.end(
                JSON.stringify({ path: `screenshots/${filename}` }),
              );
            } catch (error) {
              console.error('Unable to save local screenshot.', error);
              response.statusCode = 500;
              response.end('Unable to save screenshot.');
            }
          })();
        });
      },
    );
  },
});

const localBindingConfig = {
  main: 'vinext/server/app-router-entry',
  compatibility_flags: ['nodejs_compat'],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: 'site-creator-d1',
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: 'site-creator-r2',
        },
      ]
    : [],
};

export default defineConfig(async () => {
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= 'false';
  process.env.WRANGLER_LOG_PATH ??= '.wrangler/logs';
  process.env.MINIFLARE_REGISTRY_PATH ??= '.wrangler/registry';

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import('@cloudflare/vite-plugin');

  return {
    build: { target: ['safari15', 'ios15', 'chrome87'] },
    css: { postcss: { plugins: [tailwindcss()] } },
    server: {
      host: '0.0.0.0',
      https: localHttps,
      watch: {
        ignored: ['**/public/mediapipe/**', '**/public/opencv/**'],
        ...(isCodexSeatbeltSandbox
          ? { useFsEvents: false, usePolling: true }
          : {}),
      },
    },
    plugins: [
      localScreenshotSaver(),
      vinext(),
      sites(),
      cloudflare({
        viteEnvironment: { name: 'rsc', childEnvironments: ['ssr'] },
        config: localBindingConfig,
      }),
    ],
  };
});
