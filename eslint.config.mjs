import { defineConfig, globalIgnores } from 'eslint/config';
import nextVitals from 'eslint-config-next/core-web-vitals';
import nextTs from 'eslint-config-next/typescript';

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([
    '.next/**',
    'dist/**',
    'out/**',
    'build/**',
    'next-env.d.ts',
    // Bundled third-party runtimes are distributed assets, not app source.
    'public/draco/**',
    'public/mediapipe/**',
    'public/opencv/**',
  ]),
]);

export default eslintConfig;
