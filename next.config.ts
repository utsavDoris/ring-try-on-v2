import type { NextConfig } from 'next';
import { networkInterfaces } from 'node:os';

const localAddresses = Object.values(networkInterfaces())
  .flatMap((addresses) => addresses ?? [])
  .filter((address) => address.family === 'IPv4' && !address.internal)
  .map((address) => address.address);

const nextConfig: NextConfig = {
  allowedDevOrigins: localAddresses,
};

export default nextConfig;
