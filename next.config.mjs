/** @type {import('next').NextConfig} */

// Set BASE_PATH when deploying to a GitHub Pages project site, e.g. "/llama-local-lab".
// Cloudflare Pages / Vercel / Netlify need no base path at all.
const basePath = process.env.BASE_PATH || '';

const nextConfig = {
  // Fully static export — no server, no API routes, no database.
  // The `out/` directory can be dropped onto Cloudflare Pages or GitHub Pages as-is.
  output: 'export',
  basePath,
  assetPrefix: basePath || undefined,
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
  webpack: (config) => {
    // WebLLM and wllama ship WASM/worker assets and reference Node built-ins that
    // do not exist in the browser. Stub them so the static build stays clean.
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
      perf_hooks: false,
    };
    return config;
  },
};

export default nextConfig;
