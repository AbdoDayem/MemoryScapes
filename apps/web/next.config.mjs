/**
 * Deployed, the browser talks only to this server: /api/* is passed to the API, so there's
 * one address to expose (through a tunnel or the LAN) and no cross-origin setup. Build with
 * NEXT_PUBLIC_API_URL=/api and run the API with PUBLIC_API_URL=/api (deploy/setup.py does
 * both). In development the app can still call the API directly at http://localhost:8000.
 *
 * API_ORIGIN is where this server reaches the API, read at build time (default the API on
 * this machine). proxyTimeout: video uploads and big files can take minutes.
 * middlewareClientMaxBodySize: Next.js forwards at most this much of a request body through
 * the proxy (default 10 MB, after which an upload stalls). It holds an upload in memory
 * while forwarding it, so this matches the API's upload limit (MAX_UPLOAD_MB, 500 MB).
 */
const apiOrigin = (process.env.API_ORIGIN ?? "http://127.0.0.1:8000").replace(/\/$/, "");

/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["@memoryscapes/manifest"],
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiOrigin}/:path*` }];
  },
  experimental: { proxyTimeout: 30 * 60 * 1000, middlewareClientMaxBodySize: "520mb" },
};

export default nextConfig;
