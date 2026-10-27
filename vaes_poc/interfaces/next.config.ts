import type { NextConfig } from "next";

// Domínio público do site, usado no QR do "Você × VAE". Na Vercel vem do
// domínio de produção; fora dela pode ser definido com NEXT_PUBLIC_SITE_URL.
const production = process.env.VERCEL_PROJECT_PRODUCTION_URL;
const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ?? (production ? `https://${production}` : "");

const nextConfig: NextConfig = {
  turbopack: { root: process.cwd() },
  outputFileTracingRoot: process.cwd(),
  env: { SITE_URL: siteUrl.replace(/\/$/, "") },
};

export default nextConfig;
