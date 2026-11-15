/**
 * Turns the manifest's relative asset URLs ("splat/lods/lod0.spz") into fetchable ones.
 *
 * `asset_base` comes from the API. In development it's absolute
 * ("http://localhost:8000/memories/mem_1/files/"). Deployed behind the web app's /api proxy
 * it's root-relative ("/api/memories/mem_1/files/"), so links work at whatever address
 * people use. Root-relative bases resolve against the page's own address.
 */
export function assetResolver(assetBase: string, pageUrl: string): (url: string) => string {
  const base = new URL(assetBase, pageUrl);
  return (url: string) => new URL(url, base).toString();
}
