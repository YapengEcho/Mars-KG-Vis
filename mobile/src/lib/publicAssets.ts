/** Resolve a bundled, root-relative asset against the app's deployment path. */
export function publicAssetUrl(path: string): string {
  return import.meta.env.BASE_URL + path.slice(1)
}
