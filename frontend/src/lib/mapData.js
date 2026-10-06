// Static landscape map shipped with the site, so the hero and Atlas never wait on the backend.

// Field palette in the spirit of histological stains, tuned to read as ink on ivory paper.
export const STAIN = [
  '#3E2F73', '#5B3F99', '#7E4A9E', '#A2498C', '#C9476E', '#D4655A', '#C27C38', '#A38A2A',
  '#7B8636', '#4E7D49', '#2D7A69', '#2B6C88', '#3A5A97', '#55508F', '#77679B', '#995A7B',
  '#B3705D', '#8B6A49', '#6A6B48', '#4D6D65', '#3D5D70', '#645878', '#874C5D', '#9C7350',
]

let cache = null
let titlesCache = null

export function loadMap() {
  if (cache) return cache
  cache = Promise.all([
    fetch('/data/map.json').then((r) => r.json()),
    fetch('/data/map.bin').then((r) => r.arrayBuffer()),
  ]).then(([meta, buf]) => {
    const n = meta.count
    const stride = meta.stride || 10
    const view = new DataView(buf)
    const x = new Float32Array(n), y = new Float32Array(n)
    const d = new Uint8Array(n), t = new Uint8Array(n), pmid = new Uint32Array(n)
    for (let i = 0; i < n; i++) {
      const o = i * stride
      x[i] = view.getInt16(o, true) / 32767
      y[i] = view.getInt16(o + 2, true) / 32767
      d[i] = view.getUint8(o + 4)
      t[i] = view.getUint8(o + 5)
      pmid[i] = view.getUint32(o + 6, true)
    }
    const index = new Map()
    for (let i = 0; i < n; i++) index.set(String(pmid[i]), i)
    return { ...meta, colors: meta.domains.map((_, i) => STAIN[i % STAIN.length]), n, x, y, d, t, pmid, index }
  })
  return cache
}

// [title, year, journal] per paper, in map order. Loaded lazily (about 470 KB compressed).
export function loadTitles() {
  if (!titlesCache) titlesCache = fetch('/data/titles.json').then((r) => r.json()).catch(() => null)
  return titlesCache
}

export const fmt = (n) => (n ?? 0).toLocaleString('en-US')
