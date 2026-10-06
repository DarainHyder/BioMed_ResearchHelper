// Loads the static landscape map shipped with the site (no backend needed for the hero or the Atlas).
let cache = null

export function loadMap() {
  if (cache) return cache
  cache = Promise.all([
    fetch('/data/map.json').then((r) => r.json()),
    fetch('/data/map.bin').then((r) => r.arrayBuffer()),
  ]).then(([meta, buf]) => {
    const n = meta.count
    const stride = meta.stride || 10
    const view = new DataView(buf)
    const x = new Float32Array(n)
    const y = new Float32Array(n)
    const d = new Uint8Array(n)
    const t = new Uint8Array(n)
    const pmid = new Uint32Array(n)
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
    return { ...meta, n, x, y, d, t, pmid, index }
  })
  return cache
}

export const fmt = (n) => (n ?? 0).toLocaleString('en-US')
