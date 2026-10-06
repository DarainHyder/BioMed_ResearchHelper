import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Minus, Plus, RotateCcw } from 'lucide-react'
import { api } from '../lib/api'
import { fmt, loadMap, loadTitles } from '../lib/mapData'
import { Spinner, setDomainColors, usePaperDrawer } from '../components/ui'

const CELL = 0.025

export default function Atlas() {
  const canvasRef = useRef(null)
  const mapRef = useRef(null)
  const titlesRef = useRef(null)
  const view = useRef({ x: 0, y: 0, k: 1 })
  const grid = useRef(new Map())
  const hoverRef = useRef(-1)
  const highlight = useRef(null)
  const hidden = useRef(new Set())
  const queued = useRef(false)
  const [map, setMap] = useState(null)
  const [hover, setHover] = useState(null)
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [found, setFound] = useState(null)
  const [hiddenState, setHiddenState] = useState(new Set())
  const [openPaper, drawer] = usePaperDrawer()

  const draw = useCallback(() => {
    queued.current = false
    const m = mapRef.current, canvas = canvasRef.current
    if (!m || !canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = canvas.clientWidth, h = canvas.clientHeight
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = w * dpr; canvas.height = h * dpr }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const S = Math.min(w, h) * 0.44 * view.current.k
    const ox = w * 0.5 + view.current.x, oy = h * 0.53 + view.current.y
    const hl = highlight.current
    const r = Math.max(1.3, Math.min(3.4, 1.35 * Math.sqrt(view.current.k)))
    for (let d = 0; d < m.domains.length; d++) {
      if (hidden.current.has(d)) continue
      ctx.fillStyle = m.colors[d]
      ctx.globalAlpha = hl ? 0.2 : 0.8
      ctx.beginPath()
      for (let i = 0; i < m.n; i++) {
        if (m.d[i] !== d) continue
        const px = ox + m.x[i] * S, py = oy - m.y[i] * S
        if (px < -4 || py < -4 || px > w + 4 || py > h + 4) continue
        ctx.moveTo(px + r, py); ctx.arc(px, py, r, 0, Math.PI * 2)
      }
      ctx.fill()
    }
    ctx.globalAlpha = 1
    if (hl) {
      hl.forEach((i) => {
        const px = ox + m.x[i] * S, py = oy - m.y[i] * S
        ctx.beginPath(); ctx.arc(px, py, r + 4, 0, Math.PI * 2); ctx.strokeStyle = '#C9476E'; ctx.lineWidth = 1.4; ctx.stroke()
        ctx.beginPath(); ctx.arc(px, py, r + 0.6, 0, Math.PI * 2); ctx.fillStyle = m.colors[m.d[i]]; ctx.fill()
      })
    }
    const hv = hoverRef.current
    if (hv >= 0) {
      ctx.beginPath(); ctx.arc(ox + m.x[hv] * S, oy - m.y[hv] * S, r + 5, 0, Math.PI * 2)
      ctx.strokeStyle = '#1B1916'; ctx.lineWidth = 1.2; ctx.stroke()
    }
    // annotations: italic serif with a paper halo; largest fronts first, no collisions or repeats
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif'
    ctx.textAlign = 'center'
    ctx.lineJoin = 'round'
    const placed = [], seen = new Set()
    ;[...m.topics].sort((a, b) => b.size - a.size).forEach((t) => {
      const px = ox + t.x * S, py = oy - t.y * S
      if (px < 20 || py < 20 || px > w - 20 || py > h - 20) return
      const words = t.label.split(' · ').slice(0, 2)
      const key = words.map((x) => x.toLowerCase().replace(/s\b/g, '')).sort().join('|') // "Image" and "Images" are the same front name
      if (placed.length >= Math.round(16 * view.current.k)) return
      if (seen.has(key)) return
      const label = words.join(', ')
      const tw = ctx.measureText(label).width
      if (placed.some((b) => Math.abs(b.x - px) < (b.w + tw) / 2 + 18 && Math.abs(b.y - py) < 24)) return
      placed.push({ x: px, y: py, w: tw }); seen.add(key)
      ctx.strokeStyle = 'rgba(247,244,238,0.92)'; ctx.lineWidth = 4; ctx.strokeText(label, px, py)
      ctx.fillStyle = '#1B1916'; ctx.fillText(label, px, py)
    })
  }, [])

  const requestDraw = useCallback(() => {
    if (!queued.current) { queued.current = true; requestAnimationFrame(draw) }
  }, [draw])

  useEffect(() => {
    loadMap().then((m) => {
      setDomainColors(m.domains, m.colors)
      for (let i = 0; i < m.n; i++) {
        const key = `${Math.floor(m.x[i] / CELL)},${Math.floor(m.y[i] / CELL)}`
        if (!grid.current.has(key)) grid.current.set(key, [])
        grid.current.get(key).push(i)
      }
      mapRef.current = m
      setMap(m)
      requestDraw()
      document.fonts?.ready.then(requestDraw)
    })
    loadTitles().then((t) => { titlesRef.current = t })
    window.addEventListener('resize', requestDraw)
    window.__lenis?.stop()
    return () => { window.removeEventListener('resize', requestDraw); window.__lenis?.start() }
  }, [requestDraw])

  const toMap = (sx, sy) => {
    const c = canvasRef.current
    const S = Math.min(c.clientWidth, c.clientHeight) * 0.44 * view.current.k
    return [(sx - c.clientWidth * 0.5 - view.current.x) / S, -(sy - c.clientHeight * 0.53 - view.current.y) / S, S]
  }
  const nearest = (sx, sy) => {
    const m = mapRef.current
    const [mx, my, S] = toMap(sx, sy)
    const rad = 8 / S, cx = Math.floor(mx / CELL), cy = Math.floor(my / CELL), span = Math.ceil(rad / CELL)
    let best = -1, bd = rad * rad
    for (let gx = cx - span; gx <= cx + span; gx++) {
      for (let gy = cy - span; gy <= cy + span; gy++) {
        for (const i of grid.current.get(`${gx},${gy}`) || []) {
          if (hidden.current.has(m.d[i])) continue
          const dx = m.x[i] - mx, dy = m.y[i] - my, dd = dx * dx + dy * dy
          if (dd < bd) { bd = dd; best = i }
        }
      }
    }
    return best
  }

  const drag = useRef(null)
  const onPointerDown = (e) => { drag.current = { x: e.clientX, y: e.clientY, vx: view.current.x, vy: view.current.y, moved: false } }
  const onPointerMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top
    if (drag.current) {
      const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true
      view.current.x = drag.current.vx + dx; view.current.y = drag.current.vy + dy
      requestDraw()
      return
    }
    const i = nearest(sx, sy)
    hoverRef.current = i
    const m = mapRef.current, t = titlesRef.current
    setHover(i >= 0 ? { sx, sy, title: t?.[i]?.[0], year: t?.[i]?.[1], journal: t?.[i]?.[2], domain: m.domains[m.d[i]], color: m.colors[m.d[i]] } : null)
    requestDraw()
  }
  const onPointerUp = () => {
    if (drag.current && !drag.current.moved && hoverRef.current >= 0) openPaper(String(mapRef.current.pmid[hoverRef.current]))
    drag.current = null
  }
  const zoomAt = (factor, sx, sy) => {
    const c = canvasRef.current, k0 = view.current.k, k1 = Math.min(40, Math.max(0.6, k0 * factor))
    const cx = sx - c.clientWidth * 0.5, cy = sy - c.clientHeight * 0.53
    view.current = { x: cx - (cx - view.current.x) * (k1 / k0), y: cy - (cy - view.current.y) * (k1 / k0), k: k1 }
    requestDraw()
  }
  useEffect(() => {
    const c = canvasRef.current
    const onWheel = (e) => { e.preventDefault(); const r = c.getBoundingClientRect(); zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top) }
    c.addEventListener('wheel', onWheel, { passive: false })
    return () => c.removeEventListener('wheel', onWheel)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const flyTo = (indices) => {
    const m = mapRef.current, c = canvasRef.current
    const xs = indices.map((i) => m.x[i]), ys = indices.map((i) => m.y[i])
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length
    const spread = Math.max(0.08, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys))
    const k = Math.min(8, Math.max(1, 1.2 / spread)), S = Math.min(c.clientWidth, c.clientHeight) * 0.44 * k
    const target = { k, x: -mx * S, y: my * S }, from = { ...view.current }, t0 = performance.now()
    const step = (now) => {
      const p = Math.min(1, (now - t0) / 900), e = 1 - Math.pow(1 - p, 3)
      view.current = { x: from.x + (target.x - from.x) * e, y: from.y + (target.y - from.y) * e, k: from.k + (target.k - from.k) * e }
      draw()
      if (p < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }
  const runSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setSearching(true)
    try {
      const res = await api.search({ q: query.trim(), k: 50 })
      const idx = res.results.map((r) => mapRef.current.index.get(r.pmid)).filter((i) => i !== undefined)
      highlight.current = new Set(idx)
      setFound({ q: query.trim(), n: idx.length, top: res.results.slice(0, 5) })
      if (idx.length) flyTo(idx)
      requestDraw()
    } catch {
      setFound({ q: query.trim(), n: 0, error: true })
    } finally {
      setSearching(false)
    }
  }
  const clearSearch = () => { highlight.current = null; setFound(null); setQuery(''); requestDraw() }
  const toggle = (d) => {
    const s = new Set(hidden.current); s.has(d) ? s.delete(d) : s.add(d)
    hidden.current = s; setHiddenState(s); requestDraw()
  }

  return (
    <main className="relative h-screen overflow-hidden bg-paper pt-16">
      <canvas ref={canvasRef} className="absolute inset-x-0 bottom-0 top-16 h-[calc(100%-4rem)] w-full cursor-crosshair touch-none"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
        onPointerLeave={() => { drag.current = null; hoverRef.current = -1; setHover(null); requestDraw() }} />
      {!map && <div className="absolute inset-0 flex items-center justify-center text-ink-3"><Spinner /></div>}

      <div className="pointer-events-none absolute left-5 top-24 w-[min(380px,calc(100%-2.5rem))] sm:left-8">
        <div className="pointer-events-auto bg-paper/90 pb-3">
          <p className="kicker">Plate I</p>
          <h1 className="mt-1 font-serif text-[30px] leading-tight">The atlas of biomedical research</h1>
          <p className="caption mt-2">{map ? `${fmt(map.n)} studies placed by meaning. Drag to move, scroll to magnify, click a point to read the study.` : 'Preparing the plate…'}</p>
          <form onSubmit={runSearch} className="mt-4 flex items-end gap-3">
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Locate a question, e.g. sepsis biomarkers"
              className="w-full border-0 border-b border-ink/70 bg-transparent py-1.5 font-serif text-[17px] outline-none placeholder:text-ink-4 focus:border-hema" />
            <button className="btn-ink shrink-0 py-1.5">{searching ? <Spinner className="h-3.5 w-3.5" /> : 'Locate'}</button>
          </form>
          {found && (
            <div className="mt-3 border-l-2 border-eosin pl-3">
              <p className="meta">{found.error ? 'Search is unavailable while the server wakes. Try again shortly.' : `${found.n} nearest studies to “${found.q}”, ringed.`} <button onClick={clearSearch} className="link">Clear</button></p>
              <ol className="mt-2 space-y-1.5">
                {found.top?.map((r, i) => <li key={r.pmid} className="grid grid-cols-[1.4rem_1fr]"><span className="num">{i + 1}.</span><button onClick={() => openPaper(r.pmid)} className="line-clamp-2 text-left font-serif text-[14.5px] leading-snug hover:text-hema">{r.title}</button></li>)}
              </ol>
            </div>
          )}
        </div>
      </div>

      {map && (
        <aside data-lenis-prevent className="absolute bottom-5 right-5 max-h-[52vh] w-[250px] overflow-y-auto border-t border-ink bg-paper/90 pt-2 max-sm:hidden">
          <p className="font-sans text-[12px] font-semibold">Key: fields <span className="font-normal text-ink-3">(click to hide)</span></p>
          <ul className="mt-1">
            {map.domains.map((d, i) => (
              <li key={d}>
                <button onClick={() => toggle(i)} className={`flex w-full items-center gap-2 py-[3px] text-left font-serif text-[13.5px] ${hiddenState.has(i) ? 'text-ink-4 line-through' : 'text-ink-2 hover:text-ink'}`}>
                  <span className="h-2 w-2 rounded-full" style={{ background: hiddenState.has(i) ? 'transparent' : map.colors[i], boxShadow: `inset 0 0 0 1px ${map.colors[i]}` }} />{d}
                </button>
              </li>
            ))}
          </ul>
        </aside>
      )}

      <div className="absolute bottom-5 left-5 flex border border-ink bg-paper sm:left-8">
        {[[Plus, () => zoomAt(1.5, canvasRef.current.clientWidth / 2, canvasRef.current.clientHeight / 2), 'Magnify'],
          [Minus, () => zoomAt(1 / 1.5, canvasRef.current.clientWidth / 2, canvasRef.current.clientHeight / 2), 'Reduce'],
          [RotateCcw, () => { view.current = { x: 0, y: 0, k: 1 }; requestDraw() }, 'Reset']].map(([Icon, fn, label], i) => (
          <button key={label} onClick={fn} title={label} aria-label={label} className={`p-2.5 text-ink hover:bg-ink hover:text-paper ${i ? 'border-l border-ink' : ''}`}><Icon className="h-4 w-4" /></button>
        ))}
      </div>

      {hover && (
        <div className="pointer-events-none absolute z-10 w-[300px] border-l-2 border-ink bg-paper/95 py-2 pl-3 pr-2"
          style={{ left: Math.min(hover.sx + 16, (canvasRef.current?.clientWidth || 0) - 310), top: hover.sy + 64 + 14 }}>
          <p className="font-serif text-[15px] leading-snug">{hover.title || 'Loading title…'}</p>
          <p className="mt-1 flex items-center gap-1.5 font-sans text-[11px] text-ink-3"><span className="h-2 w-2 rounded-full" style={{ background: hover.color }} />{hover.domain}{hover.year ? `, ${hover.year}` : ''}{hover.journal ? `. ${hover.journal}` : ''}</p>
        </div>
      )}
      {drawer}
    </main>
  )
}
