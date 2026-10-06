import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Minus, Plus, RotateCcw, Search as SearchIcon, X } from 'lucide-react'
import { api } from '../lib/api'
import { fmt, loadMap } from '../lib/mapData'
import { Spinner, setDomainColors, usePaperDrawer } from '../components/ui'

const CELL = 0.025 // spatial hash cell size in map units

export default function Atlas() {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const mapRef = useRef(null)
  const view = useRef({ x: 0, y: 0, k: 1 })
  const grid = useRef(new Map())
  const hoverRef = useRef(-1)
  const highlight = useRef(null) // Set of indices
  const hidden = useRef(new Set())
  const drawQueued = useRef(false)
  const [map, setMap] = useState(null)
  const [hover, setHover] = useState(null)
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [found, setFound] = useState(null)
  const [hiddenState, setHiddenState] = useState(new Set())
  const [openPaper, drawer] = usePaperDrawer()

  const draw = useCallback(() => {
    drawQueued.current = false
    const m = mapRef.current, canvas = canvasRef.current
    if (!m || !canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = canvas.clientWidth, h = canvas.clientHeight
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const S = Math.min(w, h) * 0.45 * view.current.k
    const ox = w / 2 + view.current.x, oy = h / 2 + view.current.y
    const hl = highlight.current
    const r = Math.max(1.1, Math.min(3.2, 1.2 * Math.sqrt(view.current.k)))
    ctx.globalCompositeOperation = 'lighter'
    // batch by domain for speed
    for (let d = 0; d < m.domains.length; d++) {
      if (hidden.current.has(d)) continue
      ctx.fillStyle = m.colors[d]
      ctx.globalAlpha = hl ? 0.18 : 0.78
      ctx.beginPath()
      for (let i = 0; i < m.n; i++) {
        if (m.d[i] !== d) continue
        const px = ox + m.x[i] * S, py = oy - m.y[i] * S
        if (px < -5 || py < -5 || px > w + 5 || py > h + 5) continue
        ctx.moveTo(px + r, py)
        ctx.arc(px, py, r, 0, Math.PI * 2)
      }
      ctx.fill()
    }
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1
    if (hl) {
      hl.forEach((i) => {
        const px = ox + m.x[i] * S, py = oy - m.y[i] * S
        ctx.beginPath(); ctx.arc(px, py, r + 3.5, 0, Math.PI * 2)
        ctx.strokeStyle = '#E5482B'; ctx.lineWidth = 1.5; ctx.stroke()
        ctx.beginPath(); ctx.arc(px, py, r + 0.8, 0, Math.PI * 2)
        ctx.fillStyle = '#FBF9F4'; ctx.fill()
      })
    }
    const hv = hoverRef.current
    if (hv >= 0) {
      const px = ox + m.x[hv] * S, py = oy - m.y[hv] * S
      ctx.beginPath(); ctx.arc(px, py, r + 5, 0, Math.PI * 2)
      ctx.strokeStyle = '#FBF9F4'; ctx.lineWidth = 1.5; ctx.stroke()
    }
    // topic labels when zoomed enough to read them
    ctx.font = '500 10px "IBM Plex Mono", monospace'
    ctx.textAlign = 'center'
    // Largest topics first; skip labels that would collide with (or duplicate) one already drawn
    const placed = [], seen = new Set()
    ;[...m.topics].sort((a, b) => b.size - a.size).forEach((t) => {
      const px = ox + t.x * S, py = oy - t.y * S
      if (px < 0 || py < 0 || px > w || py > h) return
      const words = t.label.split(' · ')
      const label = words.slice(0, 2).join(' · ').toUpperCase()
      const key = [...words.slice(0, 2)].sort().join('|')
      if (seen.has(key)) return
      const tw = ctx.measureText(label).width
      if (placed.some((p) => Math.abs(p.x - px) < (p.w + tw) / 2 + 22 && Math.abs(p.y - py) < 24)) return
      placed.push({ x: px, y: py, w: tw }); seen.add(key)
      ctx.fillStyle = 'rgba(14,26,21,0.72)'
      ctx.beginPath(); ctx.roundRect(px - tw / 2 - 8, py - 10, tw + 16, 20, 10); ctx.fill()
      ctx.fillStyle = 'rgba(242,238,230,0.85)'
      ctx.fillText(label, px, py + 3.5)
    })
  }, [])

  const requestDraw = useCallback(() => {
    if (!drawQueued.current) { drawQueued.current = true; requestAnimationFrame(draw) }
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
    })
    window.addEventListener('resize', requestDraw)
    window.__lenis?.stop()
    return () => { window.removeEventListener('resize', requestDraw); window.__lenis?.start() }
  }, [requestDraw])

  const toMap = (sx, sy) => {
    const c = canvasRef.current
    const S = Math.min(c.clientWidth, c.clientHeight) * 0.45 * view.current.k
    return [(sx - c.clientWidth / 2 - view.current.x) / S, -(sy - c.clientHeight / 2 - view.current.y) / S, S]
  }

  const nearest = (sx, sy) => {
    const m = mapRef.current
    const [mx, my, S] = toMap(sx, sy)
    const rad = 8 / S
    const cx = Math.floor(mx / CELL), cy = Math.floor(my / CELL), span = Math.ceil(rad / CELL)
    let best = -1, bd = rad * rad
    for (let gx = cx - span; gx <= cx + span; gx++) {
      for (let gy = cy - span; gy <= cy + span; gy++) {
        const cell = grid.current.get(`${gx},${gy}`)
        if (!cell) continue
        for (const i of cell) {
          if (hidden.current.has(m.d[i])) continue
          const dx = m.x[i] - mx, dy = m.y[i] - my, dd = dx * dx + dy * dy
          if (dd < bd) { bd = dd; best = i }
        }
      }
    }
    return best
  }

  // pointer interactions: drag to pan, wheel to zoom, hover, click
  const drag = useRef(null)
  const onPointerDown = (e) => { drag.current = { x: e.clientX, y: e.clientY, vx: view.current.x, vy: view.current.y, moved: false } }
  const onPointerMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    if (drag.current) {
      const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true
      view.current.x = drag.current.vx + dx; view.current.y = drag.current.vy + dy
      requestDraw()
      return
    }
    const i = nearest(e.clientX - rect.left, e.clientY - rect.top)
    if (i !== hoverRef.current) {
      hoverRef.current = i
      const m = mapRef.current
      setHover(i >= 0 ? { i, sx: e.clientX - rect.left, sy: e.clientY - rect.top, domain: m.domains[m.d[i]], topic: m.topics[m.t[i]]?.label, color: m.colors[m.d[i]] } : null)
      requestDraw()
    } else if (i >= 0) {
      setHover((h) => h && { ...h, sx: e.clientX - rect.left, sy: e.clientY - rect.top })
    }
  }
  const onPointerUp = () => {
    if (drag.current && !drag.current.moved && hoverRef.current >= 0) openPaper(String(mapRef.current.pmid[hoverRef.current]))
    drag.current = null
  }
  const zoomAt = (factor, sx, sy) => {
    const c = canvasRef.current
    const k0 = view.current.k
    const k1 = Math.min(40, Math.max(0.6, k0 * factor))
    const cx = sx - c.clientWidth / 2, cy = sy - c.clientHeight / 2
    view.current.x = cx - (cx - view.current.x) * (k1 / k0)
    view.current.y = cy - (cy - view.current.y) * (k1 / k0)
    view.current.k = k1
    requestDraw()
  }
  useEffect(() => {
    const c = canvasRef.current
    const onWheel = (e) => {
      e.preventDefault()
      const rect = c.getBoundingClientRect()
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - rect.left, e.clientY - rect.top)
    }
    c.addEventListener('wheel', onWheel, { passive: false })
    return () => c.removeEventListener('wheel', onWheel)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const flyTo = (indices) => {
    const m = mapRef.current, c = canvasRef.current
    const xs = indices.map((i) => m.x[i]), ys = indices.map((i) => m.y[i])
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length
    const spread = Math.max(0.08, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys))
    const target = { k: Math.min(8, Math.max(1, 1.2 / spread)), x: 0, y: 0 }
    const S = Math.min(c.clientWidth, c.clientHeight) * 0.45 * target.k
    target.x = -mx * S; target.y = my * S
    const from = { ...view.current }, t0 = performance.now()
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
      const res = await api.search({ q: query.trim(), k: 50, mode: 'hybrid' })
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
  const reset = () => { view.current = { x: 0, y: 0, k: 1 }; requestDraw() }
  const toggle = (d) => {
    const s = new Set(hidden.current)
    s.has(d) ? s.delete(d) : s.add(d)
    hidden.current = s; setHiddenState(s); requestDraw()
  }
  const soloAll = () => { hidden.current = new Set(); setHiddenState(new Set()); requestDraw() }

  return (
    <main ref={wrapRef} className="relative h-screen overflow-hidden bg-moss text-bone" data-nav-dark>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
        onPointerLeave={() => { drag.current = null; hoverRef.current = -1; setHover(null); requestDraw() }} />
      {!map && <div className="absolute inset-0 flex items-center justify-center"><Spinner className="h-6 w-6 text-bone/60" /></div>}

      {/* Header + search */}
      <div className="pointer-events-none absolute left-0 right-0 top-24 px-5 sm:px-8">
        <div className="pointer-events-auto max-w-md">
          <h1 className="h-display text-4xl sm:text-5xl">The <span className="serif-i text-signal-soft">Atlas</span></h1>
          <p className="mt-2 text-sm text-bone/55">{map ? `${fmt(map.n)} studies, positioned by meaning. Drag to pan, scroll to zoom, click a point to read it.` : 'Loading the map…'}</p>
          <form onSubmit={runSearch} className="mt-5 flex items-center gap-2 rounded-full border border-white/15 bg-moss-2/80 p-1.5 pl-4 backdrop-blur">
            <SearchIcon className="h-4 w-4 text-bone/50" />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Light up a topic, e.g. sepsis biomarkers"
              className="min-w-0 flex-1 bg-transparent py-1.5 text-sm text-bone outline-none placeholder:text-bone/35" />
            {found && <button type="button" onClick={clearSearch} className="p-1 text-bone/50 hover:text-bone" aria-label="Clear"><X className="h-4 w-4" /></button>}
            <button className="btn bg-bone px-4 py-1.5 text-moss hover:bg-white">{searching ? <Spinner className="h-3.5 w-3.5" /> : 'Find'}</button>
          </form>
          {found && (
            <div className="mt-3 rounded-2xl border border-white/10 bg-moss-2/85 p-4 backdrop-blur animate-rise">
              <p className="text-xs text-bone/60">{found.error ? 'Search unavailable right now (server waking up?).' : `${found.n} closest studies for “${found.q}”`}</p>
              <ul className="mt-2 space-y-1.5">
                {found.top?.map((r) => (
                  <li key={r.pmid}><button onClick={() => openPaper(r.pmid)} className="line-clamp-1 text-left text-sm text-bone/85 hover:text-bone">{r.title}</button></li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Legend */}
      {map && (
        <div data-lenis-prevent className="absolute bottom-5 left-5 right-5 max-h-[34vh] overflow-y-auto rounded-2xl border border-white/10 bg-moss-2/80 p-3 backdrop-blur sm:left-auto sm:w-[320px]">
          <div className="mb-2 flex items-center justify-between px-1">
            <p className="eyebrow text-bone/45">Fields</p>
            {hiddenState.size > 0 && <button onClick={soloAll} className="text-xs text-bone/60 hover:text-bone">Show all</button>}
          </div>
          <div className="grid grid-cols-1 gap-0.5">
            {map.domains.map((d, i) => (
              <button key={d} onClick={() => toggle(i)} className={`flex items-center gap-2.5 rounded-lg px-2 py-1 text-left text-xs transition ${hiddenState.has(i) ? 'text-bone/30' : 'text-bone/80 hover:bg-white/5'}`}>
                <span className="h-2 w-2 rounded-full" style={{ background: hiddenState.has(i) ? 'transparent' : map.colors[i], boxShadow: `inset 0 0 0 1px ${map.colors[i]}` }} />
                {d}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Zoom controls */}
      <div className="absolute bottom-5 left-5 hidden flex-col overflow-hidden rounded-2xl border border-white/10 bg-moss-2/80 backdrop-blur sm:flex">
        {[[Plus, () => zoomAt(1.5, canvasRef.current.clientWidth / 2, canvasRef.current.clientHeight / 2), 'Zoom in'],
          [Minus, () => zoomAt(1 / 1.5, canvasRef.current.clientWidth / 2, canvasRef.current.clientHeight / 2), 'Zoom out'],
          [RotateCcw, reset, 'Reset view']].map(([Icon, fn, label]) => (
          <button key={label} onClick={fn} className="p-3 text-bone/70 hover:bg-white/5 hover:text-bone" aria-label={label}><Icon className="h-4 w-4" /></button>
        ))}
      </div>

      {/* Hover tooltip */}
      {hover && (
        <div className="pointer-events-none absolute z-10 max-w-[260px] rounded-xl border border-white/10 bg-moss/95 px-3 py-2 text-xs shadow-xl"
          style={{ left: Math.min(hover.sx + 14, (canvasRef.current?.clientWidth || 0) - 270), top: hover.sy + 14 }}>
          <p className="flex items-center gap-2 text-bone"><span className="h-2 w-2 rounded-full" style={{ background: hover.color }} />{hover.domain}</p>
          <p className="mt-1 text-bone/55">{hover.topic}</p>
          <p className="mt-1 font-mono text-[10px] text-bone/40">Click to open</p>
        </div>
      )}
      {drawer}
    </main>
  )
}
