import React, { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { ArrowDown, ArrowUpRight } from 'lucide-react'
import { useScrollProgress } from '../lib/hooks'
import { fmt } from '../lib/mapData'

/*
  Scroll-driven "explode" scene. Every dot is a real paper.
    0.00-0.14  papers packed into a slowly rotating sphere (domains form coloured bands)
    0.12-0.55  the sphere bursts outward, staggered per paper
    0.50-0.88  fragments settle into the paper's true position on the UMAP research map
    0.86-1.00  topic labels and calls to action fade in
*/

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v))
const easeOutExpo = (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t))
const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3)
const band = (p, a, b, fade = 0.05) => clamp((p - a) / fade) * clamp((b - p) / fade)

function makeSprite(color, size) {
  const c = document.createElement('canvas')
  c.width = c.height = size
  const g = c.getContext('2d')
  const r = size / 2
  const grad = g.createRadialGradient(r, r, 0, r, r, r)
  grad.addColorStop(0, color)
  grad.addColorStop(0.35, color)
  grad.addColorStop(1, 'rgba(0,0,0,0)')
  g.fillStyle = grad
  g.fillRect(0, 0, size, size)
  return c
}

export default function ExplodeHero({ map }) {
  const sectionRef = useRef(null)
  const canvasRef = useRef(null)
  const stageRefs = useRef([])
  const labelsRef = useRef(null)
  const progress = useScrollProgress(sectionRef)

  useEffect(() => {
    if (!map) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const small = window.innerWidth < 768
    const step = small ? 2 : 1 // halve the point count on phones
    const n = map.n
    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    // Sphere layout: papers sorted by domain so each field becomes a band on the globe
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => map.d[a] - map.d[b] || map.y[a] - map.y[b])
    const sx = new Float32Array(n), sy = new Float32Array(n), sz = new Float32Array(n)
    const golden = Math.PI * (3 - Math.sqrt(5))
    order.forEach((idx, k) => {
      const yy = 1 - (k / (n - 1)) * 2
      const rr = Math.sqrt(1 - yy * yy)
      const th = golden * k
      sx[idx] = Math.cos(th) * rr; sy[idx] = yy; sz[idx] = Math.sin(th) * rr
    })
    // Burst targets and per-paper stagger
    const bx = new Float32Array(n), by = new Float32Array(n), delay = new Float32Array(n), delay2 = new Float32Array(n)
    let seed = 7
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    for (let i = 0; i < n; i++) {
      const ang = Math.atan2(sy[i], sx[i]) + (rnd() - 0.5) * 0.9
      const dist = 0.45 + Math.pow(rnd(), 0.8) * 0.85
      bx[i] = Math.cos(ang) * dist; by[i] = Math.sin(ang) * dist
      delay[i] = rnd(); delay2[i] = (map.d[i] / map.domains.length) * 0.6 + rnd() * 0.4
    }
    const sprites = map.colors.map((c) => makeSprite(c, 24))
    const bone = makeSprite('#F2EEE6', 24)

    let w = 0, h = 0, R = 0, M = 0
    const resize = () => {
      w = canvas.clientWidth; h = canvas.clientHeight
      canvas.width = w * dpr; canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      R = Math.min(w, h) * (small ? 0.3 : 0.25)
      M = Math.min(w * 0.46, h * 0.42)
      placeLabels()
    }
    // Topic labels (largest topics) positioned on the final map
    const labelEls = []
    const placeLabels = () => {
      const box = labelsRef.current
      if (!box) return
      box.innerHTML = ''
      labelEls.length = 0
      const placed = []
      const top = [...map.topics].sort((a, b) => b.size - a.size).filter((t) => {
        const text = t.label.split(' · ').slice(0, 2).join(' · ')
        const bw = text.length * 6.6 + 24, bh = 26
        const bx = w / 2 + t.x * M, by = h / 2 - t.y * M
        if (placed.some((p) => Math.abs(p.x - bx) < (p.w + bw) / 2 + 6 && Math.abs(p.y - by) < bh + 4)) return false
        placed.push({ x: bx, y: by, w: bw })
        return true
      }).slice(0, small ? 6 : 12)
      top.forEach((t) => {
        const el = document.createElement('div')
        el.className = 'absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-full border border-white/15 bg-moss/70 px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.12em] text-bone/80 backdrop-blur'
        el.textContent = t.label.split(' · ').slice(0, 2).join(' · ')
        el.style.left = `${w / 2 + t.x * M}px`
        el.style.top = `${h / 2 - t.y * M}px`
        box.appendChild(el)
        labelEls.push(el)
      })
    }
    resize()
    window.addEventListener('resize', resize)

    let sp = reduced ? 1 : 0
    let raf
    const t0 = performance.now()
    const frame = (now) => {
      const target = reduced ? 1 : progress.current
      sp += (target - sp) * 0.1
      const p = sp
      const time = (now - t0) / 1000
      const rot = time * 0.18 + p * 2.4
      const cr = Math.cos(rot), srr = Math.sin(rot)
      const tilt = 0.32, ct = Math.cos(tilt), st = Math.sin(tilt)
      const breathe = 1 + Math.sin(time * 1.3) * 0.012

      ctx.clearRect(0, 0, w, h)
      ctx.globalCompositeOperation = 'lighter'
      const cx = w / 2
      const lift = (1 - clamp((p - 0.1) / 0.2)) * h * 0.09 // sphere sits higher, above the headline
      const cy = h / 2 - lift
      for (let i = 0; i < n; i += step) {
        // rotating, tilted sphere with perspective
        const x1 = sx[i] * cr + sz[i] * srr
        const z1 = -sx[i] * srr + sz[i] * cr
        const y1 = sy[i] * ct - z1 * st
        const z2 = sy[i] * st + z1 * ct
        const persp = 1.6 / (2.6 - z2)
        const ax = x1 * persp * R * breathe, ay = y1 * persp * R * breathe
        // explode
        const u = easeOutExpo(clamp((p - 0.12 - delay[i] * 0.12) / 0.3))
        const B = Math.max(w, h) * 0.5
        const ex = ax + (bx[i] * B - ax) * u
        const ey = ay + (by[i] * B - ay) * u
        // settle onto the research map
        const v = easeOutCubic(clamp((p - 0.44 - delay2[i] * 0.14) / 0.3))
        const v2 = easeInOutCubic(v)
        const px = ex + (map.x[i] * M - ex) * v2
        const py = ey + (-map.y[i] * M + lift - ey) * v2
        const depth = 0.25 + 0.75 * ((z2 + 1) / 2)
        const alpha = (1 - u) * depth * 0.55 + u * (1 - v) * 0.7 + v * 0.55
        const size = 3.4 + u * (1 - v) * 3 + v * 0.3
        const x0 = cx + px - size / 2, y0 = cy + py - size / 2
        const colorMix = clamp(u * 1.6) // monochrome sphere, colour emerges as it bursts
        if (colorMix < 1) {
          ctx.globalAlpha = clamp(alpha * (1 - colorMix), 0, 1)
          ctx.drawImage(bone, x0, y0, size, size)
        }
        if (colorMix > 0) {
          ctx.globalAlpha = clamp(alpha * colorMix, 0, 1)
          ctx.drawImage(sprites[map.d[i]], x0, y0, size, size)
        }
      }
      ctx.globalAlpha = 1
      ctx.globalCompositeOperation = 'source-over'

      // text stages
      const st0 = stageRefs.current
      const show = [band(p, -1, 0.12, 0.05), band(p, 0.18, 0.46), band(p, 0.56, 0.82), clamp((p - 0.86) / 0.06)]
      show.forEach((o, k) => {
        const el = st0[k]
        if (!el) return
        el.style.opacity = o
        el.style.transform = `translateY(${(1 - o) * 18}px)`
        el.style.pointerEvents = o > 0.6 ? 'auto' : 'none'
      })
      const lo = clamp((p - 0.8) / 0.08)
      labelEls.forEach((el, k) => { el.style.opacity = clamp(lo * 1.4 - k * 0.04) })
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [map, progress])

  const stats = map?.stats
  return (
    <section ref={sectionRef} className="relative h-[420vh] bg-moss" data-nav-dark>
      <div className="sticky top-0 h-screen overflow-hidden grain">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(229,72,43,0.07),transparent_60%)]" />
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
        <div ref={labelsRef} className="pointer-events-none absolute inset-0" />

        {/* Stage 1 */}
        <div ref={(el) => (stageRefs.current[0] = el)} className="absolute inset-x-0 bottom-[9vh] text-center text-bone">
          <p className="eyebrow text-bone/50">Biomedical research intelligence</p>
          <h1 className="h-display mx-auto mt-4 max-w-4xl px-6 text-[13vw] leading-[0.92] sm:text-7xl md:text-[88px]">
            The literature of medicine, <span className="serif-i text-signal-soft">mapped.</span>
          </h1>
          <p className="mt-6 flex items-center justify-center gap-2 text-sm text-bone/50">
            <ArrowDown className="h-4 w-4 animate-bounce" /> Scroll to break it open
          </p>
        </div>

        {/* Stage 2 */}
        <div ref={(el) => (stageRefs.current[1] = el)} className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center px-6 text-center text-bone opacity-0">
          <p className="h-display text-6xl sm:text-8xl md:text-[120px]">{fmt(stats?.papers)}</p>
          <p className="mt-2 font-serif text-2xl italic text-bone/80 sm:text-3xl">studies across {map?.domains.length || 24} fields of medicine</p>
          <p className="mt-6 max-w-md text-sm leading-6 text-bone/50">
            Each point is one PubMed abstract, embedded by a biomedical language model fine-tuned on this corpus.
          </p>
        </div>

        {/* Stage 3 */}
        <div ref={(el) => (stageRefs.current[2] = el)} className="pointer-events-none absolute inset-x-0 top-[14vh] px-6 text-center text-bone opacity-0">
          <p className="eyebrow text-bone/50">Unsupervised topic discovery</p>
          <h2 className="h-display mx-auto mt-3 max-w-3xl text-4xl sm:text-6xl">
            Related science settles into <span className="serif-i text-signal-soft">neighbourhoods.</span>
          </h2>
        </div>

        {/* Stage 4 */}
        <div ref={(el) => (stageRefs.current[3] = el)} className="absolute inset-x-0 bottom-[7vh] flex flex-col items-center gap-5 px-6 text-center text-bone opacity-0">
          <p className="max-w-lg text-bone/60">
            {stats?.topics} research fronts discovered automatically, from {fmt(stats?.journals)} journals, {stats?.years?.[0]} to {stats?.years?.[1]}.
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <Link to="/atlas" className="btn bg-bone text-moss hover:bg-white">Explore the Atlas <ArrowUpRight className="h-4 w-4" /></Link>
            <Link to="/search" className="btn border border-white/20 text-bone hover:bg-white/10">Search the literature</Link>
          </div>
        </div>
      </div>
    </section>
  )
}
