import React, { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useScrollProgress } from '../lib/hooks'
import { fmt, loadTitles } from '../lib/mapData'

/*
  The corpus as a DNA double helix. Every point is one real study.
    0.00-0.10  intact helix: drag to spin it, the cursor is a probe the strands bend away from,
               hover any point to read the paper it stands for
    0.10-0.40  the helix unzips from top to bottom; base pairs break, strands separate
    0.22-0.55  freed studies diffuse and take up their field's stain, one by one
    0.50-0.84  they settle into the research map (UMAP of the fine-tuned embeddings)
    0.84-1.00  the map becomes an annotated figure plate; points stay interactive
*/

const HEMA = '#3E2F73'
const EOSIN = '#C9476E'
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v))
const smooth = (t) => t * t * (3 - 2 * t)
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
const window01 = (p, a, b, f = 0.035) => clamp((p - a) / f) * clamp((b - p) / f)

function dot(color, r, dpr) {
  const s = Math.ceil((r * 2 + 2) * dpr)
  const c = document.createElement('canvas')
  c.width = c.height = s
  const g = c.getContext('2d')
  g.fillStyle = color
  g.beginPath()
  g.arc(s / 2, s / 2, r * dpr, 0, Math.PI * 2)
  g.fill()
  return c
}

const STAGES = [
  { a: -1, b: 0.1 },
  { a: 0.13, b: 0.36 },
  { a: 0.39, b: 0.6 },
  { a: 0.63, b: 0.83 },
  { a: 0.86, b: 2 },
]

export default function HelixHero({ map, onOpen }) {
  const sectionRef = useRef(null)
  const canvasRef = useRef(null)
  const tipRef = useRef(null)
  const stageRefs = useRef([])
  const progress = useScrollProgress(sectionRef)
  const titlesRef = useRef(null)

  useEffect(() => {
    // Fetch titles once the hero has painted; they are only needed for hover.
    const load = () => loadTitles().then((t) => { titlesRef.current = t })
    if ('requestIdleCallback' in window) {
      const id = window.requestIdleCallback(load, { timeout: 1500 })
      return () => window.cancelIdleCallback(id)
    }
    const id = setTimeout(load, 600)
    return () => clearTimeout(id)
  }, [])

  useEffect(() => {
    if (!map) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const n = map.n
    let small = window.innerWidth < 900
    const step = window.innerWidth < 600 ? 2 : 1

    // Order studies by field (then by map position) so staining reveals bands along the helix.
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => map.d[a] - map.d[b] || map.y[b] - map.y[a])
    const strand = new Uint8Array(n), ht = new Float32Array(n)
    const jx = new Float32Array(n), jy = new Float32Array(n), rnd = new Float32Array(n), rnd2 = new Float32Array(n)
    let seed = 11
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    const pairs = Math.ceil(n / 2)
    const partner = new Int32Array(n).fill(-1)
    order.forEach((idx, k) => {
      strand[idx] = k % 2
      ht[idx] = Math.floor(k / 2) / (pairs - 1)
      if (k % 2 === 1) { partner[idx] = order[k - 1]; partner[order[k - 1]] = idx }
    })
    for (let i = 0; i < n; i++) {
      const a = rand() * Math.PI * 2, r = Math.sqrt(rand())
      jx[i] = Math.cos(a) * r; jy[i] = Math.sin(a) * r
      rnd[i] = rand(); rnd2[i] = rand()
    }

    const sprites = [HEMA, EOSIN, ...map.colors].map((c) => dot(c, 1.7, dpr))
    const spriteS = sprites[0].width / dpr
    const px = new Float32Array(n), py = new Float32Array(n), ox = new Float32Array(n), oy = new Float32Array(n)
    const bxs = new Float32Array(n), bys = new Float32Array(n) // resting positions (before the probe pushes them)

    let w = 0, h = 0, hx = 0, Hh = 0, Rh = 0, mx = 0, my = 0, M = 0, top = 0
    let labels = []
    const resize = () => {
      w = canvas.clientWidth; h = canvas.clientHeight
      small = w < 900
      canvas.width = w * dpr; canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      hx = small ? w / 2 : w * 0.64
      Hh = h * (small ? 0.46 : 0.84)
      top = small ? h * 0.1 : h / 2 - Hh / 2
      Rh = Math.min(small ? w * 0.2 : w * 0.085, 118)
      mx = small ? w / 2 : w * 0.665
      my = small ? h * 0.4 : h * 0.5
      M = small ? Math.min(w * 0.44, h * 0.3) : Math.min(w * 0.27, h * 0.4)
      // figure annotations: largest topics, collision-free, leader lines to their centroid
      const placed = []
      labels = [...map.topics].sort((a, b) => b.size - a.size).map((t) => {
        const cx = mx + t.x * M, cy = my - t.y * M
        const side = t.x >= 0 ? 1 : -1
        const lx = cx + side * (28 + 18 * Math.abs(t.y)), ly = cy - 18
        const text = t.label.split(' · ').slice(0, 2).join(', ')
        const tw = text.length * 6.2
        const box = { x: side > 0 ? lx : lx - tw, y: ly - 10, w: tw, h: 14 }
        if (placed.some((b) => box.x < b.x + b.w + 8 && box.x + box.w + 8 > b.x && box.y < b.y + b.h + 4 && box.y + box.h + 4 > b.y)) return null
        if (box.x < (small ? 4 : w * 0.41) || box.x + box.w > w - 8) return null
        placed.push(box)
        return { cx, cy, lx, ly, side, text }
      }).filter(Boolean).slice(0, small ? 5 : 11)
    }
    resize()
    window.addEventListener('resize', resize)

    // Pointer: probe position, drag-to-spin with inertia, hover + click
    const pointer = { x: -9999, y: -9999, inside: false, down: false, lastX: 0, moved: 0 }
    let spin = 0, spinV = 0, hoverIdx = -1
    const onMove = (e) => {
      const r = canvas.getBoundingClientRect()
      const x = e.clientX - r.left, y = e.clientY - r.top
      if (pointer.down) { spinV += (x - pointer.lastX) * 0.0022; pointer.moved += Math.abs(x - pointer.lastX) }
      pointer.x = x; pointer.y = y; pointer.lastX = x; pointer.inside = true
    }
    const onDown = (e) => { pointer.down = true; pointer.lastX = e.clientX - canvas.getBoundingClientRect().left; pointer.moved = 0 }
    const onUp = () => {
      if (pointer.down && pointer.moved < 4 && hoverIdx >= 0) onOpen(String(map.pmid[hoverIdx]))
      pointer.down = false
    }
    const onLeave = () => { pointer.inside = false; pointer.down = false; pointer.x = pointer.y = -9999 }
    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('pointerdown', onDown)
    window.addEventListener('pointerup', onUp)
    canvas.addEventListener('pointerleave', onLeave)

    let sp = reduced ? 1 : 0
    let raf
    const t0 = performance.now()
    const frame = (now) => {
      sp += ((reduced ? 1 : progress.current) - sp) * 0.085
      const p = sp
      const time = (now - t0) / 1000
      spinV *= 0.94
      spin += spinV + 0.0035 * (1 - clamp(p / 0.4))
      const unzipFront = clamp((p - 0.1) / 0.26) * 1.25
      const plate = clamp((p - 0.84) / 0.06)

      ctx.clearRect(0, 0, w, h)

      // positions
      const R = 105, R2 = R * R
      const probe = pointer.inside && !pointer.down
      for (let i = 0; i < n; i += step) {
        const t = ht[i]
        const theta = t * Math.PI * 2 * 5 + spin + (strand[i] ? Math.PI : 0)
        const z = Math.sin(theta)
        const persp = 1 + z * 0.22
        let x = hx + Math.cos(theta) * Rh * persp + jx[i] * 4
        let y = top + t * Hh + jy[i] * 4 + Math.sin(time * 0.8 + t * 6) * 3
        // unzip: strands peel apart from the top down
        const unz = smooth(clamp((unzipFront - t) / 0.16))
        x += (strand[i] ? 1 : -1) * unz * (40 + 90 * rnd[i])
        y += unz * Math.sin(time * 1.6 + rnd2[i] * 6) * 6
        // stream: each study leaves the strand along its own arc to its field's region (top of the helix first)
        const tgx = mx + map.x[i] * M, tgy = my - map.y[i] * M
        const u = clamp((p - 0.22 - t * 0.32 - rnd[i] * 0.05) / 0.26)
        if (u > 0) {
          const e = easeInOut(u)
          const dx = tgx - x, dy = tgy - y
          const bend = (map.d[i] % 2 ? 1 : -1) * (0.22 + rnd2[i] * 0.28)
          const c1x = x + dx * 0.5 - dy * bend, c1y = y + dy * 0.5 + dx * bend
          const ie = 1 - e
          const wob = Math.sin(e * Math.PI) * 6
          x = ie * ie * x + 2 * ie * e * c1x + e * e * tgx + Math.cos(time * 1.3 + rnd[i] * 9) * wob
          y = ie * ie * y + 2 * ie * e * c1y + e * e * tgy + Math.sin(time * 1.1 + rnd2[i] * 9) * wob
        }
        // probe: points bend away from the cursor, springing back when it leaves
        let tx = 0, ty = 0
        if (probe) {
          const dx = x - pointer.x, dy = y - pointer.y, d2 = dx * dx + dy * dy
          if (d2 < R2 && d2 > 0.01) {
            const d = Math.sqrt(d2), f = (1 - d / R) * (1 - d / R) * 34
            tx = (dx / d) * f; ty = (dy / d) * f
          }
        }
        ox[i] += (tx - ox[i]) * 0.18; oy[i] += (ty - oy[i]) * 0.18
        bxs[i] = x; bys[i] = y
        px[i] = x + ox[i]; py[i] = y + oy[i]
      }

      // base pairs (rungs) while the helix is intact
      if (unzipFront < 1.3) {
        ctx.beginPath()
        for (let k = 0; k < n; k += 44) {
          const a = order[k], b = partner[a]
          if (b < 0 || a % step || b % step) continue // on phones only every other point is positioned
          const unz = clamp((unzipFront - ht[a]) / 0.08)
          if (unz > 0.5) continue
          ctx.moveTo(px[a], py[a]); ctx.lineTo(px[b], py[b])
        }
        ctx.strokeStyle = 'rgba(27,25,22,0.16)'
        ctx.lineWidth = 0.6
        ctx.stroke()
      }

      // points: strand colour until each study takes up its field's stain
      for (let i = 0; i < n; i += step) {
        const stained = p > 0.22 + ht[i] * 0.32 + rnd[i] * 0.05 + 0.03 // takes its stain as it leaves the strand
        const theta = ht[i] * Math.PI * 2 * 5 + spin + (strand[i] ? Math.PI : 0)
        const depth = p < 0.3 ? 0.45 + 0.55 * ((Math.sin(theta) + 1) / 2) : 1
        ctx.globalAlpha = depth * 0.8
        ctx.drawImage(sprites[stained ? 2 + map.d[i] : strand[i]], px[i] - spriteS / 2, py[i] - spriteS / 2, spriteS, spriteS)
      }
      ctx.globalAlpha = 1

      // figure annotations with leader lines
      if (plate > 0) {
        ctx.globalAlpha = plate
        ctx.strokeStyle = 'rgba(27,25,22,0.55)'
        ctx.fillStyle = '#1B1916'
        ctx.lineWidth = 0.8
        ctx.font = 'italic 13px "Source Serif 4", Georgia, serif'
        labels.forEach((l) => {
          ctx.beginPath(); ctx.arc(l.cx, l.cy, 2.2, 0, Math.PI * 2); ctx.fill()
          ctx.beginPath(); ctx.moveTo(l.cx, l.cy); ctx.lineTo(l.lx, l.ly + 4); ctx.stroke()
          ctx.textAlign = l.side > 0 ? 'left' : 'right'
          ctx.lineJoin = 'round'; ctx.strokeStyle = 'rgba(247,244,238,0.9)'; ctx.lineWidth = 4
          ctx.strokeText(l.text, l.lx + l.side * 4, l.ly)
          ctx.strokeStyle = 'rgba(27,25,22,0.55)'; ctx.lineWidth = 0.8
          ctx.fillText(l.text, l.lx + l.side * 4, l.ly)
        })
        ctx.globalAlpha = 1
      }

      // hover: nearest study under the probe
      hoverIdx = -1
      if (pointer.inside) {
        let best = 14 * 14 // the study you are touching is the one being pushed aside
        for (let i = 0; i < n; i += step) {
          const dx = bxs[i] - pointer.x, dy = bys[i] - pointer.y, d2 = dx * dx + dy * dy
          if (d2 < best) { best = d2; hoverIdx = i }
        }
        ctx.beginPath()
        ctx.arc(pointer.x, pointer.y, pointer.down ? 18 : 26, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(62,47,115,0.35)'
        ctx.lineWidth = 1
        ctx.stroke()
        if (hoverIdx >= 0) {
          ctx.beginPath(); ctx.arc(px[hoverIdx], py[hoverIdx], 5, 0, Math.PI * 2)
          ctx.strokeStyle = '#1B1916'; ctx.lineWidth = 1.2; ctx.stroke()
        }
      }
      const tip = tipRef.current
      if (tip) {
        const tt = titlesRef.current
        if (hoverIdx >= 0 && tt) {
          const [title, year, journal] = tt[hoverIdx]
          tip.style.opacity = 1
          tip.style.transform = `translate(${Math.min(pointer.x + 18, w - 330)}px, ${Math.min(pointer.y + 18, h - 120)}px)`
          tip.querySelector('[data-t]').textContent = title
          tip.querySelector('[data-m]').textContent = `${map.domains[map.d[hoverIdx]]}, ${year}${journal ? `. ${journal}` : ''}`
          tip.querySelector('[data-c]').style.background = map.colors[map.d[hoverIdx]]
        } else {
          tip.style.opacity = 0
        }
      }
      canvas.style.cursor = hoverIdx >= 0 ? 'pointer' : pointer.down ? 'grabbing' : 'grab'

      // text stages
      STAGES.forEach((s, k) => {
        const el = stageRefs.current[k]
        if (!el) return
        const o = window01(p, s.a, s.b)
        el.style.opacity = o
        el.style.transform = `translateY(${(1 - o) * 12}px)`
        el.style.pointerEvents = o > 0.5 ? 'auto' : 'none'
      })
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointerleave', onLeave)
    }
  }, [map, progress, onOpen])

  const s = map?.stats
  const Stage = ({ k, children, className = '' }) => (
    <div ref={(el) => (stageRefs.current[k] = el)} className={`absolute opacity-0 ${className}`}>{children}</div>
  )
  const col = 'left-5 right-5 bottom-8 sm:left-8 lg:left-[max(2rem,calc((100vw-1180px)/2+2rem))] lg:right-auto lg:bottom-auto lg:top-1/2 lg:w-[380px] lg:-translate-y-1/2'

  return (
    <section ref={sectionRef} className="relative h-[560vh]">
      <div className="sticky top-0 h-screen overflow-hidden">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full touch-pan-y" />

        <Stage k={0} className={col}>
          <div className="bg-paper/85 lg:bg-transparent">
            <p className="kicker">An open atlas of the biomedical literature</p>
            <h1 className="h1 mt-4">The literature of medicine, as a living specimen.</h1>
            <p className="lede mt-5">
              {fmt(s?.papers)} PubMed studies from {map?.domains.length || 24} fields, wound into a single strand.
              Every point is one study: <span className="text-hema">drag to turn it, hover to read it.</span>
            </p>
            <p className="meta mt-8">Scroll to unwind &darr;</p>
          </div>
        </Stage>

        <Stage k={1} className={col}>
          <p className="sec">§ 1</p>
          <h2 className="h2 mt-2">Unwinding the record.</h2>
          <p className="body mt-4">
            The strands hold abstracts published from {s?.years?.[0]} to {s?.years?.[1]}, ordered by field. As they separate, each
            study is read by a language model fine-tuned on this corpus, so that what a paper <em>means</em>, not just the words it
            uses, decides where it goes next.
          </p>
        </Stage>

        <Stage k={2} className={col}>
          <p className="sec">§ 2</p>
          <h2 className="h2 mt-2">A stain for every field.</h2>
          <p className="body mt-4">
            In histology, a stain makes structure visible. Here each of the {map?.domains.length || 24} fields takes its own colour,
            from CRISPR and CAR-T therapy to the gut microbiome and medical imaging.
          </p>
        </Stage>

        <Stage k={3} className={col}>
          <p className="sec">§ 3</p>
          <h2 className="h2 mt-2">Related work finds its neighbours.</h2>
          <p className="body mt-4">
            Positions come from the geometry of meaning. Studies about the same problem settle side by side, even when they
            describe it in different words, and {s?.topics} research fronts emerge without anyone labelling them.
          </p>
        </Stage>

        <Stage k={4} className={col}>
          <p className="caption"><b>Fig. 1 | The landscape of {fmt(s?.papers)} studies, coloured by field.</b> Annotated with the largest
            research fronts. Hover any point to read the study; click to open it.</p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link to="/search" className="btn-ink">Search the atlas</Link>
            <Link to="/atlas" className="btn-rule">Open the full plate</Link>
          </div>
        </Stage>

        {/* hover note, styled like a footnote */}
        <div ref={tipRef} className="pointer-events-none absolute left-0 top-0 w-[310px] border-l-2 border-ink bg-paper/95 py-2 pl-3 pr-2 opacity-0 shadow-[0_1px_0_#D8D1C4] transition-opacity duration-150">
          <p data-t className="font-serif text-[15px] leading-snug text-ink" />
          <p className="mt-1 flex items-center gap-1.5 font-sans text-[11px] text-ink-3"><span data-c className="inline-block h-2 w-2 rounded-full" /><span data-m /></p>
        </div>
      </div>
    </section>
  )
}
