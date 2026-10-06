import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { fmt, loadMap } from '../lib/mapData'
import { useReveal } from '../lib/hooks'
import { DOMAIN_COLORS, Dot, PageHead, Sparkline, setDomainColors } from '../components/ui'

const YEARS = Array.from({ length: 12 }, (_, i) => String(2014 + i)) // complete years only

function LineChart({ series, years, highlight }) {
  const max = Math.max(1, ...series.flatMap((s) => years.map((y) => s.data[y] || 0)))
  const W = 1000, H = 360, P = 36
  const x = (i) => P + (i / (years.length - 1)) * (W - 2 * P)
  const y = (v) => H - P - (v / max) * (H - 2 * P)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full">
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <line x1={P} x2={W - P} y1={y(max * f)} y2={y(max * f)} stroke="#141613" strokeOpacity="0.08" />
          <text x={P - 8} y={y(max * f) + 4} textAnchor="end" className="fill-ink-4 font-mono text-[11px]">{max * f >= 1000 ? `${Math.round(max * f / 1000)}k` : Math.round(max * f)}</text>
        </g>
      ))}
      {years.map((yr, i) => (i % 2 === 0) && <text key={yr} x={x(i)} y={H - 10} textAnchor="middle" className="fill-ink-4 font-mono text-[11px]">{yr}</text>)}
      {series.map((s) => {
        const on = !highlight || highlight === s.name
        const d = years.map((yr, i) => `${i ? 'L' : 'M'}${x(i)},${y(s.data[yr] || 0)}`).join(' ')
        return <path key={s.name} d={d} fill="none" stroke={s.color} strokeWidth={on && highlight ? 3 : 1.6}
          strokeOpacity={on ? 1 : 0.12} strokeLinejoin="round" style={{ transition: 'stroke-opacity .3s' }} />
      })}
    </svg>
  )
}

export default function Trends() {
  const [trends, setTrends] = useState(null)
  const [hover, setHover] = useState(null)
  const [error, setError] = useState(null)
  const ref = useReveal([trends])

  useEffect(() => {
    loadMap().then((m) => setDomainColors(m.domains, m.colors))
    api.trends().then(setTrends).catch((e) => setError(e.message))
  }, [])

  const rows = useMemo(() => {
    if (!trends?.pubmed_counts) return []
    return Object.entries(trends.pubmed_counts).map(([name, data]) => {
      const early = (data['2015'] + data['2016'] + data['2017']) / 3
      const late = (data['2023'] + data['2024'] + data['2025']) / 3
      return { name, data, total: YEARS.reduce((a, y) => a + (data[y] || 0), 0), growth: late / Math.max(1, early), latest: data['2025'] }
    }).sort((a, b) => b.growth - a.growth)
  }, [trends])
  const series = rows.map((r) => ({ name: r.name, data: r.data, color: DOMAIN_COLORS[r.name] }))

  return (
    <main className="min-h-screen">
      <div ref={ref} className="container-x pb-28">
        <PageHead eyebrow="Trends" title={<>How biomedicine is <span className="serif-i">moving.</span></>}>
          True publication volumes from PubMed for each field, 2014 to 2025, with the concepts gaining ground fastest in our corpus.
        </PageHead>
        {error && <div className="surface mt-10 p-6 text-signal-ink">{error}. The server may be waking up.</div>}
        {!trends && !error && <div className="skeleton mt-12 h-[420px]" />}

        {trends && (
          <>
            <section className="reveal surface mt-12 p-6 sm:p-8">
              <div className="flex flex-wrap items-end justify-between gap-4">
                <div>
                  <p className="eyebrow">Publications per year, PubMed</p>
                  <p className="mt-2 font-display text-2xl font-medium tracking-tight">{hover || 'All 24 fields'}</p>
                </div>
                <p className="text-sm text-ink-4">Hover a field below to isolate it</p>
              </div>
              <div className="mt-6"><LineChart series={series} years={YEARS} highlight={hover} /></div>
            </section>

            <section className="mt-16">
              <div className="reveal flex items-end justify-between">
                <h2 className="h-display text-3xl sm:text-4xl">Fields ranked by <span className="serif-i">acceleration</span></h2>
                <p className="hidden text-sm text-ink-4 sm:block">Average 2023 to 2025 vs. 2015 to 2017</p>
              </div>
              <div className="mt-6 overflow-hidden rounded-[22px] border border-ink/10 bg-bone-50">
                {rows.map((r, i) => (
                  <div key={r.name} onMouseEnter={() => setHover(r.name)} onMouseLeave={() => setHover(null)}
                    className="reveal grid grid-cols-[28px_1fr_90px] items-center gap-4 border-b border-ink/[0.07] px-5 py-3.5 last:border-0 hover:bg-bone-100 sm:grid-cols-[28px_1.2fr_1fr_110px_110px]"
                    style={{ transitionDelay: `${(i % 8) * 30}ms` }}>
                    <span className="font-mono text-xs text-ink-4">{String(i + 1).padStart(2, '0')}</span>
                    <span className="flex items-center gap-2.5 text-[15px] text-ink"><Dot domain={r.name} className="h-2.5 w-2.5" />{r.name}</span>
                    <span className="hidden sm:block"><Sparkline data={r.data} years={YEARS} className="h-8 w-full" color={DOMAIN_COLORS[r.name]} /></span>
                    <span className="hidden text-right font-mono text-xs text-ink-3 sm:block">{fmt(r.latest)} in 2025</span>
                    <span className={`text-right font-mono text-sm ${r.growth >= 2 ? 'text-signal' : 'text-ink-2'}`}>{r.growth.toFixed(1)}×</span>
                  </div>
                ))}
              </div>
            </section>

            <section className="mt-20 grid gap-12 lg:grid-cols-2">
              <div className="reveal">
                <h2 className="h-display text-3xl">Rising <span className="serif-i">concepts</span></h2>
                <p className="mt-2 text-sm text-ink-3">MeSH terms whose share of papers grew most from 2014 to 2018 vs. 2022 onwards.</p>
                <div className="mt-6 space-y-2">
                  {trends.emerging_mesh.slice(0, 14).map((m) => (
                    <div key={m.term} className="flex items-center gap-4">
                      <span className="w-48 flex-shrink-0 truncate text-sm text-ink-2 sm:w-64">{m.term}</span>
                      <div className="h-1.5 flex-1 rounded-full bg-ink/[0.06]">
                        <div className="h-1.5 rounded-full bg-signal" style={{ width: `${Math.min(100, (m.lift / trends.emerging_mesh[0].lift) * 100)}%` }} />
                      </div>
                      <span className="w-14 text-right font-mono text-xs text-ink-3">{m.lift >= 100 ? 'new' : `${m.lift.toFixed(1)}×`}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="reveal">
                <h2 className="h-display text-3xl">Where it is <span className="serif-i">published</span></h2>
                <p className="mt-2 text-sm text-ink-3">Most frequent journals in the corpus.</p>
                <ol className="mt-6 divide-y divide-ink/[0.07] overflow-hidden rounded-[22px] border border-ink/10 bg-bone-50">
                  {trends.top_journals.slice(0, 12).map(([j, c], i) => (
                    <li key={j} className="flex items-center gap-4 px-5 py-3 text-sm">
                      <span className="font-mono text-xs text-ink-4">{String(i + 1).padStart(2, '0')}</span>
                      <span className="flex-1 italic text-ink-2">{j}</span>
                      <span className="font-mono text-xs text-ink-3">{c}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </section>
          </>
        )}
      </div>
    </main>
  )
}
