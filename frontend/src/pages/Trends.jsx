import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { fmt, loadMap } from '../lib/mapData'
import { useReveal } from '../lib/hooks'
import { DOMAIN_COLORS, PageHead, SectionHead, setDomainColors } from '../components/ui'

const YEARS = Array.from({ length: 12 }, (_, i) => String(2014 + i))
const LETTERS = 'abcdefghijklmnopqrstuvwx'

// One panel of a small-multiples figure: annual PubMed output for a single field.
function Panel({ name, data, letter, color }) {
  const vals = YEARS.map((y) => data[y] || 0)
  const max = Math.max(1, ...vals)
  const W = 200, H = 84
  const x = (i) => 4 + (i / (YEARS.length - 1)) * (W - 8)
  const y = (v) => H - 14 - (v / max) * (H - 26)
  const line = vals.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const growth = ((data['2023'] + data['2024'] + data['2025']) / 3) / Math.max(1, (data['2015'] + data['2016'] + data['2017']) / 3)
  return (
    <figure className="border-t border-rule pt-3">
      <figcaption className="flex items-baseline justify-between gap-2">
        <span className="font-serif text-[15px] leading-tight"><b className="mr-1.5 font-sans text-[12px] font-semibold">{letter}</b>{name}</span>
        <span className={`num ${growth >= 2 ? 'text-eosin' : ''}`}>{growth.toFixed(1)}×</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 h-auto w-full">
        <line x1="4" x2={W - 4} y1={H - 14} y2={H - 14} stroke="#B9B0A1" strokeWidth="0.8" />
        <path d={`${line} L${x(YEARS.length - 1)},${H - 14} L4,${H - 14} Z`} fill={color} opacity="0.1" />
        <path d={line} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
        <circle cx={x(YEARS.length - 1)} cy={y(vals[vals.length - 1])} r="2.4" fill={color} />
        <text x="4" y={H - 2} className="fill-ink-3 font-mono text-[9px]">2014</text>
        <text x={W - 4} y={H - 2} textAnchor="end" className="fill-ink-3 font-mono text-[9px]">2025: {fmt(vals[vals.length - 1])}</text>
      </svg>
    </figure>
  )
}

export default function Trends() {
  const [trends, setTrends] = useState(null)
  const [error, setError] = useState(null)
  const [order, setOrder] = useState('growth')
  const ref = useReveal([trends, order])

  useEffect(() => {
    loadMap().then((m) => setDomainColors(m.domains, m.colors))
    api.trends().then(setTrends).catch((e) => setError(e.message))
  }, [])

  const fields = useMemo(() => {
    if (!trends?.pubmed_counts) return []
    return Object.entries(trends.pubmed_counts).map(([name, data]) => ({
      name, data,
      growth: ((data['2023'] + data['2024'] + data['2025']) / 3) / Math.max(1, (data['2015'] + data['2016'] + data['2017']) / 3),
      latest: data['2025'],
    })).sort((a, b) => (order === 'growth' ? b.growth - a.growth : order === 'volume' ? b.latest - a.latest : a.name.localeCompare(b.name)))
  }, [trends, order])

  return (
    <main ref={ref} className="page min-h-screen">
      <PageHead kicker="Trends" title="How the fields of biomedicine are moving.">
        Annual output for each field, counted directly from PubMed, alongside the concepts gaining ground fastest in the atlas.
      </PageHead>
      {error && <p className="body mt-10 text-eosin">{error}. The server may be waking up; please try again shortly.</p>}
      {!trends && !error && <div className="mt-10 grid grid-cols-2 gap-6 md:grid-cols-4">{Array.from({ length: 8 }, (_, i) => <div key={i} className="skeleton h-28" />)}</div>}

      {trends && (
        <>
          <section className="mt-12">
            <SectionHead n={1} aside={
              <span className="flex gap-4">{[['growth', 'by growth'], ['volume', 'by volume'], ['name', 'A to Z']].map(([k, l]) => (
                <button key={k} onClick={() => setOrder(k)} className={order === k ? 'text-ink underline decoration-eosin decoration-2 underline-offset-4' : 'hover:text-ink'}>{l}</button>
              ))}</span>
            }>Annual publications, by field</SectionHead>
            <div className="mt-6 grid grid-cols-2 gap-x-6 gap-y-7 md:grid-cols-3 lg:grid-cols-4">
              {fields.map((f, i) => <Panel key={f.name} name={f.name} data={f.data} letter={LETTERS[i]} color={DOMAIN_COLORS[f.name] || '#1B1916'} />)}
            </div>
            <p className="caption mt-6 max-w-3xl"><b>Fig. 2 | Annual PubMed output per field, 2014 to 2025.</b> Each panel has its own vertical
              scale. The figure at right of each title is the mean output in 2023 to 2025 relative to 2015 to 2017.</p>
          </section>

          <section className="reveal mt-20 grid gap-14 lg:grid-cols-2">
            <div>
              <SectionHead n={2}>Rising concepts</SectionHead>
              <table className="mt-3 w-full border-collapse font-sans text-[14px]">
                <thead><tr className="border-b border-ink text-left"><th className="py-2 font-semibold">MeSH concept</th><th className="py-2 text-right font-semibold">Recent studies</th><th className="py-2 text-right font-semibold">Lift</th></tr></thead>
                <tbody>
                  {trends.emerging_mesh.slice(0, 15).map((m) => (
                    <tr key={m.term} className="border-b border-rule">
                      <td className="py-2.5 pr-4 font-serif text-[16px]">{m.term}</td>
                      <td className="py-2.5 text-right tabular-nums text-ink-3">{m.recent_papers}</td>
                      <td className="py-2.5 text-right tabular-nums">{m.lift >= 100 ? 'new' : `${m.lift.toFixed(1)}×`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="caption mt-3"><b>Table 3 |</b> Share of studies tagged with each concept from 2022 onwards, relative to 2014 to 2018.</p>
            </div>
            <div>
              <SectionHead n={3}>Where it is published</SectionHead>
              <ol className="ruled mt-3 border-b border-rule">
                {trends.top_journals.slice(0, 15).map(([j, c], i) => (
                  <li key={j} className="grid grid-cols-[2rem_1fr_auto] items-baseline py-2.5">
                    <span className="num">{i + 1}.</span><i className="font-serif text-[16px]">{j}</i><span className="num">{c}</span>
                  </li>
                ))}
              </ol>
            </div>
          </section>
        </>
      )}
    </main>
  )
}
