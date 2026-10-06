import React, { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../lib/api'
import { loadMap } from '../lib/mapData'
import { DOMAIN_COLORS, Dot, PageHead, Sparkline, Spinner, setDomainColors, usePaperDrawer } from '../components/ui'

const YEARS = Array.from({ length: 12 }, (_, i) => String(2014 + i))
const growthText = (g) => (g >= 1.15 ? `${g.toFixed(1)}×` : g <= 0.8 ? 'cooling' : 'steady')

function Detail({ id, openPaper }) {
  const [t, setT] = useState(null)
  useEffect(() => { api.topic(id).then(setT).catch(() => setT({ error: true })) }, [id])
  if (!t) return <div className="flex items-center gap-2 py-6 meta"><Spinner className="h-3 w-3" /> Loading the front…</div>
  if (t.error) return <p className="meta py-6">Could not load this front; the server may be waking.</p>
  return (
    <div className="grid gap-8 border-t border-rule bg-paper-2/60 px-4 py-6 md:grid-cols-[1fr_1.4fr]">
      <div>
        <p className="font-sans text-[13px] font-semibold">Defining vocabulary</p>
        <p className="mt-1 font-serif text-[16px] italic leading-relaxed text-ink-2">{t.keywords.map(([w]) => w).join(', ')}</p>
        {t.domain_mix?.length > 1 && (
          <>
            <p className="mt-5 font-sans text-[13px] font-semibold">Contributing fields</p>
            <ul className="mt-1 space-y-1">{t.domain_mix.map(([d, c]) => <li key={d} className="flex items-center gap-2 font-serif text-[15px]"><Dot domain={d} />{d}<span className="num ml-auto">{c}</span></li>)}</ul>
          </>
        )}
        {t.top_mesh?.length > 0 && <p className="mt-5 font-serif text-[14.5px] leading-relaxed text-ink-3"><span className="font-sans text-[13px] font-semibold text-ink">MeSH: </span>{t.top_mesh.slice(0, 8).map(([m]) => m).join('; ')}.</p>}
      </div>
      <div>
        <p className="font-sans text-[13px] font-semibold">Most representative studies</p>
        <ol className="mt-1 space-y-2">
          {t.papers.map((p, i) => (
            <li key={p.pmid} className="grid grid-cols-[1.6rem_1fr_auto] gap-2">
              <span className="num pt-0.5">{i + 1}.</span>
              <button onClick={() => openPaper(p.pmid)} className="text-left font-serif text-[15.5px] leading-snug hover:text-hema">{p.title}</button>
              <span className="num pt-0.5">{p.year}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

export default function Topics() {
  const [params, setParams] = useSearchParams()
  const [topics, setTopics] = useState(null)
  const [domains, setDomains] = useState([])
  const [sort, setSort] = useState('growth')
  const [domain, setDomain] = useState('')
  const [openPaper, drawer] = usePaperDrawer()
  const open = params.get('open') !== null ? Number(params.get('open')) : null

  useEffect(() => {
    loadMap().then((m) => { setDomainColors(m.domains, m.colors); setDomains(m.domains) })
    api.topics().then((r) => setTopics(r.topics)).catch(() => setTopics([]))
  }, [])

  const rows = useMemo(() => (topics || []).filter((t) => !domain || t.domain === domain)
    .sort((a, b) => (sort === 'growth' ? b.growth - a.growth : sort === 'size' ? b.size - a.size : a.label.localeCompare(b.label))), [topics, sort, domain])
  const Th = ({ k, children, className = '' }) => (
    <th className={`py-2.5 pr-4 font-semibold ${className}`}>
      <button onClick={() => setSort(k)} className={sort === k ? 'underline decoration-eosin decoration-2 underline-offset-4' : 'hover:text-hema'}>{children}</button>
    </th>
  )

  return (
    <main className="page min-h-screen">
      <PageHead kicker="Research fronts" title="Where the literature clusters, and how fast it grows.">
        No one labelled these. Studies that sit close together in meaning were grouped by density clustering, each group was named
        from its own vocabulary, and its growth measured between 2016 to 2019 and 2023 to 2025.
      </PageHead>

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4">
        <select value={domain} onChange={(e) => setDomain(e.target.value)} className="cursor-pointer border-b border-ink bg-transparent pb-1 font-serif text-[17px] outline-none">
          <option value="">All fields</option>
          {domains.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        {topics && <span className="meta">{rows.length} fronts. Click a column heading to sort, a row to open it.</span>}
      </div>

      {!topics ? <div className="mt-8 space-y-3">{Array.from({ length: 8 }, (_, i) => <div key={i} className="skeleton h-10" />)}</div> : (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse font-sans text-[14px]">
            <thead>
              <tr className="border-y border-ink text-left">
                <th className="w-10 py-2.5 pr-2 font-semibold">#</th>
                <Th k="label">Research front</Th>
                <th className="py-2.5 pr-4 font-semibold">Field</th>
                <Th k="size" className="text-right">Studies</Th>
                <th className="w-40 py-2.5 pr-4 font-semibold">2014 to 2025</th>
                <Th k="growth" className="text-right">Growth</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((t, i) => (
                <React.Fragment key={t.id}>
                  <tr onClick={() => setParams(open === t.id ? {} : { open: t.id })} className={`cursor-pointer border-b border-rule align-middle transition-colors hover:bg-paper-2 ${open === t.id ? 'bg-paper-2' : ''}`}>
                    <td className="py-3 pr-2 num">{i + 1}</td>
                    <td className="py-3 pr-4 font-serif text-[16.5px] text-ink">{t.label}</td>
                    <td className="py-3 pr-4 text-ink-3"><span className="flex items-center gap-2"><Dot domain={t.domain} />{t.domain}</span></td>
                    <td className="py-3 pr-4 text-right tabular-nums">{t.size}</td>
                    <td className="py-3 pr-4"><Sparkline data={t.yearly} years={YEARS} className="h-7 w-36" color={DOMAIN_COLORS[t.domain]} /></td>
                    <td className={`py-3 text-right tabular-nums ${t.growth >= 2 ? 'font-semibold text-eosin' : ''}`}>{growthText(t.growth)}</td>
                  </tr>
                  {open === t.id && <tr><td colSpan={6} className="p-0"><Detail id={t.id} openPaper={openPaper} /></td></tr>}
                </React.Fragment>
              ))}
            </tbody>
          </table>
          <p className="caption mt-3"><b>Table 2 | Research fronts.</b> Growth compares the mean annual number of studies in 2023 to 2025 with
            2016 to 2019, within a corpus sampled evenly by year, so it reflects a front's rising share of its field.</p>
        </div>
      )}
      {drawer}
    </main>
  )
}
