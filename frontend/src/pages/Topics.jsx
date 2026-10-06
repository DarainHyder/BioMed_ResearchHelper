import React, { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ArrowUpRight, TrendingUp, X } from 'lucide-react'
import { api } from '../lib/api'
import { loadMap } from '../lib/mapData'
import { useReveal } from '../lib/hooks'
import { DOMAIN_COLORS, Dot, PageHead, Sparkline, setDomainColors, usePaperDrawer } from '../components/ui'

const YEARS = Array.from({ length: 12 }, (_, i) => String(2014 + i))

function growthLabel(g) {
  if (g >= 1.6) return { text: `${g.toFixed(1)}× growth`, tone: 'text-signal' }
  if (g >= 1.15) return { text: `${g.toFixed(1)}× growth`, tone: 'text-ink-2' }
  if (g <= 0.75) return { text: 'Cooling', tone: 'text-ink-4' }
  return { text: 'Steady', tone: 'text-ink-3' }
}

function TopicPanel({ id, onClose, openPaper }) {
  const [t, setT] = useState(null)
  useEffect(() => { setT(null); api.topic(id).then(setT).catch(() => onClose()) }, [id, onClose])
  useEffect(() => { window.__lenis?.stop(); return () => window.__lenis?.start() }, [])
  return (
    <div className="fixed inset-0 z-[55]">
      <div className="absolute inset-0 bg-ink/30 backdrop-blur-[2px]" onClick={onClose} />
      <aside data-lenis-prevent className="absolute right-0 top-0 h-full w-full max-w-[640px] overflow-y-auto bg-bone-50 shadow-2xl animate-rise">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-ink/10 bg-bone-50/90 px-7 py-4 backdrop-blur">
          <span className="eyebrow">Research front #{id}</span>
          <button onClick={onClose} className="rounded-full p-2 hover:bg-ink/5" aria-label="Close"><X className="h-5 w-5" /></button>
        </div>
        {!t ? <div className="space-y-4 p-8">{[70, 100, 90, 100].map((w, i) => <div key={i} className="skeleton h-6" style={{ width: `${w}%` }} />)}</div> : (
          <div className="px-7 pb-16 pt-6">
            <span className="chip"><Dot domain={t.domain} /> {t.domain}</span>
            <h2 className="mt-4 font-display text-3xl font-medium leading-tight tracking-tight">{t.label}</h2>
            <p className="mt-2 text-sm text-ink-3">{t.size} studies · {growthLabel(t.growth).text}</p>
            <div className="mt-6 rounded-2xl border border-ink/10 p-4">
              <Sparkline data={t.yearly} years={YEARS} className="h-24 w-full" color={DOMAIN_COLORS[t.domain]} />
              <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-4"><span>2014</span><span>2025</span></div>
            </div>
            <p className="eyebrow mt-8">Defining vocabulary</p>
            <div className="mt-3 flex flex-wrap gap-1.5">{t.keywords.map(([w]) => <span key={w} className="chip">{w}</span>)}</div>
            {t.domain_mix?.length > 1 && (
              <>
                <p className="eyebrow mt-8">Fields contributing</p>
                <div className="mt-3 space-y-2">
                  {t.domain_mix.map(([d, c]) => (
                    <div key={d} className="flex items-center gap-3 text-sm">
                      <Dot domain={d} /><span className="flex-1 text-ink-2">{d}</span><span className="font-mono text-xs text-ink-4">{c}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
            <p className="eyebrow mt-8">Most representative studies</p>
            <div className="mt-3 divide-y divide-ink/10 overflow-hidden rounded-2xl border border-ink/10">
              {t.papers.map((p) => (
                <button key={p.pmid} onClick={() => openPaper(p.pmid)} className="flex w-full gap-3 bg-bone-50 px-4 py-3 text-left hover:bg-bone-100">
                  <span className="flex-1 text-sm leading-6 text-ink-2">{p.title}</span>
                  <span className="font-mono text-[11px] text-ink-4">{p.year}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </aside>
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
  const openId = params.get('open')
  const ref = useReveal([topics, sort, domain])

  useEffect(() => {
    loadMap().then((m) => { setDomainColors(m.domains, m.colors); setDomains(m.domains) })
    api.topics().then((r) => setTopics(r.topics)).catch(() => setTopics([]))
  }, [])

  const shown = useMemo(() => (topics || [])
    .filter((t) => !domain || t.domain === domain)
    .sort((a, b) => (sort === 'growth' ? b.growth - a.growth : b.size - a.size)), [topics, sort, domain])
  const close = React.useCallback(() => setParams({}), [setParams])

  return (
    <main className="min-h-screen">
      <div ref={ref} className="container-x pb-28">
        <PageHead eyebrow="Topic discovery" title={<>Research fronts, <span className="serif-i">found not labelled.</span></>}>
          Clusters emerge from the geometry of the embeddings (UMAP and HDBSCAN), are named by their own vocabulary
          (class TF-IDF), and ranked by how fast they grew between 2016 to 2019 and 2023 to 2025.
        </PageHead>

        <div className="mt-10 flex flex-wrap items-center gap-2">
          <div className="flex rounded-full border border-ink/10 bg-bone-50 p-1">
            {[['growth', 'Fastest growing'], ['size', 'Largest']].map(([k, l]) => (
              <button key={k} onClick={() => setSort(k)} className={`rounded-full px-3.5 py-1.5 text-xs font-medium ${sort === k ? 'bg-ink text-bone-50' : 'text-ink-3'}`}>{l}</button>
            ))}
          </div>
          <select value={domain} onChange={(e) => setDomain(e.target.value)} className="chip cursor-pointer py-2 outline-none">
            <option value="">All fields</option>
            {domains.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          {topics && <span className="ml-auto text-sm text-ink-4">{shown.length} fronts</span>}
        </div>

        {!topics && <div className="mt-8 grid gap-4 md:grid-cols-2 lg:grid-cols-3">{Array.from({ length: 6 }, (_, i) => <div key={i} className="skeleton h-64" />)}</div>}
        <div className="mt-8 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {shown.map((t, i) => {
            const g = growthLabel(t.growth)
            return (
              <button key={t.id} onClick={() => setParams({ open: t.id })}
                className="reveal surface surface-hover group flex flex-col p-6 text-left" style={{ transitionDelay: `${(i % 6) * 50}ms` }}>
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-2 text-ink-3"><Dot domain={t.domain} /> {t.domain}</span>
                  <span className={`flex items-center gap-1 font-mono ${g.tone}`}>{t.growth >= 1.15 && <TrendingUp className="h-3.5 w-3.5" />}{g.text}</span>
                </div>
                <h3 className="mt-4 font-display text-xl font-medium leading-snug tracking-tight">{t.label}</h3>
                <div className="mt-auto pt-6">
                  <Sparkline data={t.yearly} years={YEARS} className="h-14 w-full" color={DOMAIN_COLORS[t.domain]} />
                  <div className="mt-4 flex items-center justify-between">
                    <div className="flex flex-wrap gap-1">{t.keywords.slice(3, 6).map(([w]) => <span key={w} className="rounded-full bg-ink/[0.05] px-2 py-0.5 text-[11px] text-ink-3">{w}</span>)}</div>
                    <span className="flex items-center gap-1 text-xs text-ink-4">{t.size}<ArrowUpRight className="h-3.5 w-3.5 transition group-hover:text-signal" /></span>
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </div>
      {openId !== null && <TopicPanel id={Number(openId)} onClose={close} openPaper={openPaper} />}
      {drawer}
    </main>
  )
}
