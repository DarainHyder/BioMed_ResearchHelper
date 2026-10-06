import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ArrowRight, FileText, Search as SearchIcon, SlidersHorizontal, X } from 'lucide-react'
import { api } from '../lib/api'
import { loadMap } from '../lib/mapData'
import { Dot, PageHead, ResultCard, Spinner, setDomainColors, usePaperDrawer } from '../components/ui'

const MODES = [['hybrid', 'Hybrid'], ['semantic', 'Semantic'], ['keyword', 'Keyword']]
const SUGGESTED = ['CRISPR off-target effects', 'long-term outcomes of GLP-1 therapy', 'antibiotic resistance in ICU',
  'epigenetic clocks and mortality', 'organoid models of the brain', 'AI triage chest radiographs']

function YearHistogram({ years, from, to, onPick }) {
  const keys = Object.keys(years || {}).map(Number).sort((a, b) => a - b)
  if (!keys.length) return null
  const max = Math.max(...Object.values(years))
  return (
    <div>
      <div className="flex h-20 items-end gap-[3px]">
        {keys.map((y) => {
          const on = (!from || y >= from) && (!to || y <= to)
          return (
            <button key={y} title={`${y}: ${years[y]} papers`} onClick={() => onPick(y)}
              className={`flex-1 rounded-t-[3px] transition-colors ${on ? 'bg-ink/75 hover:bg-signal' : 'bg-ink/15 hover:bg-ink/30'}`}
              style={{ height: `${Math.max(6, (years[y] / max) * 100)}%` }} />
          )
        })}
      </div>
      <div className="mt-1.5 flex justify-between font-mono text-[10px] text-ink-4"><span>{keys[0]}</span><span>{keys[keys.length - 1]}</span></div>
    </div>
  )
}

export default function Search() {
  const [params, setParams] = useSearchParams()
  const q = params.get('q') || ''
  const mode = params.get('mode') || 'hybrid'
  const domains = params.getAll('domain')
  const yearFrom = Number(params.get('from')) || undefined
  const yearTo = Number(params.get('to')) || undefined
  const sort = params.get('sort') || 'relevance'

  const [input, setInput] = useState(q)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [allDomains, setAllDomains] = useState([])
  const [showFilters, setShowFilters] = useState(false)
  const [openPaper, drawer] = usePaperDrawer()
  const abortRef = useRef(null)

  useEffect(() => { loadMap().then((m) => { setDomainColors(m.domains, m.colors); setAllDomains(m.domains) }) }, [])
  useEffect(() => setInput(q), [q])

  useEffect(() => {
    if (!q) { setData(null); return }
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setLoading(true); setError(null)
    api.search({ q, k: 20, mode, domain: domains, year_from: yearFrom, year_to: yearTo, sort }, ctrl.signal)
      .then((d) => { setData(d); setLoading(false) })
      .catch((e) => { if (e.name !== 'AbortError') { setError(e.message); setLoading(false) } })
    return () => ctrl.abort()
  }, [q, mode, domains.join('|'), yearFrom, yearTo, sort]) // eslint-disable-line react-hooks/exhaustive-deps

  const update = (patch) => {
    const next = new URLSearchParams(params)
    Object.entries(patch).forEach(([k, v]) => {
      next.delete(k)
      if (Array.isArray(v)) v.forEach((x) => next.append(k, x))
      else if (v !== undefined && v !== null && v !== '') next.set(k, v)
    })
    setParams(next)
  }
  const toggleDomain = (d) => update({ domain: domains.includes(d) ? domains.filter((x) => x !== d) : [...domains, d] })
  const pickYear = (y) => {
    if (yearFrom === y && yearTo === y) update({ from: '', to: '' })
    else update({ from: y, to: y })
  }
  const activeFilters = domains.length + (yearFrom ? 1 : 0)
  const facetDomains = useMemo(() => data?.facets?.domains || [], [data])

  return (
    <main className="min-h-screen">
      <div className="container-x pb-24">
        <PageHead eyebrow="Search" title={<>Search by <span className="serif-i">meaning.</span></>}>
          Hybrid retrieval fuses a fine-tuned biomedical encoder with BM25 keyword matching, so you find studies that use different words for the same idea.
        </PageHead>

        <form onSubmit={(e) => { e.preventDefault(); input.trim() && update({ q: input.trim() }) }}
          className="mt-10 flex items-center gap-2 rounded-full border border-ink/15 bg-bone-50 p-2 pl-6 shadow-[0_24px_50px_-40px_rgba(20,22,19,.6)]">
          <SearchIcon className="h-5 w-5 flex-shrink-0 text-ink-4" />
          <input autoFocus value={input} onChange={(e) => setInput(e.target.value)} placeholder="e.g. resistance mechanisms to PD-1 blockade"
            className="min-w-0 flex-1 bg-transparent py-2.5 text-[16px] outline-none placeholder:text-ink-4" />
          {input && <button type="button" onClick={() => setInput('')} className="p-2 text-ink-4 hover:text-ink" aria-label="Clear"><X className="h-4 w-4" /></button>}
          <button className="btn-ink">Search</button>
        </form>

        <div className="mt-5 flex flex-wrap items-center gap-2">
          <div className="flex rounded-full border border-ink/10 bg-bone-50 p-1">
            {MODES.map(([m, label]) => (
              <button key={m} onClick={() => update({ mode: m === 'hybrid' ? '' : m })}
                className={`rounded-full px-3.5 py-1.5 text-xs font-medium transition ${mode === m ? 'bg-ink text-bone-50' : 'text-ink-3 hover:text-ink'}`}>{label}</button>
            ))}
          </div>
          <button onClick={() => setShowFilters((s) => !s)} className={`chip ${showFilters || activeFilters ? 'chip-on' : ''}`}>
            <SlidersHorizontal className="h-3.5 w-3.5" /> Fields{activeFilters ? ` (${activeFilters})` : ''}
          </button>
          <select value={sort} onChange={(e) => update({ sort: e.target.value === 'relevance' ? '' : e.target.value })} className="chip cursor-pointer outline-none">
            <option value="relevance">Most relevant</option>
            <option value="newest">Newest first</option>
          </select>
          {activeFilters > 0 && <button className="text-xs text-ink-3 underline underline-offset-4" onClick={() => update({ domain: [], from: '', to: '' })}>Clear filters</button>}
          {q && <Link to={`/brief?q=${encodeURIComponent(q)}`} className="btn-signal ml-auto py-2 text-xs"><FileText className="h-3.5 w-3.5" /> Brief this question</Link>}
        </div>

        {showFilters && (
          <div className="mt-4 flex flex-wrap gap-1.5 rounded-3xl border border-ink/10 bg-bone-50 p-4 animate-rise">
            {allDomains.map((d) => (
              <button key={d} onClick={() => toggleDomain(d)} className={`chip ${domains.includes(d) ? 'chip-on' : 'hover:border-ink/30'}`}>
                <Dot domain={d} /> {d}
              </button>
            ))}
          </div>
        )}

        {!q && (
          <div className="mt-16">
            <p className="eyebrow">Try</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {SUGGESTED.map((s) => (
                <button key={s} onClick={() => update({ q: s })} className="surface surface-hover flex items-center justify-between p-5 text-left">
                  <span className="text-ink-2">{s}</span><ArrowRight className="h-4 w-4 text-ink-4" />
                </button>
              ))}
            </div>
          </div>
        )}

        {q && (
          <div className="mt-12 grid gap-10 lg:grid-cols-[minmax(0,1fr)_280px]">
            <div>
              <div className="mb-5 flex items-center justify-between text-sm text-ink-3">
                {loading ? <span className="flex items-center gap-2"><Spinner /> Searching {fmtMode(mode)}…</span>
                  : data ? <span>{data.total} relevant studies · <span className="font-mono text-xs">{data.took_ms} ms</span></span> : null}
              </div>
              {error && <div className="surface p-6 text-signal-ink">{error}. The server may be waking up; try again in a few seconds.</div>}
              {loading && !data && <div className="space-y-3">{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton h-36" />)}</div>}
              <div className={`space-y-3 transition-opacity ${loading ? 'opacity-50' : ''}`}>
                {data?.results.map((r, i) => <ResultCard key={r.pmid} r={r} terms={data.terms} onOpen={openPaper} index={i + 1} />)}
                {data && !data.results.length && <div className="surface p-10 text-center text-ink-3">No studies matched. Try fewer filters or the Semantic mode.</div>}
              </div>
            </div>

            {data && data.results.length > 0 && (
              <aside className="min-w-0 space-y-8 lg:sticky lg:top-28 lg:self-start">
                <div>
                  <p className="eyebrow">Where results cluster</p>
                  <div className="mt-4 space-y-2.5">
                    {facetDomains.slice(0, 8).map(([d, c]) => (
                      <button key={d} onClick={() => toggleDomain(d)} className="group block w-full text-left">
                        <div className="flex items-center justify-between text-sm">
                          <span className="flex min-w-0 items-center gap-2 text-ink-2 group-hover:text-ink"><Dot domain={d} /><span className="truncate">{d}</span></span>
                          <span className="font-mono text-xs text-ink-4">{c}</span>
                        </div>
                        <div className="mt-1.5 h-1 rounded-full bg-ink/[0.06]">
                          <div className="h-1 rounded-full bg-ink/70" style={{ width: `${(c / facetDomains[0][1]) * 100}%` }} />
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="eyebrow">By publication year</p>
                  <div className="mt-4"><YearHistogram years={data.facets.years} from={yearFrom} to={yearTo} onPick={pickYear} /></div>
                </div>
                {data.facets.topics?.length > 0 && (
                  <div>
                    <p className="eyebrow">Research fronts</p>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {data.facets.topics.map((t) => <Link key={t.id} to={`/topics?open=${t.id}`} className="chip hover:border-ink/30">{t.label.split(' · ').slice(0, 2).join(' · ')}</Link>)}
                    </div>
                  </div>
                )}
              </aside>
            )}
          </div>
        )}
      </div>
      {drawer}
    </main>
  )
}

const fmtMode = (m) => ({ hybrid: 'by meaning and keywords', semantic: 'by meaning', keyword: 'by keywords' }[m])
