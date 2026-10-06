import React, { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../lib/api'
import { loadMap } from '../lib/mapData'
import { Dot, PageHead, Reference, Spinner, setDomainColors, usePaperDrawer } from '../components/ui'

const MODES = [['hybrid', 'Meaning and keywords'], ['semantic', 'Meaning only'], ['keyword', 'Keywords only']]
const SUGGESTED = ['CRISPR off-target effects', 'long-term outcomes of GLP-1 therapy', 'antibiotic resistance in intensive care',
  'epigenetic clocks and mortality', 'organoid models of the human brain', 'AI triage of chest radiographs']

function Years({ years, from, to, onPick }) {
  const keys = Object.keys(years || {}).map(Number).sort((a, b) => a - b)
  if (!keys.length) return null
  const max = Math.max(...Object.values(years))
  return (
    <div>
      <div className="flex h-16 items-end gap-[2px] border-b border-ink">
        {keys.map((y) => {
          const on = (!from || y >= from) && (!to || y <= to)
          return <button key={y} title={`${y}: ${years[y]}`} onClick={() => onPick(y)} style={{ height: `${Math.max(6, (years[y] / max) * 100)}%` }}
            className={`flex-1 transition-colors ${on ? 'bg-ink hover:bg-eosin' : 'bg-rule hover:bg-ink-4'}`} />
        })}
      </div>
      <div className="mt-1 flex justify-between num"><span>{keys[0]}</span><span>{keys[keys.length - 1]}</span></div>
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
  const [openPaper, drawer] = usePaperDrawer()
  const abortRef = useRef(null)

  useEffect(() => { loadMap().then((m) => setDomainColors(m.domains, m.colors)) }, [])
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
  const pickYear = (y) => (yearFrom === y && yearTo === y ? update({ from: '', to: '' }) : update({ from: y, to: y }))
  const filtered = domains.length || yearFrom

  return (
    <main className="page min-h-screen">
      <PageHead kicker="Search" title="Find studies by what they mean.">
        Your words are matched two ways at once: by a language model trained on these abstracts, and by the exact terms. Studies
        that describe the same idea in different language are found together.
      </PageHead>

      <form className="mt-10" onSubmit={(e) => { e.preventDefault(); input.trim() && update({ q: input.trim() }) }}>
        <div className="flex items-end gap-4">
          <input autoFocus value={input} onChange={(e) => setInput(e.target.value)} className="field" placeholder="mechanisms of resistance to PD-1 blockade" aria-label="Search query" />
          <button className="btn-ink mb-1 shrink-0">Search</button>
        </div>
      </form>

      <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-rule">
        {MODES.map(([m, label]) => (
          <button key={m} onClick={() => update({ mode: m === 'hybrid' ? '' : m })} className={`tab ${mode === m ? 'tab-on' : ''}`}>{label}</button>
        ))}
        <span className="ml-auto flex items-center gap-4 pb-2">
          <select value={sort} onChange={(e) => update({ sort: e.target.value === 'relevance' ? '' : e.target.value })} className="cursor-pointer bg-transparent font-sans text-sm text-ink-3 outline-none">
            <option value="relevance">Most relevant first</option>
            <option value="newest">Newest first</option>
          </select>
          {q && <Link to={`/brief?q=${encodeURIComponent(q)}`} className="font-sans text-sm text-hema hover:underline">Write a brief on this &rarr;</Link>}
        </span>
      </div>

      {!q && (
        <section className="mt-10 max-w-2xl">
          <p className="font-sans text-[13px] font-semibold">Some places to start</p>
          <ul className="ruled mt-2 border-b border-rule">
            {SUGGESTED.map((s) => (
              <li key={s}><button onClick={() => update({ q: s })} className="w-full py-3 text-left font-serif text-xl text-ink-2 hover:text-hema">{s}</button></li>
            ))}
          </ul>
        </section>
      )}

      {q && (
        <div className="mt-8 grid gap-12 lg:grid-cols-[minmax(0,1fr)_260px]">
          <section>
            <p className="meta">
              {loading ? <span className="inline-flex items-center gap-2"><Spinner className="h-3 w-3" /> Searching…</span>
                : data ? <>{data.total} relevant studies for <i className="font-serif text-[15px] text-ink">“{data.query}”</i> <span className="num">· {data.took_ms} ms</span></> : null}
              {filtered ? <button className="ml-3 link" onClick={() => update({ domain: [], from: '', to: '' })}>clear filters</button> : null}
            </p>
            {error && <p className="body mt-6 text-eosin">{error}. The server may be waking up; please try again in a few seconds.</p>}
            {loading && !data && <div className="mt-6 space-y-6">{[0, 1, 2].map((i) => <div key={i} className="space-y-2"><div className="skeleton h-5 w-4/5" /><div className="skeleton h-3 w-2/5" /><div className="skeleton h-4 w-full" /></div>)}</div>}
            <ol className={`ruled mt-3 border-y border-rule transition-opacity ${loading ? 'opacity-40' : ''}`}>
              {data?.results.map((r, i) => <li key={r.pmid}><Reference r={r} n={i + 1} terms={data.terms} onOpen={openPaper} /></li>)}
            </ol>
            {data && !data.results.length && <p className="body mt-6">No studies matched. Try fewer filters, or search by meaning only.</p>}
          </section>

          {data?.results.length > 0 && (
            <aside className="min-w-0 space-y-10 lg:sticky lg:top-24 lg:self-start">
              <div>
                <p className="font-sans text-[13px] font-semibold">Fields</p>
                <ul className="mt-2">
                  {data.facets.domains.slice(0, 10).map(([d, c]) => (
                    <li key={d}>
                      <button onClick={() => toggleDomain(d)} className={`flex w-full items-baseline gap-2 py-1.5 text-left ${domains.includes(d) ? 'text-ink' : 'text-ink-2 hover:text-ink'}`}>
                        <Dot domain={d} className="h-2 w-2 translate-y-[1px]" />
                        <span className={`flex-1 truncate font-serif text-[15.5px] ${domains.includes(d) ? 'underline decoration-eosin decoration-2 underline-offset-4' : ''}`}>{d}</span>
                        <span className="num">{c}</span>
                      </button>
                    </li>
                  ))}
                </ul>
                {domains.filter((d) => !data.facets.domains.some(([x]) => x === d)).map((d) => (
                  <button key={d} onClick={() => toggleDomain(d)} className="meta mt-1 block link">remove {d}</button>
                ))}
              </div>
              <div>
                <p className="mb-3 font-sans text-[13px] font-semibold">Year of publication</p>
                <Years years={data.facets.years} from={yearFrom} to={yearTo} onPick={pickYear} />
              </div>
              {data.facets.topics?.length > 0 && (
                <div>
                  <p className="font-sans text-[13px] font-semibold">Research fronts</p>
                  <ul className="mt-2 space-y-1.5">
                    {data.facets.topics.map((t) => <li key={t.id}><Link to={`/topics?open=${t.id}`} className="font-serif text-[15.5px] text-ink-2 hover:text-hema">{t.label}</Link></li>)}
                  </ul>
                </div>
              )}
            </aside>
          )}
        </div>
      )}
      {drawer}
    </main>
  )
}
