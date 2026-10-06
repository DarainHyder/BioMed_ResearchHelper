import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { X } from 'lucide-react'
import { api } from '../lib/api'

export const DOMAIN_COLORS = {}
export const setDomainColors = (domains, colors) => domains.forEach((d, i) => { DOMAIN_COLORS[d] = colors[i] })

export const Dot = ({ domain, className = 'h-2 w-2' }) => (
  <span className={`inline-block flex-shrink-0 rounded-full ${className}`} style={{ background: DOMAIN_COLORS[domain] || '#A39C92' }} />
)

export function Highlight({ text, terms }) {
  const parts = useMemo(() => {
    if (!terms?.length || !text) return [text]
    const re = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
    return text.split(re)
  }, [text, terms])
  if (parts.length === 1) return text
  const set = new Set(terms.map((t) => t.toLowerCase()))
  return parts.map((p, i) => (set.has(p?.toLowerCase()) ? <mark key={i} className="hit">{p}</mark> : <React.Fragment key={i}>{p}</React.Fragment>))
}

export function Sparkline({ data, years, className = 'h-8 w-full', color = '#1B1916' }) {
  const ys = years || Object.keys(data).sort()
  const vals = ys.map((y) => data[y] || 0)
  const max = Math.max(1, ...vals)
  const d = vals.map((v, i) => `${i ? 'L' : 'M'}${((i / Math.max(1, vals.length - 1)) * 100).toFixed(2)},${(29 - (v / max) * 26).toFixed(2)}`).join(' ')
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" className={className}>
      <path d={d} fill="none" stroke={color} strokeWidth="1.3" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  )
}

export const Spinner = ({ className = 'h-4 w-4' }) => (
  <span className={`inline-block animate-spin rounded-full border-[1.5px] border-current border-r-transparent ${className}`} />
)

export function PageHead({ kicker, title, children }) {
  return (
    <header className="border-b border-ink pb-8 pt-28 sm:pt-32">
      <p className="kicker animate-ink">{kicker}</p>
      <h1 className="h1 mt-3 max-w-4xl animate-ink" style={{ animationDelay: '50ms' }}>{title}</h1>
      {children && <div className="lede mt-5 max-w-3xl animate-ink" style={{ animationDelay: '100ms' }}>{children}</div>}
    </header>
  )
}

export function SectionHead({ n, children, aside }) {
  return (
    <div className="flex items-end justify-between gap-6 border-b border-rule pb-3">
      <h2 className="h2"><span className="sec mr-3 align-middle">§ {n}</span>{children}</h2>
      {aside && <div className="meta pb-1">{aside}</div>}
    </div>
  )
}

const authorsShort = (r) => `${(r.authors || []).join(', ')}${r.n_authors > (r.authors || []).length ? ', et al.' : ''}`

// A search result set as a bibliography entry.
export function Reference({ r, n, terms, onOpen }) {
  return (
    <article className="group grid grid-cols-[2.5rem_1fr] gap-x-2 py-5 sm:grid-cols-[3rem_1fr_7rem]">
      <span className="num pt-1">{n}.</span>
      <div className="min-w-0">
        <button onClick={() => onOpen(r.pmid)} className="text-left font-serif text-[19px] leading-snug text-ink decoration-hema/40 underline-offset-4 group-hover:underline">
          <Highlight text={r.title} terms={terms} />
        </button>
        <p className="meta mt-1.5 truncate">{authorsShort(r)} <i className="font-serif text-[14px]">{r.journal}</i>. {r.year}. PMID {r.pmid}</p>
        <p className="mt-2 font-serif text-[15.5px] leading-relaxed text-ink-3"><Highlight text={r.snippet} terms={terms} /></p>
        <p className="tag mt-2 flex items-center gap-1.5"><Dot domain={r.domain} className="h-1.5 w-1.5" />{r.domain}{r.pub_types?.length ? `; ${r.pub_types.join('; ')}` : ''}</p>
      </div>
      {r.similarity !== null && r.similarity !== undefined && (
        <div className="col-start-2 mt-2 sm:col-start-3 sm:mt-1.5 sm:text-right">
          <span className="num">{Math.round(r.similarity * 100)}% match</span>
          <div className="mt-1 h-px w-24 bg-rule sm:ml-auto"><div className="h-px bg-ink" style={{ width: `${Math.max(0, (r.similarity - 0.3) / 0.6) * 100}%` }} /></div>
        </div>
      )}
    </article>
  )
}

const LABELS = /(?=\b(?:Background|Introduction|Objectives?|Aims?|Purpose|Methods?|Design|Setting|Results?|Findings|Conclusions?|Interpretation|Significance): )/

export function PaperDrawer({ pmid, onClose, onOpen }) {
  const [paper, setPaper] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!pmid) return
    setPaper(null); setError(null)
    api.paper(pmid).then(setPaper).catch((e) => setError(e.message))
  }, [pmid])

  useEffect(() => {
    if (!pmid) return
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    document.documentElement.style.overflow = 'hidden'
    window.__lenis?.stop()
    return () => { window.removeEventListener('keydown', onKey); document.documentElement.style.overflow = ''; window.__lenis?.start() }
  }, [pmid, onClose])

  if (!pmid) return null
  return (
    <div className="fixed inset-0 z-[60]">
      <div className="absolute inset-0 bg-ink/25" onClick={onClose} />
      <aside data-lenis-prevent className="absolute right-0 top-0 h-full w-full max-w-[720px] overflow-y-auto border-l border-ink bg-paper animate-ink">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-rule bg-paper px-8 py-3">
          <span className="num">PMID {pmid}</span>
          <button onClick={onClose} className="p-1 text-ink-3 hover:text-ink" aria-label="Close"><X className="h-5 w-5" /></button>
        </div>
        {error && <p className="body p-8 text-eosin">{error}. The server may be waking up; try again in a moment.</p>}
        {!paper && !error && <div className="space-y-3 p-8">{[70, 95, 85, 100, 100, 60].map((w, i) => <div key={i} className="skeleton h-4" style={{ width: `${w}%` }} />)}</div>}
        {paper && (
          <article className="px-8 pb-20 pt-8">
            <p className="font-sans text-[12px] uppercase tracking-[0.12em] text-ink-3">{paper.journal} · {paper.year}</p>
            <h2 className="mt-3 font-serif text-[30px] font-semibold leading-tight tracking-[-0.01em]">{paper.title}</h2>
            <p className="meta mt-4 leading-6">{paper.authors?.join(', ')}{paper.n_authors > (paper.authors || []).length ? `, and ${paper.n_authors - paper.authors.length} more` : ''}</p>
            <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 border-y border-rule py-3 font-sans text-[13px]">
              <a className="link" href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`} target="_blank" rel="noreferrer">PubMed record</a>
              {paper.doi && <a className="link" href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer">Full text via DOI</a>}
              <span className="flex items-center gap-1.5 text-ink-3"><Dot domain={paper.domain} />{paper.domain}</span>
              <span className="text-ink-3">Front: {paper.topic_label}</span>
            </div>
            <h3 className="mt-8 font-sans text-[13px] font-semibold">Abstract</h3>
            <div className="mt-2 space-y-3 font-serif text-[17.5px] leading-[1.75] text-ink-2">
              {paper.abstract.split(LABELS).map((para, i) => {
                const m = para.match(/^(\w[\w ]*?): (.*)$/s)
                return <p key={i}>{m ? <><b className="font-semibold text-ink">{m[1]}.</b> {m[2]}</> : para}</p>
              })}
            </div>
            {paper.mesh?.length > 0 && (
              <p className="mt-8 font-serif text-[15px] leading-relaxed text-ink-3">
                <span className="font-sans text-[13px] font-semibold text-ink">MeSH terms: </span>{paper.mesh.slice(0, 18).join('; ')}.
              </p>
            )}
            {paper.similar?.length > 0 && (
              <>
                <h3 className="mt-10 font-sans text-[13px] font-semibold">Related studies <span className="font-normal text-ink-3">(nearest in meaning)</span></h3>
                <ol className="ruled mt-2 border-y border-rule">
                  {paper.similar.map((s, i) => (
                    <li key={s.pmid}>
                      <button onClick={() => onOpen(s.pmid)} className="grid w-full grid-cols-[1.8rem_1fr_auto] gap-2 py-3 text-left hover:text-hema">
                        <span className="num pt-0.5">{i + 1}.</span>
                        <span className="font-serif text-[15.5px] leading-snug">{s.title}</span>
                        <span className="num pt-0.5">{s.year}</span>
                      </button>
                    </li>
                  ))}
                </ol>
              </>
            )}
          </article>
        )}
      </aside>
    </div>
  )
}

export function usePaperDrawer() {
  const [pmid, setPmid] = useState(null)
  const close = useCallback(() => setPmid(null), [])
  return [setPmid, <PaperDrawer key="drawer" pmid={pmid} onClose={close} onOpen={setPmid} />]
}
