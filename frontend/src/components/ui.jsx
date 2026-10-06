import React, { useEffect, useMemo, useState } from 'react'
import { ArrowUpRight, X } from 'lucide-react'
import { api } from '../lib/api'

export const DOMAIN_COLORS = {}
export const setDomainColors = (domains, colors) => domains.forEach((d, i) => { DOMAIN_COLORS[d] = colors[i] })

export const Dot = ({ domain, className = 'h-2 w-2' }) => (
  <span className={`inline-block flex-shrink-0 rounded-full ${className}`} style={{ background: DOMAIN_COLORS[domain] || '#8A8B80' }} />
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

export function Sparkline({ data, years, className = 'h-10 w-full', color = '#141613', fill = true }) {
  const ys = years || Object.keys(data).sort()
  const vals = ys.map((y) => data[y] || 0)
  const max = Math.max(1, ...vals)
  const pts = vals.map((v, i) => [(i / Math.max(1, vals.length - 1)) * 100, 30 - (v / max) * 26 - 2])
  const d = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ')
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" className={className}>
      {fill && <path d={`${d} L100,30 L0,30 Z`} fill={color} opacity="0.08" />}
      <path d={d} fill="none" stroke={color} strokeWidth="1.4" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  )
}

export function Spinner({ className = 'h-4 w-4' }) {
  return <span className={`inline-block animate-spin rounded-full border-2 border-current border-r-transparent ${className}`} />
}

export function ResultCard({ r, terms, onOpen, index }) {
  return (
    <button onClick={() => onOpen(r.pmid)} className="surface surface-hover group block w-full p-6 text-left">
      <div className="flex items-center gap-2 text-xs text-ink-4">
        {index !== undefined && <span className="font-mono text-ink-5">{String(index).padStart(2, '0')}</span>}
        <Dot domain={r.domain} />
        <span>{r.domain}</span>
        <span className="text-ink-5">·</span>
        <span>{r.year}</span>
        {r.pub_types?.map((t) => <span key={t} className="rounded-full bg-ink/[0.05] px-2 py-0.5 text-[11px] text-ink-3">{t}</span>)}
        {r.similarity !== null && r.similarity !== undefined && (
          <span className="ml-auto font-mono text-[11px] text-ink-4">{(r.similarity * 100).toFixed(0)}% match</span>
        )}
      </div>
      <h3 className="mt-3 font-display text-lg font-medium leading-snug text-ink transition-colors group-hover:text-signal-ink">
        <Highlight text={r.title} terms={terms} />
      </h3>
      <p className="mt-2 line-clamp-2 text-sm leading-6 text-ink-3"><Highlight text={r.snippet} terms={terms} /></p>
      <p className="mt-3 truncate text-xs text-ink-4">
        {r.authors?.join(', ')}{r.n_authors > 3 ? ' et al.' : ''} · <span className="italic">{r.journal}</span>
      </p>
    </button>
  )
}

export function PaperDrawer({ pmid, onClose, onOpen }) {
  const [paper, setPaper] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!pmid) return
    setPaper(null); setError(null)
    api.paper(pmid).then(setPaper).catch((e) => setError(e.message))
  }, [pmid])

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    document.documentElement.style.overflow = pmid ? 'hidden' : ''
    window.__lenis?.[pmid ? 'stop' : 'start']()
    return () => { window.removeEventListener('keydown', onKey); document.documentElement.style.overflow = ''; window.__lenis?.start() }
  }, [pmid, onClose])

  if (!pmid) return null
  return (
    <div className="fixed inset-0 z-[60]">
      <div className="absolute inset-0 bg-ink/30 backdrop-blur-[2px]" onClick={onClose} />
      <aside data-lenis-prevent className="absolute right-0 top-0 h-full w-full max-w-[680px] overflow-y-auto bg-bone-50 shadow-2xl animate-rise">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-ink/10 bg-bone-50/90 px-7 py-4 backdrop-blur">
          <span className="font-mono text-xs text-ink-4">PMID {pmid}</span>
          <button onClick={onClose} className="rounded-full p-2 text-ink-3 hover:bg-ink/5" aria-label="Close"><X className="h-5 w-5" /></button>
        </div>
        {error && <p className="p-8 text-signal-ink">{error}</p>}
        {!paper && !error && (
          <div className="space-y-4 p-8">{[80, 60, 100, 100, 90].map((w, i) => <div key={i} className="skeleton h-5" style={{ width: `${w}%` }} />)}</div>
        )}
        {paper && (
          <div className="px-7 pb-16 pt-6">
            <div className="flex flex-wrap items-center gap-2 text-xs text-ink-3">
              <span className="chip"><Dot domain={paper.domain} /> {paper.domain}</span>
              <span className="chip">{paper.year}</span>
              <span className="chip">Topic: {paper.topic_label}</span>
            </div>
            <h2 className="mt-5 font-display text-[28px] font-medium leading-tight tracking-tight">{paper.title}</h2>
            <p className="mt-4 text-sm text-ink-3">{paper.authors?.join(', ')}{paper.n_authors > paper.authors?.length ? ` +${paper.n_authors - paper.authors.length}` : ''}</p>
            <p className="mt-1 text-sm italic text-ink-4">{paper.journal}</p>
            <div className="mt-5 flex flex-wrap gap-2">
              <a className="btn-ink py-2" href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`} target="_blank" rel="noreferrer">PubMed <ArrowUpRight className="h-4 w-4" /></a>
              {paper.doi && <a className="btn-line py-2" href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer">Full text (DOI) <ArrowUpRight className="h-4 w-4" /></a>}
            </div>
            <p className="eyebrow mt-10">Abstract</p>
            <div className="mt-3 space-y-3 font-serif text-[17px] leading-8 text-ink-2">
              {paper.abstract.split(/(?=\b(?:Background|Introduction|Objectives?|Aims?|Purpose|Methods?|Design|Results?|Findings|Conclusions?|Interpretation|Significance): )/).map((para, i) => <p key={i}>{para}</p>)}
            </div>
            {paper.mesh?.length > 0 && (
              <>
                <p className="eyebrow mt-10">MeSH concepts</p>
                <div className="mt-3 flex flex-wrap gap-1.5">{paper.mesh.slice(0, 18).map((m) => <span key={m} className="chip">{m}</span>)}</div>
              </>
            )}
            {paper.similar?.length > 0 && (
              <>
                <p className="eyebrow mt-10">Nearest neighbours in the embedding space</p>
                <div className="mt-3 divide-y divide-ink/10 overflow-hidden rounded-2xl border border-ink/10">
                  {paper.similar.map((s) => (
                    <button key={s.pmid} onClick={() => onOpen(s.pmid)} className="flex w-full items-start gap-3 bg-bone-50 px-4 py-3 text-left hover:bg-bone-100">
                      <Dot domain={s.domain} className="mt-2 h-2 w-2" />
                      <span className="flex-1 text-sm leading-6 text-ink-2">{s.title}</span>
                      <span className="font-mono text-[11px] text-ink-4">{s.year}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </aside>
    </div>
  )
}

export function usePaperDrawer() {
  const [pmid, setPmid] = useState(null)
  const close = React.useCallback(() => setPmid(null), [])
  const drawer = <PaperDrawer pmid={pmid} onClose={close} onOpen={setPmid} />
  return [setPmid, drawer]
}

export function PageHead({ eyebrow, title, children }) {
  return (
    <div className="pt-36 sm:pt-40">
      <p className="eyebrow animate-rise">{eyebrow}</p>
      <h1 className="h-display mt-4 max-w-4xl text-5xl leading-[0.98] animate-rise sm:text-6xl md:text-7xl" style={{ animationDelay: '60ms' }}>{title}</h1>
      {children && <div className="mt-6 max-w-2xl text-lg text-ink-3 animate-rise" style={{ animationDelay: '120ms' }}>{children}</div>}
    </div>
  )
}
