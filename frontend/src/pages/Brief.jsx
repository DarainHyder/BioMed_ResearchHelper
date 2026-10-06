import React, { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../lib/api'
import { loadMap } from '../lib/mapData'
import { Dot, PageHead, Spinner, setDomainColors, usePaperDrawer } from '../components/ui'

const QUESTIONS = [
  'Does the gut microbiome influence response to cancer immunotherapy?',
  'What are the main safety risks of CAR-T cell therapy?',
  'How accurate is deep learning for detecting diabetic retinopathy?',
  'Do GLP-1 receptor agonists reduce cardiovascular events?',
  'What drives antimicrobial resistance in hospital settings?',
  'Can epigenetic clocks predict mortality?',
]

const Sup = ({ n, onClick }) => (
  <sup><button onClick={onClick} className="ml-0.5 font-sans text-[11px] font-semibold text-hema hover:underline">{n}</button></sup>
)

export default function Brief() {
  const [params, setParams] = useSearchParams()
  const q = params.get('q') || ''
  const [input, setInput] = useState(q)
  const [brief, setBrief] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [openPaper, drawer] = usePaperDrawer()

  useEffect(() => { loadMap().then((m) => setDomainColors(m.domains, m.colors)) }, [])
  useEffect(() => {
    setInput(q)
    if (!q) { setBrief(null); return }
    setLoading(true); setError(null)
    api.brief({ question: q, k: 8 }).then((b) => { setBrief(b); setLoading(false) }).catch((e) => { setError(e.message); setLoading(false) })
  }, [q])

  const ask = (text) => text.trim() && setParams({ q: text.trim() })
  const pm = (n) => brief?.papers?.[n - 1]?.pmid

  return (
    <main className="page min-h-screen">
      <PageHead kicker="Evidence briefs" title="Put a question to the literature.">
        The leading studies are retrieved, and the sentences that best answer you are drawn from their abstracts, each with
        its reference. Nothing is paraphrased unless an AI synthesis is enabled, and then it is marked as such.
      </PageHead>

      <form className="mt-10" onSubmit={(e) => { e.preventDefault(); ask(input) }}>
        <div className="flex items-end gap-4">
          <input value={input} onChange={(e) => setInput(e.target.value)} maxLength={500} className="field" placeholder="What are the main mechanisms of resistance to PD-1 blockade?" aria-label="Question" />
          <button className="btn-ink mb-1 shrink-0" disabled={loading}>{loading ? <Spinner /> : 'Prepare brief'}</button>
        </div>
      </form>

      {!q && (
        <section className="mt-10 max-w-3xl">
          <p className="font-sans text-[13px] font-semibold">Questions others have asked</p>
          <ol className="ruled mt-2 border-b border-rule">
            {QUESTIONS.map((x, i) => (
              <li key={x}><button onClick={() => ask(x)} className="grid w-full grid-cols-[2rem_1fr] py-3 text-left font-serif text-xl text-ink-2 hover:text-hema"><span className="num pt-1.5">{i + 1}.</span>{x}</button></li>
            ))}
          </ol>
        </section>
      )}

      {error && <p className="body mt-10 text-eosin">{error}. The server may be waking up; please try again in a moment.</p>}
      {loading && <div className="mt-12 max-w-3xl space-y-3">{[60, 100, 96, 100, 88, 100, 70].map((w, i) => <div key={i} className="skeleton h-4" style={{ width: `${w}%` }} />)}</div>}

      {brief && !loading && (
        <article className="mt-14 grid gap-12 lg:grid-cols-[minmax(0,1fr)_280px]">
          <div className="max-w-[720px]">
            <p className="kicker">Research brief</p>
            <h2 className="mt-2 font-serif text-[34px] font-normal leading-tight">{brief.question}</h2>
            <p className="meta mt-3">Drawn from {brief.papers.length} studies{brief.evidence.year_span ? `, ${brief.evidence.year_span[0]} to ${brief.evidence.year_span[1]}` : ''}. Prepared in {(brief.took_ms / 1000).toFixed(1)} s.</p>

            {brief.synthesis && (
              <section className="mt-8 border-l-2 border-hema pl-5">
                <p className="font-sans text-[13px] font-semibold">Synthesis <span className="font-normal text-ink-3">(AI-written, grounded in the cited sentences)</span></p>
                <p className="body mt-2">{brief.synthesis}</p>
              </section>
            )}

            <h3 className="mt-10 font-sans text-[13px] font-semibold">What the evidence says</h3>
            {brief.summary.length ? (
              <div className="mt-2 space-y-4 font-serif text-[19px] leading-[1.75] text-ink-2">
                {brief.summary.map((s, i) => <p key={i} className="animate-ink" style={{ animationDelay: `${i * 60}ms` }}>{s.text}<Sup n={s.cite} onClick={() => openPaper(pm(s.cite))} /></p>)}
              </div>
            ) : <p className="body mt-2">No sufficiently relevant studies were found.</p>}

            {brief.key_findings?.length > 0 && (
              <>
                <h3 className="mt-10 font-sans text-[13px] font-semibold">Quantitative findings</h3>
                <ol className="ruled mt-2 border-y border-rule">
                  {brief.key_findings.map((f, i) => (
                    <li key={i} className="grid grid-cols-[2rem_1fr] py-3"><span className="num pt-1">{i + 1}.</span>
                      <p className="font-serif text-[16.5px] leading-relaxed text-ink-2">{f.text}<Sup n={f.cite} onClick={() => openPaper(pm(f.cite))} /></p></li>
                  ))}
                </ol>
              </>
            )}

            <h3 className="mt-12 font-sans text-[13px] font-semibold">References</h3>
            <ol className="mt-2 space-y-3">
              {brief.papers.map((p, i) => (
                <li key={p.pmid} className="grid grid-cols-[2rem_1fr] font-serif text-[15.5px] leading-relaxed text-ink-2">
                  <span className="num pt-0.5">{i + 1}.</span>
                  <span>{p.authors?.join(', ')}{p.n_authors > (p.authors || []).length ? ', et al' : ''}. <button onClick={() => openPaper(p.pmid)} className="text-left text-ink hover:text-hema hover:underline">{p.title}</button> <i>{p.journal}</i>. {p.year}. <span className="num">PMID {p.pmid}</span></span>
                </li>
              ))}
            </ol>
            <p className="caption mt-8">Method: query-focused extractive summarisation (maximal marginal relevance) over the top {brief.papers.length} studies.
              Read the original studies before drawing clinical conclusions.</p>
          </div>

          <aside className="space-y-8 lg:sticky lg:top-24 lg:self-start">
            <div>
              <p className="border-b border-ink pb-2 font-sans text-[13px] font-semibold">Evidence profile</p>
              <dl className="ruled font-sans text-[14px]">
                <div className="flex justify-between py-2.5"><dt className="text-ink-3">Studies</dt><dd className="tabular-nums">{brief.papers.length}</dd></div>
                {brief.evidence.year_span && <div className="flex justify-between py-2.5"><dt className="text-ink-3">Published</dt><dd className="tabular-nums">{brief.evidence.year_span[0]} to {brief.evidence.year_span[1]}</dd></div>}
                <div className="flex justify-between py-2.5"><dt className="text-ink-3">Mean relevance</dt><dd className="tabular-nums">{Math.round(brief.evidence.mean_relevance * 100)}%</dd></div>
                {brief.evidence.study_designs.map(([d, c]) => <div key={d} className="flex justify-between py-2.5"><dt className="text-ink-3">{d}</dt><dd className="tabular-nums">{c}</dd></div>)}
              </dl>
            </div>
            {brief.evidence.domains?.length > 0 && (
              <div>
                <p className="border-b border-ink pb-2 font-sans text-[13px] font-semibold">Fields</p>
                <ul className="mt-2 space-y-1.5">{brief.evidence.domains.map(([d, c]) => <li key={d} className="flex items-center gap-2 font-serif text-[15px]"><Dot domain={d} />{d}<span className="num ml-auto">{c}</span></li>)}</ul>
              </div>
            )}
            {brief.evidence.concepts.length > 0 && (
              <div>
                <p className="border-b border-ink pb-2 font-sans text-[13px] font-semibold">Recurring concepts</p>
                <p className="mt-2 font-serif text-[15px] leading-relaxed text-ink-2">{brief.evidence.concepts.map(([m]) => m).join('; ')}.</p>
              </div>
            )}
          </aside>
        </article>
      )}
      {drawer}
    </main>
  )
}
