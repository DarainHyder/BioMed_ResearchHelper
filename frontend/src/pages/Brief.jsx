import React, { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ArrowRight, Sparkles } from 'lucide-react'
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

const Cite = ({ n, onClick }) => (
  <button onClick={onClick} className="mx-0.5 inline-flex h-[18px] min-w-[18px] -translate-y-[2px] items-center justify-center rounded-md bg-ink px-1 align-middle font-sans text-[10px] font-medium not-italic text-bone-50 transition hover:bg-signal">
    {n}
  </button>
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
    api.brief({ question: q, k: 8 })
      .then((b) => { setBrief(b); setLoading(false) })
      .catch((e) => { setError(e.message); setLoading(false) })
  }, [q])

  const ask = (text) => text.trim() && setParams({ q: text.trim() })
  const paperAt = (n) => brief?.papers?.[n - 1]?.pmid

  return (
    <main className="min-h-screen">
      <div className="container-x pb-28">
        <PageHead eyebrow="Evidence briefs" title={<>Ask the literature <span className="serif-i">a question.</span></>}>
          BioAtlas retrieves the most relevant studies, then selects the sentences that best answer you, each one cited, with the
          quantitative findings and an evidence profile alongside.
        </PageHead>

        <form onSubmit={(e) => { e.preventDefault(); ask(input) }} className="mt-10">
          <div className="rounded-[26px] border border-ink/15 bg-bone-50 p-2 shadow-[0_24px_50px_-40px_rgba(20,22,19,.6)] focus-within:border-ink/35">
            <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={2} maxLength={500}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(input) } }}
              placeholder="e.g. What are the main mechanisms of resistance to PD-1 blockade?"
              className="w-full resize-none bg-transparent px-4 pt-3 font-serif text-xl outline-none placeholder:text-ink-4" />
            <div className="flex items-center justify-between px-2 pb-1">
              <span className="text-xs text-ink-4">Enter to ask · cites up to 8 studies</span>
              <button className="btn-ink" disabled={loading}>{loading ? <Spinner /> : <>Generate brief <ArrowRight className="h-4 w-4" /></>}</button>
            </div>
          </div>
        </form>

        {!q && (
          <div className="mt-14 grid gap-3 md:grid-cols-2">
            {QUESTIONS.map((x) => (
              <button key={x} onClick={() => ask(x)} className="surface surface-hover p-6 text-left font-serif text-lg leading-snug text-ink-2">{x}</button>
            ))}
          </div>
        )}

        {error && <div className="surface mt-10 p-6 text-signal-ink">{error}. The server may be waking up; please retry in a moment.</div>}

        {loading && (
          <div className="mt-14 grid gap-10 lg:grid-cols-[1.5fr_1fr]">
            <div className="space-y-4">{[100, 92, 97, 85, 70].map((w, i) => <div key={i} className="skeleton h-6" style={{ width: `${w}%` }} />)}</div>
            <div className="skeleton h-72" />
          </div>
        )}

        {brief && !loading && (
          <div className="mt-14 grid gap-12 lg:grid-cols-[1.5fr_1fr]">
            <article>
              <p className="eyebrow">Question</p>
              <h2 className="mt-3 font-serif text-3xl leading-tight text-ink sm:text-[34px]">{brief.question}</h2>

              {brief.synthesis && (
                <div className="mt-10 rounded-[22px] bg-moss p-7 text-bone">
                  <p className="eyebrow flex items-center gap-2 text-bone/50"><Sparkles className="h-3.5 w-3.5" /> AI synthesis, grounded in the cited sentences</p>
                  <p className="mt-4 font-serif text-lg leading-8 text-bone/90">{brief.synthesis}</p>
                </div>
              )}

              <p className="eyebrow mt-10">What the evidence says</p>
              {brief.summary.length ? (
                <div className="mt-4 space-y-5 font-serif text-[19px] leading-[1.75] text-ink-2">
                  {brief.summary.map((s, i) => (
                    <p key={i} className="animate-rise" style={{ animationDelay: `${i * 70}ms` }}>
                      {s.text}<Cite n={s.cite} onClick={() => openPaper(paperAt(s.cite))} />
                    </p>
                  ))}
                </div>
              ) : <p className="mt-4 text-ink-3">No relevant studies were found for this question.</p>}

              {brief.key_findings?.length > 0 && (
                <>
                  <p className="eyebrow mt-12">Quantitative findings</p>
                  <div className="mt-4 grid gap-3 sm:grid-cols-2">
                    {brief.key_findings.map((f, i) => (
                      <div key={i} className="surface p-5">
                        <p className="text-[15px] leading-7 text-ink-2">{f.text}</p>
                        <button onClick={() => openPaper(paperAt(f.cite))} className="mt-3 font-mono text-xs text-ink-4 hover:text-signal">Source [{f.cite}]</button>
                      </div>
                    ))}
                  </div>
                </>
              )}
              <p className="mt-10 text-xs text-ink-4">
                Method: {brief.method === 'extractive-mmr' ? 'query-focused extractive summarisation with maximal marginal relevance' : 'extractive selection plus grounded LLM synthesis'} · {brief.took_ms} ms.
                Summaries quote the source abstracts; always read the original studies before drawing conclusions.
              </p>
            </article>

            <aside className="space-y-6 lg:sticky lg:top-28 lg:self-start">
              <div className="surface p-6">
                <p className="eyebrow">Evidence profile</p>
                <dl className="mt-5 space-y-5 text-sm">
                  {brief.evidence.year_span && (
                    <div className="flex justify-between"><dt className="text-ink-3">Publication span</dt><dd className="font-mono">{brief.evidence.year_span[0]} to {brief.evidence.year_span[1]}</dd></div>
                  )}
                  <div className="flex justify-between"><dt className="text-ink-3">Mean relevance</dt><dd className="font-mono">{Math.round(brief.evidence.mean_relevance * 100)}%</dd></div>
                  {brief.evidence.study_designs.length > 0 && (
                    <div>
                      <dt className="text-ink-3">Study designs</dt>
                      <dd className="mt-2 flex flex-wrap gap-1.5">{brief.evidence.study_designs.map(([d, c]) => <span key={d} className="chip">{d} · {c}</span>)}</dd>
                    </div>
                  )}
                  {brief.evidence.concepts.length > 0 && (
                    <div>
                      <dt className="text-ink-3">Recurring concepts</dt>
                      <dd className="mt-2 flex flex-wrap gap-1.5">{brief.evidence.concepts.map(([m]) => <span key={m} className="chip">{m}</span>)}</dd>
                    </div>
                  )}
                </dl>
              </div>
              <div>
                <p className="eyebrow mb-3">Sources</p>
                <ol className="divide-y divide-ink/10 overflow-hidden rounded-[22px] border border-ink/10 bg-bone-50">
                  {brief.papers.map((p, i) => (
                    <li key={p.pmid}>
                      <button onClick={() => openPaper(p.pmid)} className="flex w-full gap-3 px-4 py-3.5 text-left hover:bg-bone-100">
                        <span className="mt-0.5 font-mono text-xs text-ink-4">[{i + 1}]</span>
                        <span className="flex-1">
                          <span className="line-clamp-2 text-sm leading-6 text-ink-2">{p.title}</span>
                          <span className="mt-1 flex items-center gap-1.5 text-xs text-ink-4"><Dot domain={p.domain} className="h-1.5 w-1.5" /> {p.journal} · {p.year}</span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ol>
              </div>
            </aside>
          </div>
        )}
      </div>
      {drawer}
    </main>
  )
}
