import React, { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowRight, ArrowUpRight, Search as SearchIcon } from 'lucide-react'
import ExplodeHero from '../components/ExplodeHero'
import { setDomainColors } from '../components/ui'
import { useReveal } from '../lib/hooks'
import { fmt, loadMap } from '../lib/mapData'

const MODEL_ROWS = [
  ['bm25', 'BM25 keyword search', 'Lexical baseline'],
  ['v1_all_mpnet_base_v2', 'all-mpnet-base-v2', 'The original v1 model (110M params)'],
  ['bge_small_base', 'bge-small, off the shelf', '33M params'],
  ['bge_small_finetuned', 'bge-small, fine-tuned here', 'Contrastive training on this corpus'],
  ['hybrid_bm25_finetuned', 'Hybrid: BM25 + fine-tuned', 'Reciprocal rank fusion (production)'],
]

const EXAMPLES = ['CAR-T cytokine release syndrome management', 'gut microbiome and depression',
  'GLP-1 agonists cardiovascular outcomes', 'deep learning for diabetic retinopathy']

function Benchmark({ retrieval }) {
  const ref = useReveal([retrieval])
  if (!retrieval?.bge_small_finetuned) return null
  const best = Math.max(...MODEL_ROWS.map(([k]) => retrieval[k]?.['recall@1'] || 0))
  const v1 = retrieval.v1_all_mpnet_base_v2?.['recall@1']
  const ours = retrieval.hybrid_bm25_finetuned?.['recall@1']
  return (
    <section ref={ref} className="container-x py-28 md:py-36">
      <div className="grid gap-14 lg:grid-cols-[1fr_1.25fr] lg:gap-20">
        <div className="reveal">
          <p className="eyebrow">Benchmark · held-out papers</p>
          <h2 className="h-display mt-4 text-4xl leading-[1.02] sm:text-5xl">
            Trained on the corpus, <span className="serif-i">not just prompted.</span>
          </h2>
          <p className="mt-6 text-lg leading-8 text-ink-3">
            We fine-tuned a compact biomedical embedding model on title, abstract and MeSH pairs, then measured how often each
            system ranks the right paper first among all 14,693 abstracts, given only its title. None of the test papers were
            seen in training.
          </p>
          {v1 && ours && (
            <p className="mt-8 font-display text-6xl font-medium tracking-tightest text-signal">
              +{Math.round((ours / v1 - 1) * 100)}%
              <span className="ml-3 align-middle font-sans text-base font-normal tracking-normal text-ink-3">top-1 accuracy vs. the v1 model</span>
            </p>
          )}
        </div>
        <div className="reveal surface p-2 sm:p-3" style={{ transitionDelay: '120ms' }}>
          {MODEL_ROWS.map(([key, name, note]) => {
            const r = retrieval[key]
            if (!r) return null
            const prod = key === 'hybrid_bm25_finetuned'
            return (
              <div key={key} className={`rounded-2xl px-5 py-4 ${prod ? 'bg-moss text-bone' : ''}`}>
                <div className="flex items-baseline justify-between gap-4">
                  <div>
                    <p className={`font-medium ${prod ? '' : 'text-ink'}`}>{name}</p>
                    <p className={`text-xs ${prod ? 'text-bone/50' : 'text-ink-4'}`}>{note}</p>
                  </div>
                  <div className="text-right font-mono text-sm">
                    <span className={prod ? 'text-bone' : 'text-ink'}>{(r['recall@1'] * 100).toFixed(1)}%</span>
                    <span className={`ml-3 text-xs ${prod ? 'text-bone/50' : 'text-ink-4'}`}>MRR {r['mrr@10'].toFixed(3)}</span>
                  </div>
                </div>
                <div className={`mt-3 h-1.5 overflow-hidden rounded-full ${prod ? 'bg-white/10' : 'bg-ink/[0.06]'}`}>
                  <div className={`h-full rounded-full transition-[width] duration-[1600ms] ease-out ${prod ? 'bg-signal' : 'bg-ink/70'}`}
                    style={{ width: `${(r['recall@1'] / best) * 100}%` }} />
                </div>
              </div>
            )
          })}
          <p className="px-5 pb-3 pt-2 text-xs text-ink-4">Recall@1: share of 1,514 held-out titles whose own paper is the very first result among all 14,693 studies. MRR@10 shown alongside.</p>
        </div>
      </div>
    </section>
  )
}

function Capabilities() {
  const ref = useReveal()
  const cards = [
    { to: '/search', k: '01', title: 'Hybrid search', body: 'Meaning and keywords, fused. Filter by field and year, see where results cluster.',
      visual: (
        <div className="space-y-2.5">
          <div className="flex items-center gap-2 rounded-full border border-ink/10 bg-bone px-3 py-2 text-xs text-ink-3"><SearchIcon className="h-3.5 w-3.5" /> checkpoint inhibitor resistance</div>
          {[92, 88, 84].map((s, i) => <div key={s} className="flex items-center gap-2"><div className="h-2 flex-1 rounded-full bg-ink/[0.07]"><div className="h-2 rounded-full bg-ink/70" style={{ width: `${s - i * 8}%` }} /></div><span className="font-mono text-[10px] text-ink-4">{s}%</span></div>)}
        </div>) },
    { to: '/brief', k: '02', title: 'Evidence briefs', body: 'Ask a question, get a cited synthesis with key quantitative findings and an evidence profile.',
      visual: (
        <p className="font-serif text-[15px] leading-7 text-ink-2">
          Response rates improved with combination therapy <sup className="rounded bg-signal px-1 font-sans text-[9px] text-white">2</sup>, while
          resistance was linked to antigen loss <sup className="rounded bg-ink px-1 font-sans text-[9px] text-white">4</sup>.
        </p>) },
    { to: '/topics', k: '03', title: 'Topic discovery', body: 'Research fronts found without labels, named by their own vocabulary, ranked by growth.',
      visual: (
        <div className="flex flex-wrap gap-1.5">{['organoid', 'tumor microenvironment', 'spatial transcriptomics', 'base editing', 'microbiota'].map((w, i) => <span key={w} className={`chip ${i === 1 ? 'chip-on' : ''}`}>{w}</span>)}</div>) },
    { to: '/trends', k: '04', title: 'Real trends', body: 'True PubMed publication volumes per field since 2014, plus the concepts rising fastest.',
      visual: (
        <div className="flex h-16 items-end gap-1">{[18, 22, 25, 31, 36, 44, 52, 61, 66, 74, 83, 92].map((h, i) => <div key={i} className={`flex-1 rounded-sm ${i > 8 ? 'bg-signal' : 'bg-ink/15'}`} style={{ height: `${h}%` }} />)}</div>) },
  ]
  return (
    <section ref={ref} className="container-x pb-28 md:pb-36">
      <div className="reveal flex flex-col justify-between gap-6 md:flex-row md:items-end">
        <h2 className="h-display max-w-2xl text-4xl leading-[1.02] sm:text-5xl">Four instruments, <span className="serif-i">one corpus.</span></h2>
        <p className="max-w-sm text-ink-3">Every view reads from the same fine-tuned representation of the literature, so search, briefs, topics and trends always agree.</p>
      </div>
      <div className="mt-14 grid gap-4 md:grid-cols-2">
        {cards.map((c, i) => (
          <Link key={c.k} to={c.to} className="reveal surface surface-hover group flex min-h-[300px] flex-col justify-between p-8" style={{ transitionDelay: `${i * 90}ms` }}>
            <div className="flex items-start justify-between">
              <span className="font-mono text-xs text-ink-4">{c.k}</span>
              <ArrowUpRight className="h-5 w-5 text-ink-4 transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-signal" />
            </div>
            <div className="my-8 max-w-sm">{c.visual}</div>
            <div>
              <h3 className="font-display text-2xl font-medium tracking-tight">{c.title}</h3>
              <p className="mt-2 max-w-md text-ink-3">{c.body}</p>
            </div>
          </Link>
        ))}
      </div>
    </section>
  )
}

function Domains({ map }) {
  if (!map) return null
  const row = map.domains.map((d, i) => ({ d, c: map.colors[i] }))
  const Track = ({ items, reverse }) => (
    <div className="flex overflow-hidden">
      <div className="flex shrink-0 animate-marquee gap-10 pr-10" style={{ animationDirection: reverse ? 'reverse' : 'normal' }}>
        {[...items, ...items].map((x, i) => (
          <span key={i} className="flex items-center gap-3 whitespace-nowrap font-serif text-3xl italic text-ink-2 sm:text-4xl">
            <span className="h-3 w-3 rounded-full" style={{ background: x.c }} />{x.d}
          </span>
        ))}
      </div>
    </div>
  )
  return (
    <section className="border-y border-ink/10 bg-bone-50 py-14">
      <div className="space-y-8 [mask-image:linear-gradient(90deg,transparent,#000_10%,#000_90%,transparent)]">
        <Track items={row.slice(0, 12)} />
        <Track items={row.slice(12)} reverse />
      </div>
    </section>
  )
}

function HowItWorks() {
  const ref = useReveal()
  const steps = [
    ['Ingest', 'Year-stratified sampling of PubMed: 24 fields, 2014 to today, so trends are real rather than a recency artefact.'],
    ['Adapt', 'A 33M-parameter encoder is fine-tuned with in-batch contrastive learning on title, abstract and MeSH pairs.'],
    ['Index', 'Dense vectors plus a sparse BM25 index, fused at query time with reciprocal rank fusion.'],
    ['Discover', 'UMAP and HDBSCAN find research fronts; class TF-IDF names them; growth is measured over time.'],
    ['Serve', 'An int8 ONNX encoder answers queries in milliseconds on a small CPU, no GPU required.'],
  ]
  return (
    <section ref={ref} className="relative overflow-hidden bg-moss py-28 text-bone grain md:py-36" data-nav-dark>
      <div className="container-x relative">
        <p className="reveal eyebrow text-bone/40">Under the hood</p>
        <h2 className="reveal h-display mt-4 max-w-3xl text-4xl leading-[1.02] sm:text-5xl">
          From raw abstracts to a <span className="serif-i text-signal-soft">searchable map</span> in five steps.
        </h2>
        <div className="mt-16 grid gap-px overflow-hidden rounded-[22px] border border-white/10 bg-white/10 md:grid-cols-5">
          {steps.map(([t, b], i) => (
            <div key={t} className="reveal bg-moss p-7" style={{ transitionDelay: `${i * 80}ms` }}>
              <p className="font-mono text-xs text-signal-soft">0{i + 1}</p>
              <p className="mt-10 font-display text-xl font-medium">{t}</p>
              <p className="mt-3 text-sm leading-6 text-bone/55">{b}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function SearchCTA() {
  const ref = useReveal()
  const navigate = useNavigate()
  const [q, setQ] = useState('')
  const go = (query) => query.trim() && navigate(`/search?q=${encodeURIComponent(query.trim())}`)
  return (
    <section ref={ref} className="container-x py-28 text-center md:py-36">
      <h2 className="reveal h-display mx-auto max-w-3xl text-5xl leading-[0.98] sm:text-6xl">
        What does the evidence <span className="serif-i">say?</span>
      </h2>
      <form className="reveal mx-auto mt-10 flex max-w-2xl items-center gap-2 rounded-full border border-ink/15 bg-bone-50 p-2 pl-6 shadow-[0_30px_60px_-40px_rgba(20,22,19,.5)]"
        onSubmit={(e) => { e.preventDefault(); go(q) }}>
        <SearchIcon className="h-5 w-5 text-ink-4" />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search 14,000+ studies by meaning"
          className="flex-1 bg-transparent py-2 text-[15px] outline-none placeholder:text-ink-4" />
        <button className="btn-ink">Search <ArrowRight className="h-4 w-4" /></button>
      </form>
      <div className="reveal mt-6 flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((e) => <button key={e} onClick={() => go(e)} className="chip hover:border-ink/30 hover:text-ink">{e}</button>)}
      </div>
    </section>
  )
}

export default function Home() {
  const [map, setMap] = useState(null)
  const introRef = useReveal([map])
  useEffect(() => {
    loadMap().then((m) => { setDomainColors(m.domains, m.colors); setMap(m) })
  }, [])

  return (
    <main>
      <ExplodeHero map={map} />
      <section ref={introRef} className="container-x py-28 md:py-40">
        <p className="reveal eyebrow">Why BioAtlas</p>
        <p className="reveal mt-6 max-w-5xl font-display text-3xl font-medium leading-[1.18] tracking-tight text-ink sm:text-[44px]">
          More than a million biomedical papers are published every year. BioAtlas reads them the way a researcher would:
          <span className="serif-i text-ink-3"> by meaning, not keywords,</span> then shows you where a question sits in the wider landscape of science.
        </p>
        {map && (
          <div className="reveal mt-16 grid grid-cols-2 gap-px overflow-hidden rounded-[22px] border border-ink/10 bg-ink/10 md:grid-cols-4">
            {[[fmt(map.stats.papers), 'studies indexed'], [map.domains.length, 'fields of medicine'], [map.stats.topics, 'research fronts'],
              [fmt(map.stats.journals), 'journals']].map(([v, l]) => (
              <div key={l} className="bg-bone px-6 py-8">
                <p className="h-display text-4xl sm:text-5xl">{v}</p>
                <p className="mt-2 text-sm text-ink-3">{l}</p>
              </div>
            ))}
          </div>
        )}
      </section>
      <Capabilities />
      <Domains map={map} />
      <Benchmark retrieval={map?.retrieval} />
      <HowItWorks />
      <SearchCTA />
    </main>
  )
}
