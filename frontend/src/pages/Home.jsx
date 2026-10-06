import React, { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import HelixHero from '../components/HelixHero'
import { SectionHead, setDomainColors, usePaperDrawer } from '../components/ui'
import { useReveal } from '../lib/hooks'
import { fmt, loadMap } from '../lib/mapData'

const MODELS = [
  ['bm25', 'BM25 keyword retrieval', 'n/a'],
  ['v1_all_mpnet_base_v2', 'all-mpnet-base-v2 (the original v1 model)', '110M'],
  ['bge_small_base', 'bge-small, off the shelf', '33M'],
  ['bge_small_finetuned', 'bge-small, fine-tuned on this corpus', '33M'],
  ['hybrid_bm25_finetuned', 'Hybrid: BM25 and fine-tuned, rank-fused', '33M'],
]

const INSTRUMENTS = [
  ['/search', 'Search', 'Find studies by meaning and by keyword at once, filtered by field and year, with the matching passage quoted.'],
  ['/brief', 'Evidence briefs', 'Pose a clinical or scientific question. The most relevant sentences across the leading studies are assembled and cited.'],
  ['/atlas', 'The Atlas', 'The full plate: every study in its place. Light up any question and see where the evidence lives.'],
  ['/topics', 'Research fronts', 'Clusters that emerged on their own, named by their vocabulary and ranked by how quickly they grew.'],
  ['/trends', 'Trends', 'Publication volume in each field from 2014 to 2025, counted directly from PubMed, with the concepts on the rise.'],
]

function Abstract({ map, ev }) {
  const f = ev?.hybrid_bm25_finetuned
  const v1 = ev?.v1_all_mpnet_base_v2
  return (
    <section className="border-y border-ink py-10">
      <div className="grid gap-8 lg:grid-cols-[200px_1fr]">
        <div>
          <p className="font-sans text-[13px] font-semibold">Abstract</p>
          <p className="meta mt-2 leading-6">BioAtlas, version 2<br />Data from NCBI PubMed</p>
        </div>
        <div className="body max-w-3xl space-y-4 text-[18px]">
          <p><b className="font-semibold text-ink">Background.</b> More than a million biomedical papers appear each year. Keyword search finds the
            words, not the ideas, and says nothing about how a question relates to the rest of science.</p>
          <p><b className="font-semibold text-ink">Methods.</b> We sampled {fmt(map?.stats.papers)} abstracts evenly across {map?.domains.length} fields and every year
            from 2014, fine-tuned a 33M-parameter language model on title, abstract and MeSH pairs, and combined it with keyword
            retrieval. Topics were found by density clustering of the learned representation.</p>
          <p><b className="font-semibold text-ink">Results.</b> {f && v1 ? <>Given only a title, the system ranks the right study first in {(f['recall@1'] * 100).toFixed(1)}% of
            1,514 held-out cases, against {(v1['recall@1'] * 100).toFixed(1)}% for the model used in version 1 (Table 1). </> : null}
            {map?.stats.topics} research fronts emerged without labels. Queries run in about ten milliseconds on a small CPU.</p>
          <p><b className="font-semibold text-ink">Conclusions.</b> A small model trained on the literature it serves can outperform a larger general one, and
            makes the shape of a field visible as well as searchable.</p>
        </div>
      </div>
    </section>
  )
}

export default function Home() {
  const [map, setMap] = useState(null)
  const [openPaper, drawer] = usePaperDrawer()
  const navigate = useNavigate()
  const [q, setQ] = useState('')
  const ref = useReveal([map])
  useEffect(() => { loadMap().then((m) => { setDomainColors(m.domains, m.colors); setMap(m) }) }, [])

  const ev = map?.retrieval
  const best = useMemo(() => ev && Math.max(...MODELS.map(([k]) => ev[k]?.['recall@1'] || 0)), [ev])
  const fieldCounts = useMemo(() => {
    if (!map) return []
    const c = new Array(map.domains.length).fill(0)
    for (let i = 0; i < map.n; i++) c[map.d[i]]++
    return map.domains.map((d, i) => [d, c[i], map.colors[i]])
  }, [map])

  return (
    <main>
      <HelixHero map={map} onOpen={openPaper} />

      <div ref={ref} className="page">
        <div className="reveal"><Abstract map={map} ev={ev} /></div>

        <section className="reveal mt-20">
          <SectionHead n={1}>Five instruments, one corpus</SectionHead>
          <ol className="ruled">
            {INSTRUMENTS.map(([to, name, text], i) => (
              <li key={to}>
                <Link to={to} className="group grid grid-cols-[2.5rem_1fr] gap-x-4 py-6 sm:grid-cols-[3rem_14rem_1fr_6rem]">
                  <span className="num pt-1.5">{String(i + 1).padStart(2, '0')}</span>
                  <span className="font-serif text-2xl text-ink group-hover:text-hema">{name}</span>
                  <span className="body col-start-2 mt-1 sm:col-start-auto sm:mt-0">{text}</span>
                  <span className="hidden pt-1.5 text-right font-sans text-sm text-ink-3 group-hover:text-hema sm:block">Open &rarr;</span>
                </Link>
              </li>
            ))}
          </ol>
        </section>

        {ev?.hybrid_bm25_finetuned && (
          <section className="reveal mt-20">
            <SectionHead n={2} aside="Held-out evaluation">How well it finds the right study</SectionHead>
            <div className="mt-6 overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse font-sans text-[14px]">
                <thead>
                  <tr className="border-y border-ink text-left">
                    <th className="py-2.5 pr-4 font-semibold">System</th>
                    <th className="py-2.5 pr-4 text-right font-semibold">Parameters</th>
                    <th className="py-2.5 pr-4 text-right font-semibold">Recall@1</th>
                    <th className="py-2.5 pr-4 text-right font-semibold">Recall@10</th>
                    <th className="py-2.5 text-right font-semibold">MRR@10</th>
                  </tr>
                </thead>
                <tbody>
                  {MODELS.map(([k, name, params]) => {
                    const r = ev[k]
                    if (!r) return null
                    const top = r['recall@1'] === best
                    return (
                      <tr key={k} className="border-b border-rule">
                        <td className="py-3 pr-4 font-serif text-[16px]">{name}</td>
                        <td className="py-3 pr-4 text-right tabular-nums text-ink-3">{params}</td>
                        <td className={`py-3 pr-4 text-right tabular-nums ${top ? 'font-semibold text-eosin' : ''}`}>
                          <span className="mr-3 inline-block h-[3px] w-20 bg-paper-3 align-middle"><span className="block h-[3px] bg-ink" style={{ width: `${((r['recall@1'] - 0.7) / (best - 0.7)) * 100}%` }} /></span>
                          {r['recall@1'].toFixed(3)}
                        </td>
                        <td className="py-3 pr-4 text-right tabular-nums">{r['recall@10'].toFixed(3)}</td>
                        <td className={`py-3 text-right tabular-nums ${top ? 'font-semibold' : ''}`}>{r['mrr@10'].toFixed(3)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="caption mt-3 max-w-3xl"><b>Table 1 | Retrieval of held-out studies.</b> Each of 1,514 titles never seen in training is
              used as a query against all {fmt(map?.stats.papers)} abstracts; the target is the study it came from. Recall@1 is the share
              ranked first. The production system is the hybrid.</p>
          </section>
        )}

        {map && (
          <section className="reveal mt-20">
            <SectionHead n={3} aside={`${fmt(map.stats.papers)} studies · ${fmt(map.stats.journals)} journals`}>Fields in the atlas</SectionHead>
            <ul className="mt-2 columns-1 gap-10 sm:columns-2 lg:columns-3">
              {fieldCounts.map(([d, c, color]) => (
                <li key={d} className="flex break-inside-avoid items-baseline gap-3 border-b border-rule py-2.5">
                  <span className="h-2.5 w-2.5 translate-y-[1px] rounded-full" style={{ background: color }} />
                  <Link to={`/search?q=${encodeURIComponent(d)}&domain=${encodeURIComponent(d)}`} className="flex-1 font-serif text-[17px] hover:text-hema">{d}</Link>
                  <span className="num">{fmt(c)}</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="reveal mt-20 grid gap-10 lg:grid-cols-[1fr_1fr]">
          <div>
            <SectionHead n={4}>Methods, briefly</SectionHead>
            <div className="body mt-5 space-y-4">
              <p><b className="font-semibold text-ink">Sampling.</b> PubMed was queried for each field and each year separately, by relevance, so that
                the record is not dominated by recent work and growth can be measured honestly.</p>
              <p><b className="font-semibold text-ink">Representation.</b> A compact encoder (bge-small) was trained with in-batch contrastive
                loss to match titles and MeSH-style queries to their abstracts. It is served as an 8-bit model of 34 MB.</p>
              <p><b className="font-semibold text-ink">Discovery.</b> UMAP reduces the representation; HDBSCAN finds dense regions; class-based
                TF-IDF names each one from its own vocabulary.</p>
            </div>
          </div>
          <div className="lg:pt-[68px]">
            <form onSubmit={(e) => { e.preventDefault(); q.trim() && navigate(`/search?q=${encodeURIComponent(q.trim())}`) }}>
              <label className="font-sans text-[13px] font-semibold" htmlFor="ask">Ask the literature</label>
              <input id="ask" value={q} onChange={(e) => setQ(e.target.value)} className="field mt-2" placeholder="resistance to checkpoint inhibitors" />
              <div className="mt-4 flex items-center justify-between">
                <span className="meta">Searches all {fmt(map?.stats.papers)} abstracts</span>
                <button className="btn-ink">Search</button>
              </div>
            </form>
          </div>
        </section>
      </div>
      {drawer}
    </main>
  )
}
