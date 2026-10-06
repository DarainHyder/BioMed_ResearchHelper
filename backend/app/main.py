"""BioAtlas API: hybrid literature search, research briefs, topics, trends."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from . import brief as brief_mod
from . import search as search_mod
from .config import CORS_ORIGIN_REGEX, CORS_ORIGINS, GEMINI_API_KEY, MAX_CONCURRENT_BRIEFS
from .store import Store

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
STATE = {}


@asynccontextmanager
async def lifespan(app):
    STATE['store'] = await asyncio.to_thread(Store)
    STATE['brief_slots'] = asyncio.Semaphore(MAX_CONCURRENT_BRIEFS)
    STATE['started'] = time.time()
    yield


app = FastAPI(title='BioAtlas API', version='2.0.0', lifespan=lifespan,
              description='Hybrid semantic search, research briefs, topic discovery and trends over PubMed abstracts.')
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_origin_regex=CORS_ORIGIN_REGEX,
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


def store():
    return STATE['store']


class BriefRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(8, ge=3, le=15)
    domain: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None


@app.get('/')
def root():
    return {'name': 'BioAtlas API', 'version': '2.0.0', 'docs': '/docs', 'status': 'ok'}


@app.get('/api/health')
def health():
    s = store()
    return {'status': 'ok', 'papers': len(s.papers), 'uptime_s': int(time.time() - STATE['started']),
            'llm_synthesis': bool(GEMINI_API_KEY)}


@app.get('/api/stats')
def stats():
    s = store()
    m = s.meta
    return {**{k: m[k] for k in ('papers', 'topics', 'journals', 'years', 'authors', 'mesh_terms', 'domains',
                                  'domain_colors', 'encoder', 'built_at')},
            'retrieval_eval': m.get('retrieval_eval'), 'llm_synthesis': bool(GEMINI_API_KEY)}


@app.get('/api/search')
def search(q: str = Query(..., min_length=2, max_length=300), k: int = Query(20, ge=1, le=50),
           offset: int = Query(0, ge=0, le=250), mode: str = Query('hybrid', pattern='^(hybrid|semantic|keyword)$'),
           domain: Optional[List[str]] = Query(None), year_from: Optional[int] = None, year_to: Optional[int] = None,
           sort: str = Query('relevance', pattern='^(relevance|newest)$')):
    t0 = time.perf_counter()
    out = search_mod.search(store(), q, k, mode, domain, year_from, year_to, sort, offset)
    out['took_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.get('/api/papers/{pmid}')
def paper(pmid: str):
    s = store()
    if pmid not in s.by_pmid:
        raise HTTPException(404, 'Paper not found')
    p = s.papers[s.by_pmid[pmid]]
    return {**p, 'topic_label': s.topics[p['topic']]['label'], 'similar': search_mod.similar(s, pmid, 6)}


@app.post('/api/brief')
async def brief(req: BriefRequest):
    async with STATE['brief_slots']:
        t0 = time.perf_counter()
        out = await asyncio.to_thread(brief_mod.build_brief, store(), req.question, req.k, req.domain,
                                      req.year_from, req.year_to)
    out['took_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.get('/api/topics')
def topics(domain: Optional[str] = None, sort: str = Query('size', pattern='^(size|growth)$')):
    items = [{k: t[k] for k in ('id', 'label', 'keywords', 'size', 'domain', 'domain_mix', 'yearly', 'growth', 'x', 'y')}
             for t in store().topics if not domain or t['domain'] == domain]
    items.sort(key=lambda t: -t[sort])
    return {'topics': items}


@app.get('/api/topics/{topic_id}')
def topic(topic_id: int):
    s = store()
    if not 0 <= topic_id < len(s.topics):
        raise HTTPException(404, 'Topic not found')
    t = s.topics[topic_id]
    return {**t, 'papers': [search_mod.card(s, s.by_pmid[p]) for p in t['representative']]}


@app.get('/api/trends')
def trends():
    return store().trends
