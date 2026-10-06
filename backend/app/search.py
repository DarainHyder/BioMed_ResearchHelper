"""Hybrid retrieval: BM25 + fine-tuned dense vectors fused with reciprocal rank fusion."""
import re
from collections import Counter

import numpy as np

RRF_K = 60
SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z(\[])')


def _ranks(scores, mask):
    s = np.where(mask, scores, -np.inf)
    order = np.argsort(-s)
    ranks = np.empty(len(s), np.int64)
    ranks[order] = np.arange(len(s))
    return ranks


def build_mask(store, domain=None, year_from=None, year_to=None, topic=None):
    mask = np.ones(len(store.papers), bool)
    if domain:
        wanted = [store.domains.index(d) for d in domain if d in store.domains]
        mask &= np.isin(store.domain_ids, wanted)
    if year_from:
        mask &= store.years >= year_from
    if year_to:
        mask &= store.years <= year_to
    if topic is not None:
        mask &= store.topic_ids == topic
    return mask


def score(store, query, mode='hybrid', mask=None):
    """Returns (fused_scores, cosine_similarities, matched_term_ids)."""
    mask = np.ones(len(store.papers), bool) if mask is None else mask
    q = store.encoder.query(query)
    cos = store.E @ q
    if mode == 'semantic':
        return np.where(mask, cos, -np.inf), cos, []
    bm, terms = store.bm25_scores(query)
    if mode == 'keyword':
        return np.where(mask & (bm > 0), bm, -np.inf), cos, terms
    fused = 1.0 / (RRF_K + 1 + _ranks(cos, mask)) + 1.0 / (RRF_K + 1 + _ranks(bm, mask & (bm > 0)))
    return np.where(mask, fused, -np.inf), cos, terms


def snippet(paper, query_terms):
    """Abstract sentence with the most query-term hits (falls back to the first sentence)."""
    sents = SENT_RE.split(paper['abstract'])
    if not query_terms:
        return sents[0][:320]
    best = max(sents, key=lambda s: sum(t in s.lower() for t in query_terms))
    return best[:320]


def card(store, i, cos=None, terms=()):
    p = store.papers[i]
    return {
        'pmid': p['pmid'], 'title': p['title'], 'journal': p['journal'], 'year': p['year'], 'domain': p['domain'],
        'topic': p['topic'], 'authors': p['authors'][:3], 'n_authors': p['n_authors'], 'doi': p.get('doi'),
        'pub_types': [t for t in p.get('pub_types', []) if t != 'Journal Article'][:2],
        'similarity': None if cos is None else round(float(cos[i]), 4),
        'snippet': snippet(p, terms),
    }


def search(store, query, k=20, mode='hybrid', domain=None, year_from=None, year_to=None, sort='relevance', offset=0):
    mask = build_mask(store, domain, year_from, year_to)
    fused, cos, term_ids = score(store, query, mode, mask)
    inv_vocab_terms = [store.inv_vocab[j] for j in term_ids]
    valid = np.isfinite(fused)
    n_valid = int(valid.sum())
    pool = min(n_valid, 300)
    top = np.argpartition(-fused, pool - 1)[:pool] if pool else np.array([], int)
    top = top[np.argsort(-fused[top])]
    # relevance floor so facets describe genuinely related papers
    rel = top[cos[top] >= max(0.45, float(cos[top[0]]) - 0.2)] if len(top) else top
    if sort == 'newest':
        top = top[:100][np.argsort(-store.years[top[:100]], kind='stable')]
    page = top[offset:offset + k]
    facets = {
        'domains': Counter(store.papers[i]['domain'] for i in rel).most_common(),
        'years': dict(sorted(Counter(int(store.years[i]) for i in rel).items())),
        'topics': [{'id': t, 'label': store.topics[t]['label'], 'count': c}
                   for t, c in Counter(int(store.topic_ids[i]) for i in rel).most_common(6)],
    }
    return {
        'query': query, 'mode': mode, 'total': len(rel), 'results': [card(store, int(i), cos, inv_vocab_terms) for i in page],
        'facets': facets, 'terms': inv_vocab_terms,
    }


def similar(store, pmid, k=8):
    i = store.by_pmid[pmid]
    s = store.E @ store.E[i]
    s[i] = -np.inf
    top = np.argpartition(-s, k)[:k]
    return [card(store, int(j), s) for j in top[np.argsort(-s[top])]]
