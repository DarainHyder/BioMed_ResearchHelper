"""
Research briefs: evidence synthesis over the top papers for a question.

Always available (no API key): extractive, query-focused multi-document summarisation.
  1. hybrid retrieval of the top papers
  2. split abstracts into sentences, embed them with the same ONNX encoder
  3. score = relevance to the question + bonus for Results/Conclusions sentences
  4. Maximal Marginal Relevance selection (diverse, at most 2 sentences per paper),
     every sentence keeps a numbered citation
  5. evidence profile: study designs, year span, recurring MeSH concepts

Optional: if GEMINI_API_KEY is set, a short abstractive synthesis grounded only in the
selected, numbered sentences is added.
"""
import logging
import re
from collections import Counter

import httpx
import numpy as np

from .config import GEMINI_API_KEY, GEMINI_MODEL
from .search import SENT_RE, build_mask, card, score

log = logging.getLogger('brief')

LABEL_RE = re.compile(r'^(Background|Introduction|Objectives?|Aims?|Purpose|Methods?|Design|Setting|Results?|Findings|'
                      r'Conclusions?|Interpretation|Significance|Discussion)\s*:\s*', re.I)
OUTCOME_LABELS = {'results', 'result', 'findings', 'conclusion', 'conclusions', 'interpretation', 'significance'}
DESIGN_TYPES = ['Meta-Analysis', 'Systematic Review', 'Randomized Controlled Trial', 'Clinical Trial', 'Review',
                'Observational Study', 'Comparative Study', 'Case Reports']
GENERIC_MESH = {'Humans', 'Animals', 'Female', 'Male', 'Mice', 'Adult', 'Middle Aged', 'Aged', 'Child', 'Adolescent'}
# Sentences about the paper itself rather than its findings ("This review summarises...", "Herein, we...")
META_RE = re.compile(r'^(here(in)?\b|in this (review|article|paper|chapter|study)\b)|\bwe (also |will |further |then |first |here )?'
                     r'(review|summari[sz]e|discuss|describe|highlight|assess|explore|examine|provide|present|focus|intend|'
                     r'aim|expound|outline|introduce|propose to)'
                     r'|\b(this|the present|our) (narrative |systematic |comprehensive )?(review|article|paper|chapter)\b', re.I)
JUNK_RE = re.compile(r'PROSPERO|CRD4\d|ClinicalTrials\.gov|registration|^\(|^\[|doi\.org|©', re.I)
CONTEXT_LABELS = {'background', 'introduction', 'objective', 'objectives', 'aim', 'aims', 'purpose', 'methods', 'method',
                  'design', 'setting'}
NUMERIC_RE = re.compile(r'\d+(\.\d+)?\s?(%|percent|fold|mg|months?|years?|patients|participants)|[pP]\s?[<=]\s?0?\.\d+|'
                        r'\b(HR|OR|RR|AUC|CI)\b')


def _sentences(paper):
    out, label = [], None
    for raw in SENT_RE.split(paper['abstract']):
        m = LABEL_RE.match(raw)
        if m:
            label = m.group(1).lower()
            raw = raw[m.end():]
        raw = raw.strip()
        if 40 <= len(raw) <= 600 and not JUNK_RE.search(raw):
            out.append((raw, label))
    return out


def _mmr(sim_q, sim_ss, owners, n, lam=0.72, per_owner=2):
    chosen, used = [], Counter()
    cand = list(range(len(sim_q)))
    while cand and len(chosen) < n:
        best, best_val = None, -1e9
        for c in cand:
            if used[owners[c]] >= per_owner:
                continue
            red = max((sim_ss[c, j] for j in chosen), default=0.0)
            val = lam * sim_q[c] - (1 - lam) * red
            if val > best_val:
                best, best_val = c, val
        if best is None:
            break
        chosen.append(best)
        used[owners[best]] += 1
        cand.remove(best)
    return chosen


def _llm_synthesis(question, numbered):
    if not GEMINI_API_KEY:
        return None
    prompt = (
        'You are a biomedical research analyst. Using ONLY the numbered evidence below, write a 4-6 sentence '
        'synthesis that answers the question. Cite sources inline like [1] or [2][4]. State limitations or '
        'disagreements if present. Do not add facts that are not in the evidence.\n\n'
        f'Question: {question}\n\nEvidence:\n' + '\n'.join(f'[{i}] {t}' for i, t in numbered)
    )
    try:
        r = httpx.post(f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent',
                       params={'key': GEMINI_API_KEY}, timeout=25,
                       json={'contents': [{'parts': [{'text': prompt}]}], 'generationConfig': {'temperature': 0.3}})
        r.raise_for_status()
        return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as exc:  # the extractive brief is always returned anyway
        log.warning('LLM synthesis unavailable: %s', str(exc)[:200])
        return None


def build_brief(store, question, k=8, domain=None, year_from=None, year_to=None, n_sentences=6):
    mask = build_mask(store, domain, year_from, year_to)
    fused, cos, _ = score(store, question, 'hybrid', mask)
    top = np.argsort(-fused)[:k]
    top = [int(i) for i in top if np.isfinite(fused[i])]
    if not top:
        return {'question': question, 'summary': [], 'papers': [], 'key_findings': [], 'evidence': {}}

    sents, owners, labels = [], [], []
    for rank, i in enumerate(top):
        for text, label in _sentences(store.papers[i]):
            sents.append(text); owners.append(rank); labels.append(label)
    S = store.encoder.encode(sents, max_len=96)
    q = store.encoder.query(question)
    rel = S @ q
    bonus = np.array([(0.06 if (l or '') in OUTCOME_LABELS else -0.04 if (l or '') in CONTEXT_LABELS else 0.0)
                      - (0.15 if META_RE.search(t) else 0.0) for t, l in zip(sents, labels)])
    # unlabelled abstracts usually open with generic background
    first = np.array([1.0 if k == 0 or owners[k] != owners[k - 1] else 0.0 for k in range(len(sents))])
    bonus -= 0.03 * first * np.array([l is None for l in labels])
    rank_prior = np.array([0.03 * (1 - o / max(1, len(top))) for o in owners])
    sim_q = rel + bonus + rank_prior
    chosen = _mmr(sim_q, S @ S.T, owners, n_sentences)
    chosen.sort(key=lambda c: (owners[c], c))  # keep each paper's sentences in reading order

    findings_idx = sorted((c for c in range(len(sents)) if NUMERIC_RE.search(sents[c]) and (labels[c] or '') not in CONTEXT_LABELS
                           and rel[c] > 0.45), key=lambda c: -sim_q[c])
    seen_owner, key_findings = set(), []
    for c in findings_idx:
        if owners[c] in seen_owner:
            continue
        seen_owner.add(owners[c])
        key_findings.append({'text': sents[c], 'cite': owners[c] + 1})
        if len(key_findings) == 4:
            break

    cited = [store.papers[i] for i in top]
    designs = Counter(t for p in cited for t in p.get('pub_types', []) if t in DESIGN_TYPES)
    years = [p['year'] for p in cited if p['year']]
    mesh = Counter(m for p in cited for m in p['mesh'] if m not in GENERIC_MESH)
    summary = [{'text': sents[c], 'cite': owners[c] + 1, 'section': labels[c]} for c in chosen]
    return {
        'question': question,
        'summary': summary,
        'key_findings': key_findings,
        'synthesis': _llm_synthesis(question, [(s['cite'], s['text']) for s in summary]),
        'papers': [card(store, i, cos) for i in top],
        'evidence': {
            'study_designs': designs.most_common(),
            'year_span': [min(years), max(years)] if years else None,
            'concepts': mesh.most_common(8),
            'domains': Counter(p['domain'] for p in cited).most_common(),
            'mean_relevance': round(float(np.mean([cos[i] for i in top])), 3),
        },
        'method': 'extractive-mmr' + ('+llm' if GEMINI_API_KEY else ''),
    }
