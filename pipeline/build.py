"""
Build every runtime artifact from the corpus and the fine-tuned encoder.

backend/artifacts/
  papers.jsonl.gz      paper records (+ topic id, map coordinates)
  embeddings.f16.npy   L2-normalised document vectors (float16)
  bm25.npz, vocab.json sparse BM25 matrix (IDF baked in)
  encoder.onnx         int8-quantised query encoder (CLS pooling + L2 norm inside the graph)
  tokenizer.json       fast tokenizer for the encoder
  topics.json          discovered topics: label, keywords, size, domain mix, yearly counts, growth
  trends.json          real PubMed volumes per domain/year, emerging MeSH terms, journals
  meta.json            corpus + model metrics
frontend/public/data/
  map.bin, map.json    compact 2D landscape for the scroll animation and Atlas view

  python pipeline/build.py
"""
import gzip
import re
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARTIFACTS, BM25, FRONTEND_DATA, QUERY_PREFIX, STOP, WORK, doc_text, load_papers, tokenize  # noqa: E402

MODEL_DIR = os.path.join(WORK, 'bioatlas-embed')

# Muted, print-quality palette for 24 domains (works on both bone and ink backgrounds)
PALETTE = ['#E2492F', '#D98E32', '#C9B037', '#8DA34A', '#4E9A6B', '#2E8C86', '#3A7CA5', '#5868B0', '#7C5CA8',
           '#A1528F', '#C4506E', '#B86B4B', '#8F7A55', '#6E8B74', '#4F7D8C', '#6B6FA0', '#946A9E', '#B5677D',
           '#CC7A4A', '#A49A3E', '#5E9E86', '#4C88B0', '#8B5E83', '#C08F6A']


def ctfidf_labels(texts_by_topic, n_words=8):
    """Class-based TF-IDF over unigrams+bigrams -> top keywords per topic."""
    from sklearn.feature_extraction.text import CountVectorizer
    topics = sorted(texts_by_topic)
    docs = [' '.join(texts_by_topic[t]) for t in topics]
    cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2,
                         max_df=0.5)
    X = cv.fit_transform(docs).astype(np.float32).toarray()
    tf = X / X.sum(axis=1, keepdims=True)
    idf = np.log(1 + X.sum() / X.sum(axis=0))
    scores = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    out = {}
    for row, t in enumerate(topics):
        ranked = [(vocab[j], float(scores[row, j])) for j in np.argsort(-scores[row])[:80]]
        out[t] = select_keywords(ranked, n_words)
    return out


def _stem(tok):
    if tok.endswith('ies') and len(tok) > 5:
        return tok[:-3] + 'y'
    return tok[:-1] if tok.endswith('s') and len(tok) > 4 else tok


def _stems(phrase):
    return {_stem(t) for t in re.split(r'[\s\-]+', phrase) if t}


def select_keywords(ranked, n):
    """Non-redundant keywords: a bigram absorbs the unigrams it contains, plurals and overlaps are dropped."""
    chosen = []  # list of (phrase, score, stems)
    for phrase, score in ranked:
        if len(phrase) <= 2:
            continue
        stems = _stems(phrase)
        if any(stems <= c[2] for c in chosen):
            continue  # nothing new
        absorbed = [c for c in chosen if c[2] < stems]
        rest = [c for c in chosen if c not in absorbed]
        if any(stems & c[2] for c in rest):
            continue  # partial overlap with another keyword adds little
        chosen = rest  # a more specific phrase replaces the unigrams it contains ("gut" -> "gut microbiome")
        chosen.append((phrase, score, stems))
        if len(chosen) == n:
            break
    return [(p, s) for p, s, _ in chosen]


ACRONYMS = {'crispr': 'CRISPR', 'mrna': 'mRNA', 'covid': 'COVID', 'covid-19': 'COVID-19', 'sars-cov-2': 'SARS-CoV-2',
            'car-t': 'CAR-T', 'glp-1': 'GLP-1', 'pd-1': 'PD-1', 'pd-l1': 'PD-L1', 'ctdna': 'ctDNA', 'scrna-seq': 'scRNA-seq',
            'rna-seq': 'RNA-seq', 'alphafold': 'AlphaFold', 'mri': 'MRI', 'ct': 'CT', 'ai': 'AI', 'hiv': 'HIV',
            'aav': 'AAV', 'hcc': 'HCC', 'dna': 'DNA', 'rna': 'RNA', 'ipsc': 'iPSC', 'ipscs': 'iPSCs', 'ad': "Alzheimer's",
            'pd': "Parkinson's", 'mdd': 'MDD', 'ecg': 'ECG', 'nsclc': 'NSCLC', 'icu': 'ICU', 'ici': 'ICI', 'icis': 'ICIs',
            'sle': 'SLE', 'amr': 'AMR', 'nps': 'NPs', 'evs': 'EVs', 'msc': 'MSC', 'mscs': 'MSCs', 'tau': 'Tau', 'ibd': 'IBD',
            'covid-19': 'COVID-19', 'cas9': 'Cas9', 'egfr': 'EGFR', 'tme': 'TME', 'bcma': 'BCMA', 'raav': 'rAAV', 'her2': 'HER2', 'tbi': 'TBI', 'copd': 'COPD', 'ckd': 'CKD', 'nash': 'NASH', 'masld': 'MASLD', 't-cell': 'T-cell', 'b-cell': 'B-cell', 'car': 'CAR', 'hba1c': 'HbA1c'}


def pretty(phrase):
    def fmt(t):
        if t in ACRONYMS:
            return ACRONYMS[t]
        if len(t) <= 4 and t.isalpha() and not set(t) & set('aeiouy'):
            return t.upper()  # vowel-less short tokens are acronyms (CSF, CT, MRI)
        return '-'.join(ACRONYMS.get(p, p.capitalize()) for p in t.split('-'))
    return ' '.join(fmt(t) for t in phrase.split())


def topic_label(keywords, k=3):
    return ' · '.join(pretty(w) for w, _ in keywords[:k])


def growth(yearly, recent=(2023, 2025), base=(2016, 2019)):
    r = np.mean([yearly.get(str(y), 0) for y in range(recent[0], recent[1] + 1)])
    b = np.mean([yearly.get(str(y), 0) for y in range(base[0], base[1] + 1)])
    return float((r + 1) / (b + 1))


def export_onnx(model_dir, out_dir):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer

    class Encoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            h = self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
            cls = h[:, 0]  # bge uses CLS pooling
            return torch.nn.functional.normalize(cls, dim=-1)

    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = Encoder(AutoModel.from_pretrained(model_dir)).eval()
    ex = tok(['example query about crispr'], return_tensors='pt')
    fp32 = os.path.join(WORK, 'encoder_fp32.onnx')
    names = ['input_ids', 'attention_mask', 'token_type_ids']
    torch.onnx.export(enc, tuple(ex[n] for n in names), fp32, input_names=names, output_names=['embedding'],
                      dynamic_axes={**{n: {0: 'batch', 1: 'seq'} for n in names}, 'embedding': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    quantize_dynamic(fp32, os.path.join(out_dir, 'encoder.onnx'), weight_type=QuantType.QInt8, per_channel=True)
    tok.backend_tokenizer.save(os.path.join(out_dir, 'tokenizer.json'))
    return fp32


def main():
    import umap
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import HDBSCAN

    os.makedirs(ARTIFACTS, exist_ok=True)
    os.makedirs(FRONTEND_DATA, exist_ok=True)
    t0 = time.time()
    papers = load_papers()
    docs = [doc_text(p) for p in papers]
    n = len(papers)

    # 1) document embeddings with the fine-tuned encoder
    model = SentenceTransformer(MODEL_DIR, device='cuda')
    E = model.encode(docs, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    np.save(os.path.join(ARTIFACTS, 'embeddings.f16.npy'), E.astype(np.float16))
    print(f'embedded {n} papers in {time.time() - t0:.0f}s')

    # 2) BM25
    bm25 = BM25().fit(docs)
    from scipy import sparse
    sparse.save_npz(os.path.join(ARTIFACTS, 'bm25.npz'), bm25.matrix)
    with open(os.path.join(ARTIFACTS, 'vocab.json'), 'w') as f:
        json.dump({k: int(v) for k, v in bm25.vocab.items()}, f)

    # 3) topics: UMAP(5d) -> HDBSCAN; outliers joined to the nearest topic centroid
    u5 = umap.UMAP(n_components=5, n_neighbors=20, min_dist=0.0, metric='cosine', random_state=42).fit_transform(E)
    labels = HDBSCAN(min_cluster_size=max(25, n // 400), min_samples=10).fit_predict(u5)
    ids = sorted(set(labels) - {-1})
    cent = np.stack([E[labels == t].mean(0) for t in ids])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    out_mask = labels == -1
    labels[out_mask] = np.array(ids)[np.argmax(E[out_mask] @ cent.T, axis=1)]
    remap = {t: i for i, t in enumerate(ids)}
    labels = np.array([remap[t] for t in labels])
    print(f'{len(ids)} topics ({out_mask.mean():.0%} outliers re-assigned)')

    # 4) 2D landscape map
    u2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.08, spread=1.2, metric='cosine', random_state=7).fit_transform(E)
    u2 = (u2 - u2.mean(0)) / np.abs(u2 - u2.mean(0)).max()

    # 5) topic descriptions
    by_topic = defaultdict(list)
    for p, t in zip(papers, labels):
        by_topic[int(t)].append(f"{p['title']} {p['title']} {p['abstract']}")
    kw = ctfidf_labels(by_topic)
    domains = sorted({p['domain'] for p in papers})
    dom_idx = {d: i for i, d in enumerate(domains)}
    topics = []
    for t in range(len(ids)):
        members = np.where(labels == t)[0]
        yearly = Counter(str(papers[i]['year']) for i in members)
        dmix = Counter(papers[i]['domain'] for i in members)
        mesh = Counter(m for i in members for m in papers[i]['mesh'])
        c = E[members].mean(0)
        rep = members[np.argsort(-(E[members] @ c))[:8]]
        words = [w for w, _ in kw[t]]
        topics.append({
            'id': t, 'label': topic_label(kw[t]), 'keywords': kw[t], 'size': int(len(members)),
            'domain': dmix.most_common(1)[0][0], 'domain_mix': dmix.most_common(5),
            'yearly': dict(sorted(yearly.items())), 'growth': round(growth(yearly), 3),
            'top_mesh': mesh.most_common(10), 'representative': [papers[i]['pmid'] for i in rep],
            'x': float(u2[members, 0].mean()), 'y': float(u2[members, 1].mean()),
        })
    with open(os.path.join(ARTIFACTS, 'topics.json'), 'w') as f:
        json.dump(topics, f)

    # 6) papers (+ topic, coords)
    with gzip.open(os.path.join(ARTIFACTS, 'papers.jsonl.gz'), 'wt') as f:
        for i, p in enumerate(papers):
            rec = {k: p.get(k) for k in ('pmid', 'title', 'abstract', 'journal', 'year', 'month', 'mesh', 'keywords',
                                         'doi', 'domain', 'pub_types')}
            rec['authors'] = p['authors'][:8]
            rec['n_authors'] = len(p['authors'])
            rec['topic'] = int(labels[i])
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # 7) trends: real PubMed volumes + emerging MeSH terms within the corpus
    counts_path = os.path.join(WORK, 'pubmed_counts.json')
    pubmed_counts = json.load(open(counts_path)) if os.path.exists(counts_path) else {}
    early = Counter(m for p in papers if p['year'] and p['year'] <= 2018 for m in p['mesh'])
    late = Counter(m for p in papers if p['year'] and p['year'] >= 2022 for m in p['mesh'])
    n_early = sum(1 for p in papers if p['year'] and p['year'] <= 2018) or 1
    n_late = sum(1 for p in papers if p['year'] and p['year'] >= 2022) or 1
    generic = {'Humans', 'Animals', 'Female', 'Male', 'Mice', 'Adult', 'Middle Aged', 'Aged', 'Child', 'Adolescent'}
    emerging = sorted(((m, (late[m] / n_late + 1e-4) / (early[m] / n_early + 1e-4), late[m]) for m in late
                       if late[m] >= 15 and m not in generic), key=lambda x: -x[1])[:25]
    trends = {
        'pubmed_counts': pubmed_counts,
        'corpus_yearly': dict(sorted(Counter(str(p['year']) for p in papers).items())),
        'emerging_mesh': [{'term': m, 'lift': round(l, 2), 'recent_papers': c} for m, l, c in emerging],
        'top_journals': Counter(p['journal'] for p in papers).most_common(15),
        'top_mesh': [(m, c) for m, c in Counter(m for p in papers for m in p['mesh']).most_common(40) if m not in generic][:20],
    }
    with open(os.path.join(ARTIFACTS, 'trends.json'), 'w') as f:
        json.dump(trends, f)

    # 8) ONNX query encoder (int8) + parity check against the PyTorch model
    fp32 = export_onnx(MODEL_DIR, ARTIFACTS)
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(ARTIFACTS, 'tokenizer.json'))
    sess = ort.InferenceSession(os.path.join(ARTIFACTS, 'encoder.onnx'), providers=['CPUExecutionProvider'])
    probe = [QUERY_PREFIX + q for q in ['CAR-T cytokine release syndrome', 'gut microbiome and depression',
                                        'deep learning chest x-ray', 'epigenetic clock mortality']]
    encs = tk.encode_batch(probe)
    L = max(len(e.ids) for e in encs)
    feed = {'input_ids': np.array([e.ids + [0] * (L - len(e.ids)) for e in encs], np.int64),
            'attention_mask': np.array([e.attention_mask + [0] * (L - len(e.ids)) for e in encs], np.int64)}
    feed['token_type_ids'] = np.zeros_like(feed['input_ids'])
    q_onnx = sess.run(['embedding'], feed)[0]
    q_torch = model.encode(probe, normalize_embeddings=True)
    cos = float(np.mean(np.sum(q_onnx * q_torch, axis=1)))
    print(f'ONNX int8 vs torch cosine: {cos:.4f}  ({os.path.getsize(os.path.join(ARTIFACTS, "encoder.onnx")) / 1e6:.1f} MB)')
    os.remove(fp32)

    # 9) metadata (+ retrieval benchmark)
    ev_path = os.path.join(WORK, 'retrieval_eval.json')
    meta = {
        'papers': n, 'domains': domains, 'domain_colors': dict(zip(domains, PALETTE)), 'topics': len(ids),
        'journals': len({p['journal'] for p in papers}), 'years': [min(p['year'] for p in papers), max(p['year'] for p in papers)],
        'authors': len({a for p in papers for a in p['authors']}), 'mesh_terms': len({m for p in papers for m in p['mesh']}),
        'encoder': {'base': 'BAAI/bge-small-en-v1.5', 'fine_tuned': True, 'dim': int(E.shape[1]), 'onnx_int8_cosine_vs_fp32': round(cos, 4)},
        'retrieval_eval': json.load(open(ev_path)) if os.path.exists(ev_path) else None,
        'built_at': time.strftime('%Y-%m-%d'),
    }
    with open(os.path.join(ARTIFACTS, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    # 10) compact map for the frontend (10 bytes/paper): int16 x, int16 y, uint8 domain, uint8 topic, uint32 pmid
    xy = np.clip(np.round(u2 * 32767), -32767, 32767).astype(np.int16)
    buf = np.zeros(n, dtype=[('x', '<i2'), ('y', '<i2'), ('d', 'u1'), ('t', 'u1'), ('p', '<u4')])
    buf['x'], buf['y'] = xy[:, 0], xy[:, 1]
    buf['p'] = [int(p['pmid']) for p in papers]
    buf['d'] = [dom_idx[p['domain']] for p in papers]
    buf['t'] = labels.astype(np.uint8)
    buf.tofile(os.path.join(FRONTEND_DATA, 'map.bin'))
    with open(os.path.join(FRONTEND_DATA, 'map.json'), 'w') as f:
        json.dump({'count': n, 'stride': 10, 'domains': domains, 'colors': PALETTE[:len(domains)],
                   'topics': [{'id': t['id'], 'label': t['label'], 'x': t['x'], 'y': t['y'], 'size': t['size'],
                               'domain': t['domain']} for t in topics],
                   'stats': {k: meta[k] for k in ('papers', 'topics', 'journals', 'years', 'authors', 'mesh_terms')},
                   'retrieval': {k: v for k, v in (meta['retrieval_eval'] or {}).items() if not k.startswith('_')}},
                  f, separators=(',', ':'))
    # 11) titles in map order, for instant hover on the landing helix and the Atlas
    rows = []
    for p in papers:
        t = p['title'].rstrip('.')
        rows.append([t if len(t) <= 150 else t[:147].rstrip() + '...', p['year'], (p['journal'] or '')[:60]])
    with open(os.path.join(FRONTEND_DATA, 'titles.json'), 'w') as f:
        json.dump(rows, f, ensure_ascii=False, separators=(',', ':'))
    print(f'done in {time.time() - t0:.0f}s')


def relabel():
    """Recompute topic keywords/labels from saved assignments (artifacts) without re-clustering."""
    papers = [json.loads(l) for l in gzip.open(os.path.join(ARTIFACTS, 'papers.jsonl.gz'), 'rt')]
    by_topic = defaultdict(list)
    for p in papers:
        by_topic[p['topic']].append(f"{p['title']} {p['title']} {p['abstract']}")
    kw = ctfidf_labels(by_topic)
    topics = json.load(open(os.path.join(ARTIFACTS, 'topics.json')))
    for t in topics:
        t['keywords'], t['label'] = kw[t['id']], topic_label(kw[t['id']])
    json.dump(topics, open(os.path.join(ARTIFACTS, 'topics.json'), 'w'))
    mpath = os.path.join(FRONTEND_DATA, 'map.json')
    m = json.load(open(mpath))
    labels = {t['id']: t['label'] for t in topics}
    for t in m['topics']:
        t['label'] = labels[t['id']]
    json.dump(m, open(mpath, 'w'), separators=(',', ':'))
    for t in sorted(topics, key=lambda t: -t['size'])[:20]:
        print(f"{t['size']:5d}  {t['label']}")


if __name__ == '__main__':
    relabel() if '--relabel' in sys.argv else main()
