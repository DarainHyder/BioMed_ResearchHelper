"""Shared helpers for the offline pipeline (also mirrored by backend/app/text.py at runtime)."""
import hashlib
import json
import os
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'pipeline', 'work')
ARTIFACTS = os.path.join(ROOT, 'backend', 'artifacts')
FRONTEND_DATA = os.path.join(ROOT, 'frontend', 'public', 'data')

BASE_MODEL = 'BAAI/bge-small-en-v1.5'
LEGACY_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # what v1 used
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]")
STOP = set(ENGLISH_STOP_WORDS) | {'study', 'studies', 'patients', 'results', 'methods', 'conclusion', 'conclusions',
                                  'background', 'objective', 'objectives', 'using', 'used', 'based', 'also', 'however'}


def clean(text):
    text = unicodedata.normalize('NFKC', text or '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]


def doc_text(p):
    return f"{p['title']} {p['abstract']}"


def load_papers(path=None):
    path = path or os.path.join(WORK, 'raw_papers.jsonl')
    papers = []
    seen = set()
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            if p['pmid'] in seen:
                continue
            seen.add(p['pmid'])
            p['title'], p['abstract'] = clean(p['title']), clean(p['abstract'])
            papers.append(p)
    return papers


def is_heldout(pmid, frac=0.1):
    """Deterministic held-out split by PMID hash (never used for fine-tuning)."""
    return int(hashlib.md5(pmid.encode()).hexdigest(), 16) % 1000 < frac * 1000


class BM25:
    """Sparse BM25 with IDF baked into the document matrix, so scoring is a single sparse mat-vec."""

    def __init__(self, k1=1.2, b=0.75, min_df=2):
        self.k1, self.b, self.min_df = k1, b, min_df

    def fit(self, texts):
        from scipy import sparse
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=self.min_df,
                             ngram_range=(1, 1), dtype=np.float32)
        tf = cv.fit_transform(texts).tocsr()
        self.vocab = cv.vocabulary_
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.matrix = sparse.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        return self

    def query_vec(self, text):
        idx = sorted({self.vocab[t] for t in tokenize(text) if t in self.vocab})
        return idx

    def scores(self, text):
        idx = self.query_vec(text)
        if not idx:
            return np.zeros(self.matrix.shape[0], np.float32)
        return np.asarray(self.matrix[:, idx].sum(axis=1)).ravel()
