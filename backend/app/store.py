"""Loads the prebuilt artifacts once at startup and exposes them read-only."""
import gzip
import json
import logging
import os
import re
import time
from collections import OrderedDict
from threading import Lock

import numpy as np
from scipy import sparse

from .config import ARTIFACTS, QUERY_PREFIX

log = logging.getLogger('store')

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]")


class Encoder:
    """int8 ONNX bge-small encoder + fast tokenizer. ~5-15 ms per query on a small CPU."""

    def __init__(self, path):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get('ORT_THREADS', 2))
        self.session = ort.InferenceSession(os.path.join(path, 'encoder.onnx'), opts, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(os.path.join(path, 'tokenizer.json'))
        self.cache = OrderedDict()
        self.lock = Lock()

    def encode(self, texts, max_len=128):
        self.tokenizer.enable_truncation(max_len)
        encs = self.tokenizer.encode_batch(texts)
        L = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), L), np.int64)
        mask = np.zeros((len(encs), L), np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = e.attention_mask
        return self.session.run(['embedding'], {'input_ids': ids, 'attention_mask': mask,
                                                'token_type_ids': np.zeros_like(ids)})[0]

    def query(self, text):
        key = text.strip().lower()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        vec = self.encode([QUERY_PREFIX + text])[0]
        with self.lock:
            self.cache[key] = vec
            if len(self.cache) > 2048:
                self.cache.popitem(last=False)
        return vec


class Store:
    def __init__(self, path=ARTIFACTS):
        t0 = time.time()
        with gzip.open(os.path.join(path, 'papers.jsonl.gz'), 'rt') as f:
            self.papers = [json.loads(line) for line in f]
        self.by_pmid = {p['pmid']: i for i, p in enumerate(self.papers)}
        self.E = np.load(os.path.join(path, 'embeddings.f16.npy')).astype(np.float32)
        self.bm25 = sparse.load_npz(os.path.join(path, 'bm25.npz')).tocsc()
        with open(os.path.join(path, 'vocab.json')) as f:
            self.vocab = json.load(f)
        self.inv_vocab = [''] * (max(self.vocab.values()) + 1)
        for term, j in self.vocab.items():
            self.inv_vocab[j] = term
        with open(os.path.join(path, 'topics.json')) as f:
            self.topics = json.load(f)
        with open(os.path.join(path, 'trends.json')) as f:
            self.trends = json.load(f)
        with open(os.path.join(path, 'meta.json')) as f:
            self.meta = json.load(f)
        self.years = np.array([p['year'] or 0 for p in self.papers], np.int16)
        self.domains = self.meta['domains']
        dom_idx = {d: i for i, d in enumerate(self.domains)}
        self.domain_ids = np.array([dom_idx[p['domain']] for p in self.papers], np.int16)
        self.topic_ids = np.array([p['topic'] for p in self.papers], np.int16)
        self.encoder = Encoder(path)
        self.encoder.query('warm up')
        log.info('Loaded %d papers, %d topics in %.1fs', len(self.papers), len(self.topics), time.time() - t0)

    def bm25_scores(self, text):
        # Stop words were excluded when the vocabulary was built, so vocab membership is the only filter.
        idx = sorted({self.vocab[t] for t in TOKEN_RE.findall(text.lower()) if t in self.vocab})
        if not idx:
            return np.zeros(len(self.papers), np.float32), []
        return np.asarray(self.bm25[:, idx].sum(axis=1)).ravel(), idx
