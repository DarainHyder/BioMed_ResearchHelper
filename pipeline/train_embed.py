"""
Domain-adapt a compact embedding model to the corpus and benchmark retrieval.

Training pairs (held-out papers excluded):
  * title            -> title + abstract     (find the paper a title describes)
  * MeSH/keyword query -> title + abstract   (concept-style queries, like real searches)
Loss: MultipleNegativesRankingLoss (in-batch negatives), GPU, bf16.

Evaluation on held-out papers: each held-out title is a query, the corpus is EVERY paper
(~17k), the target is the query's own paper. Reports Recall@1/10, MRR@10, and domain
precision@10 (topical coherence of the top 10) for BM25, v1's all-mpnet-base-v2,
base bge-small, the fine-tuned model, and hybrid (BM25 + fine-tuned, reciprocal rank fusion).

  python pipeline/train_embed.py --epochs 3
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_MODEL, BM25, LEGACY_MODEL, QUERY_PREFIX, WORK, doc_text, is_heldout, load_papers  # noqa: E402

MODEL_OUT = os.path.join(WORK, 'bioatlas-embed')


def mesh_query(p, rng):
    terms = [t for t in p.get('mesh', []) + p.get('keywords', []) if len(t) > 3]
    if len(terms) < 2:
        return None
    rng.shuffle(terms)
    return ' '.join(terms[:rng.randint(2, 4)])


def rrf(*rankings, k=60):
    """Reciprocal rank fusion of several score vectors -> fused score vector."""
    fused = np.zeros_like(rankings[0], dtype=np.float32)
    for s in rankings:
        order = np.argsort(-s)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        fused += 1.0 / (k + ranks + 1)
    return fused


def evaluate(name, score_fn, queries, targets, domains, all_domains, k=10):
    r1 = r10 = mrr = dprec = 0.0
    t0 = time.time()
    for q, tgt, dom in zip(queries, targets, domains):
        s = score_fn(q)
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        hits = np.where(top == tgt)[0]
        if len(hits):
            r10 += 1
            mrr += 1 / (hits[0] + 1)
            r1 += hits[0] == 0
        dprec += np.mean([all_domains[i] == dom for i in top])
    n = len(queries)
    res = {'recall@1': r1 / n, 'recall@10': r10 / n, 'mrr@10': mrr / n, 'domain_precision@10': dprec / n,
           'ms_per_query': 1000 * (time.time() - t0) / n}
    print(f"{name:28s} R@1 {res['recall@1']:.3f}  R@10 {res['recall@10']:.3f}  MRR@10 {res['mrr@10']:.3f}  "
          f"DomP@10 {res['domain_precision@10']:.3f}")
    return res


def encode(model, texts, prefix='', bs=256):
    return model.encode([prefix + t for t in texts], batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=192)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--skip-legacy', action='store_true')
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    random.seed(0)
    rng = random.Random(0)
    papers = load_papers()
    docs = [doc_text(p) for p in papers]
    all_domains = [p['domain'] for p in papers]
    held = [i for i, p in enumerate(papers) if is_heldout(p['pmid'])]
    train_idx = [i for i in range(len(papers)) if i not in set(held)]
    print(f'{len(papers)} papers | train {len(train_idx)} | held-out {len(held)} | GPU {torch.cuda.get_device_name(0)}')

    anchors, positives = [], []
    for i in train_idx:
        anchors.append(QUERY_PREFIX + papers[i]['title']); positives.append(docs[i])
        mq = mesh_query(papers[i], rng)
        if mq:
            anchors.append(QUERY_PREFIX + mq); positives.append(docs[i])
    train_ds = Dataset.from_dict({'anchor': anchors, 'positive': positives}).shuffle(seed=0)
    print(f'{len(train_ds)} training pairs')

    queries = [papers[i]['title'] for i in held]
    q_domains = [papers[i]['domain'] for i in held]
    results = {}

    bm25 = BM25().fit(docs)
    results['bm25'] = evaluate('BM25', bm25.scores, queries, held, q_domains, all_domains)

    def dense_eval(name, model, prefix):
        D = torch.tensor(encode(model, docs), device='cuda')
        Q = encode(model, queries, prefix)
        qmap = {q: i for i, q in enumerate(queries)}
        return evaluate(name, lambda q: (D @ torch.tensor(Q[qmap[q]], device='cuda')).cpu().numpy(),
                        queries, held, q_domains, all_domains), D, Q, qmap

    if not args.skip_legacy:
        legacy = SentenceTransformer(LEGACY_MODEL, device='cuda')
        results['v1_all_mpnet_base_v2'], *_ = dense_eval('v1 all-mpnet-base-v2 (110M)', legacy, '')
        del legacy

    base = SentenceTransformer(BASE_MODEL, device='cuda')
    results['bge_small_base'], *_ = dense_eval('bge-small base (33M)', base, QUERY_PREFIX)

    t0 = time.time()
    loss = MultipleNegativesRankingLoss(base)
    targs = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(WORK, 'st_runs'), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, learning_rate=args.lr, warmup_ratio=0.1, bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=50, save_strategy='no', report_to='none', seed=0,
    )
    SentenceTransformerTrainer(model=base, args=targs, train_dataset=train_ds, loss=loss).train()
    train_s = time.time() - t0
    base.save(MODEL_OUT)
    print(f'fine-tuned in {train_s:.0f}s -> {MODEL_OUT}')

    results['bge_small_finetuned'], D, Q, qmap = dense_eval('bge-small fine-tuned (33M)', base, QUERY_PREFIX)
    results['hybrid_bm25_finetuned'] = evaluate(
        'hybrid BM25 + fine-tuned (RRF)',
        lambda q: rrf(bm25.scores(q), (D @ torch.tensor(Q[qmap[q]], device='cuda')).cpu().numpy()),
        queries, held, q_domains, all_domains)

    results['_meta'] = {'papers': len(papers), 'heldout_queries': len(held), 'train_pairs': len(train_ds),
                        'epochs': args.epochs, 'batch': args.batch, 'train_seconds': round(train_s, 1),
                        'base_model': BASE_MODEL, 'gpu': torch.cuda.get_device_name(0)}
    with open(os.path.join(WORK, 'retrieval_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
