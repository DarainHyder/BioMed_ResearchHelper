---
title: BioAtlas API
emoji: 🧬
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BioAtlas | Biomedical Research Intelligence

A living map of biomedical research: hybrid semantic search, cited evidence briefs, unsupervised topic
discovery and real publication trends over **14,693 PubMed studies from 24 fields, 2014 to 2026**.

- **Live site:** https://bio-med-research-helper-yzop.vercel.app
- **API:** https://sawabedarain-biomed-ai-backend.hf.space/docs

> [!NOTE]
> **Portfolio project, not a production service or medical advice.** The API runs on Hugging Face's free
> CPU tier, so the first request after a quiet period can take about 30 seconds while it wakes up. Evidence
> briefs are extractive by default; set a `GEMINI_API_KEY` secret to add an optional grounded AI synthesis.

## What it does

| | |
|---|---|
| **Hybrid search** | Fine-tuned dense retrieval fused with BM25 (reciprocal rank fusion). Filter by field and year; facets show where results cluster. |
| **Evidence briefs** | Ask a question; get the most relevant sentences across the top studies (MMR selection, every sentence cited), quantitative findings, and an evidence profile (study designs, years, concepts). |
| **Atlas** | Interactive map of every study, positioned by meaning (UMAP). Light up any query on the map. |
| **Topics** | Research fronts discovered without labels (UMAP + HDBSCAN), named with class TF-IDF, ranked by growth. |
| **Trends** | True PubMed publication volumes per field, rising MeSH concepts, top journals. |

The landing page opens with a scroll-driven "explode" animation built from the real data: every dot is a
paper, packed into a sphere by field, bursting apart and settling into its true position on the map.

## Retrieval benchmark

Held-out evaluation: each of 1,514 unseen paper titles is a query against all 14,693 abstracts; the target is its own paper.

| Model | Params | Recall@1 | Recall@10 | MRR@10 |
|---|---|---|---|---|
| BM25 (keywords) | - | 0.931 | 0.985 | 0.952 |
| all-mpnet-base-v2 (v1 model) | 110M | 0.798 | 0.959 | 0.855 |
| bge-small, off the shelf | 33M | 0.913 | 0.989 | 0.941 |
| bge-small, fine-tuned here | 33M | 0.964 | 0.997 | 0.977 |
| **Hybrid: BM25 + fine-tuned (production)** | 33M | 0.967 | 0.995 | 0.978 |

Fine-tuning: 26,257 title/MeSH-query to abstract pairs, MultipleNegativesRankingLoss, 3 epochs, 134 s on an RTX 5090. The deployed encoder is int8 per-channel ONNX (34 MB) and keeps Recall@1 at 0.961 vs 0.964 in fp32.

## Architecture

```
pipeline/ (offline, GPU)                        backend/ (Hugging Face Space, CPU)        frontend/ (Vercel)
  ingest.py   year-stratified PubMed sample  ->   FastAPI                                  React + Vite + Tailwind
  counts.py   true PubMed volumes per field       int8 ONNX encoder + fast tokenizer       Lenis smooth scroll
  train_embed fine-tune bge-small (MNRL)          dense (numpy) + BM25 (scipy sparse)      canvas explode scene + Atlas
  build.py    vectors, BM25, topics, map,         RRF fusion, MMR briefs, topics, trends   static map.bin (no backend
              ONNX export + int8 quantisation                                             needed for the hero)
```

Why it is efficient: v1 loaded BART-large, BERTopic and a 110M-parameter encoder at startup on a free CPU.
v2 serves a 33M-parameter encoder as int8 ONNX (no PyTorch in the image), precomputes topics and the map
offline, and answers searches in milliseconds.

## Run locally

```bash
conda env create -f environment.yml && conda activate biomed

# Rebuild the data and models (about 20 minutes with a GPU)...
bash pipeline/run_all.sh python
# ...or download the published artifacts from the Space
huggingface-cli download sawabedarain/biomed-ai-backend --repo-type space --include "artifacts/*" --local-dir backend

cd backend && uvicorn app.main:app --port 7860      # API on http://localhost:7860/docs
pytest tests -q

cd frontend && npm install && npm run dev           # site on http://localhost:5173
```

## Deploy

- **Backend:** the root `Dockerfile` builds the Space; `backend/artifacts/` is uploaded alongside the code.
- **Frontend:** Vercel project with root directory `frontend/`; `VITE_API_URL` points to the Space
  (defaults to it when unset).

Data: abstracts and metadata from NCBI PubMed via E-utilities. Abstracts remain the property of their publishers.
