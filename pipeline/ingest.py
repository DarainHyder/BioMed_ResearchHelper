"""
Year-stratified PubMed ingestion.

v1 asked PubMed for the newest N papers per domain, so almost the whole corpus came from
2023-2024 and "trends" were meaningless. Here every (domain, year) cell is sampled
separately by relevance, which gives a balanced corpus from 2014 to today.

  python pipeline/ingest.py --per-year 55            # ~24 domains x 13 years x 55
Output: pipeline/work/raw_papers.jsonl
"""
import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests

EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work')
TOOL = 'bioatlas_research_pipeline'

# 24 domains spanning molecular biology, clinical medicine, public health and digital health
DOMAINS = {
    'CRISPR & Gene Editing': 'CRISPR gene editing therapy',
    'mRNA Vaccines': 'mRNA vaccine',
    'CAR-T & Cell Therapy': 'CAR-T cell therapy',
    'Cancer Immunotherapy': 'immune checkpoint inhibitor cancer immunotherapy',
    'Liquid Biopsy': 'liquid biopsy circulating tumor DNA',
    'Precision Oncology': 'precision oncology targeted therapy genomic profiling',
    'Gut Microbiome': 'gut microbiome',
    'Antimicrobial Resistance': 'antimicrobial resistance',
    'Infectious Disease Epidemiology': 'emerging infectious disease outbreak epidemiology',
    "Alzheimer's Disease": 'Alzheimer disease biomarkers',
    "Parkinson's & Neuroinflammation": 'Parkinson disease neuroinflammation',
    'Mental Health': 'depression treatment randomized trial',
    'Cardiovascular Medicine': 'cardiovascular disease prevention',
    'Diabetes & Obesity': 'GLP-1 receptor agonist obesity diabetes',
    'Autoimmunity & Immunometabolism': 'immunometabolism autoimmune disease',
    'Epigenetics & Aging': 'epigenetic clock aging',
    'Stem Cells & Regeneration': 'stem cell regenerative medicine',
    'Organoids': 'organoids disease modeling',
    'Gene Therapy': 'AAV gene therapy rare disease',
    'Single-Cell Genomics': 'single-cell RNA sequencing',
    'Protein Structure & Design': 'protein structure prediction design',
    'Nanomedicine': 'nanoparticle drug delivery',
    'AI in Medical Imaging': 'deep learning medical imaging',
    'Digital Health & Wearables': 'wearable sensors digital health',
}


def esearch(session, term, year, retmax):
    params = {'db': 'pubmed', 'term': f'({term}) AND hasabstract AND {year}[dp]', 'retmax': retmax,
              'sort': 'relevance', 'retmode': 'json', 'tool': TOOL}
    for attempt in range(5):
        r = session.get(EUTILS + 'esearch.fcgi', params=params, timeout=60)
        if r.status_code == 200:
            return r.json()['esearchresult']['idlist']
        time.sleep(2 ** attempt)
    r.raise_for_status()


def efetch(session, pmids):
    for attempt in range(5):
        r = session.post(EUTILS + 'efetch.fcgi', data={'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'xml', 'tool': TOOL},
                         timeout=120)
        if r.status_code == 200:
            return r.content
        time.sleep(2 ** attempt)
    r.raise_for_status()


def text_of(el):
    return ''.join(el.itertext()).strip() if el is not None else ''


MONTHS = {m: i for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 1)}


def parse(xml_bytes):
    out = []
    for art in ET.fromstring(xml_bytes).findall('.//PubmedArticle'):
        pmid = text_of(art.find('.//PMID'))
        title = text_of(art.find('.//ArticleTitle'))
        parts = []
        for ab in art.findall('.//Abstract/AbstractText'):
            t = text_of(ab)
            if t:
                label = ab.get('Label')
                parts.append(f'{label.title()}: {t}' if label else t)
        abstract = ' '.join(parts)
        if not pmid or not title or len(abstract) < 200:
            continue
        pd = art.find('.//JournalIssue/PubDate')
        year = text_of(pd.find('Year')) if pd is not None else ''
        if not year and pd is not None:
            year = text_of(pd.find('MedlineDate'))[:4]
        month = text_of(pd.find('Month')) if pd is not None else ''
        month = MONTHS.get(month[:3].lower(), int(month) if month.isdigit() else 0)
        authors = []
        for a in art.findall('.//AuthorList/Author'):
            last, fore = text_of(a.find('LastName')), text_of(a.find('ForeName'))
            if last:
                authors.append(f'{fore} {last}'.strip())
        doi = ''
        for aid in art.findall('.//ArticleIdList/ArticleId'):
            if aid.get('IdType') == 'doi':
                doi = text_of(aid)
        out.append({
            'pmid': pmid, 'title': title, 'abstract': abstract, 'authors': authors,
            'journal': text_of(art.find('.//Journal/Title')),
            'year': int(year) if year.isdigit() else None, 'month': month or None,
            'mesh': [text_of(m) for m in art.findall('.//MeshHeading/DescriptorName')],
            'keywords': [text_of(k) for k in art.findall('.//KeywordList/Keyword') if text_of(k)],
            'pub_types': [text_of(p) for p in art.findall('.//PublicationTypeList/PublicationType')],
            'doi': doi,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-year', type=int, default=55)
    ap.add_argument('--from-year', type=int, default=2014)
    ap.add_argument('--to-year', type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    session = requests.Session()

    # 1) esearch every (domain, year) cell; NCBI allows 3 req/s without an API key
    cells = [(d, q, y) for d, q in DOMAINS.items() for y in range(args.from_year, args.to_year + 1)]
    domain_of = {}
    for i, (domain, query, year) in enumerate(cells):
        for pmid in esearch(session, query, year, args.per_year):
            domain_of.setdefault(pmid, (domain, year))
        time.sleep(0.34)
        if i % 24 == 0:
            print(f'esearch {i + 1}/{len(cells)}  unique pmids so far: {len(domain_of)}', flush=True)

    # 2) efetch in batches of 200 (sequential to respect the rate limit)
    pmids = list(domain_of)
    papers = []
    for i in range(0, len(pmids), 200):
        papers += parse(efetch(session, pmids[i:i + 200]))
        time.sleep(0.34)
        print(f'efetch {min(i + 200, len(pmids))}/{len(pmids)}  parsed {len(papers)}', flush=True)

    with open(os.path.join(WORK, 'raw_papers.jsonl'), 'w') as f:
        for p in papers:
            p['domain'] = domain_of[p['pmid']][0]
            if p['year'] is None:
                p['year'] = domain_of[p['pmid']][1]
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f'wrote {len(papers)} papers')


if __name__ == '__main__':
    main()
