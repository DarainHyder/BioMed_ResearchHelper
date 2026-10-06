"""
Real PubMed publication volume per domain and year (esearch count only, no records fetched).
The corpus is sampled evenly per (domain, year), so absolute growth must come from here.
Output: pipeline/work/pubmed_counts.json
"""
import json
import os
import time

import requests

from ingest import DOMAINS, EUTILS, TOOL, WORK


def main(from_year=2014, to_year=2026):
    s = requests.Session()
    out = {}
    for domain, query in DOMAINS.items():
        out[domain] = {}
        for year in range(from_year, to_year + 1):
            for attempt in range(5):
                r = s.get(EUTILS + 'esearch.fcgi', params={'db': 'pubmed', 'term': f'({query}) AND {year}[dp]',
                                                          'rettype': 'count', 'retmode': 'json', 'tool': TOOL}, timeout=60)
                if r.status_code == 200:
                    break
                time.sleep(2 ** attempt)
            out[domain][str(year)] = int(r.json()['esearchresult']['count'])
            time.sleep(0.34)
        print(domain, out[domain], flush=True)
    with open(os.path.join(WORK, 'pubmed_counts.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
