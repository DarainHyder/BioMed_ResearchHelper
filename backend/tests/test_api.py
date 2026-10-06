"""API tests against the real built artifacts (run after pipeline/build.py)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_stats(client):
    assert client.get('/api/health').json()['status'] == 'ok'
    s = client.get('/api/stats').json()
    assert s['papers'] > 10000 and len(s['domains']) == 24 and s['topics'] > 10


@pytest.mark.parametrize('mode', ['hybrid', 'semantic', 'keyword'])
def test_search_modes(client, mode):
    r = client.get('/api/search', params={'q': 'CAR-T cytokine release syndrome', 'mode': mode, 'k': 10}).json()
    assert r['results'], mode
    assert any('car' in x['title'].lower() or 'cytokine' in x['title'].lower() for x in r['results'][:5])


def test_search_filters(client):
    r = client.get('/api/search', params={'q': 'vaccine immunogenicity', 'domain': ['mRNA Vaccines'],
                                          'year_from': 2020, 'year_to': 2022}).json()
    assert r['results']
    assert all(x['domain'] == 'mRNA Vaccines' and 2020 <= x['year'] <= 2022 for x in r['results'])
    assert r['facets']['domains'] and r['facets']['years']


def test_semantic_matches_paraphrase(client):
    # no shared keywords with typical titles: semantic search must still find the field
    r = client.get('/api/search', params={'q': 'bugs that antibiotics can no longer kill', 'mode': 'semantic'}).json()
    assert sum(x['domain'] == 'Antimicrobial Resistance' for x in r['results'][:10]) >= 5


def test_paper_and_similar(client):
    pmid = client.get('/api/search', params={'q': 'organoids'}).json()['results'][0]['pmid']
    p = client.get(f'/api/papers/{pmid}').json()
    assert p['abstract'] and len(p['similar']) == 6 and pmid not in {s['pmid'] for s in p['similar']}
    assert client.get('/api/papers/0').status_code == 404


def test_brief(client):
    b = client.post('/api/brief', json={'question': 'Do GLP-1 receptor agonists reduce cardiovascular events?'}).json()
    assert 3 <= len(b['summary']) <= 6 and len(b['papers']) == 8
    assert all(1 <= s['cite'] <= 8 for s in b['summary'])
    assert b['evidence']['year_span']
    assert client.post('/api/brief', json={'question': 'x'}).status_code == 422


def test_topics_and_trends(client):
    t = client.get('/api/topics', params={'sort': 'growth'}).json()['topics']
    assert t[0]['growth'] >= t[-1]['growth']
    d = client.get(f"/api/topics/{t[0]['id']}").json()
    assert d['papers'] and d['keywords']
    tr = client.get('/api/trends').json()
    assert len(tr['pubmed_counts']) == 24 and tr['emerging_mesh']


def test_cors(client):
    r = client.get('/api/health', headers={'Origin': 'https://bio-med-research-helper-yzop.vercel.app'})
    assert r.headers['access-control-allow-origin'] == 'https://bio-med-research-helper-yzop.vercel.app'
    r = client.get('/api/health', headers={'Origin': 'https://evil.example.com'})
    assert 'access-control-allow-origin' not in r.headers
