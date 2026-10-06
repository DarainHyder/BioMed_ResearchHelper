import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.environ.get('ARTIFACTS_DIR', os.path.join(BASE_DIR, 'artifacts'))

CORS_ORIGINS = [o.strip() for o in os.environ.get(
    'CORS_ORIGINS',
    'http://localhost:5173,http://localhost:4173,https://bio-med-research-helper-yzop.vercel.app,'
    'https://bio-med-research-helper.vercel.app',
).split(',') if o.strip()]
CORS_ORIGIN_REGEX = os.environ.get('CORS_ORIGIN_REGEX', r'https://bio-med-research-helper.*\.vercel\.app')

QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

# Optional LLM synthesis for research briefs (extractive briefs always work without it)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')

MAX_CONCURRENT_BRIEFS = int(os.environ.get('MAX_CONCURRENT_BRIEFS', 2))
