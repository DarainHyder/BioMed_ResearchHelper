import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # NCBI API configuration
    NCBI_EMAIL = os.getenv('NCBI_EMAIL', '')
    NCBI_API_KEY = os.getenv('NCBI_API_KEY', '')
    
    if not NCBI_EMAIL:
        raise ValueError("NCBI_EMAIL is required in .env file")
    
    # Research configuration
    _domains_str = os.getenv('RESEARCH_DOMAINS', '')
    if _domains_str:
        RESEARCH_DOMAINS = [d.strip() for d in _domains_str.split(',') if d.strip()]
    else:
        RESEARCH_DOMAINS = [os.getenv('RESEARCH_DOMAIN', 'biomedical research')]
        
    PAPERS_PER_DOMAIN = int(os.getenv('PAPERS_PER_DOMAIN', '500'))
    
    # Legacy fallbacks for compatibility with older scripts
    RESEARCH_DOMAIN = RESEARCH_DOMAINS[0] if RESEARCH_DOMAINS else ''
    MAX_PAPERS = int(os.getenv('MAX_PAPERS', str(len(RESEARCH_DOMAINS) * PAPERS_PER_DOMAIN)))
    
    DATE_FROM = os.getenv('DATE_FROM', '2018/01/01')
    DATE_TO = os.getenv('DATE_TO', '2024/12/31')
    
    # Model configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
    SUMMARIZATION_MODEL = os.getenv('SUMMARIZATION_MODEL', 'facebook/bart-large-cnn')
    
    # Directory paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
    MODELS_DIR = Path(os.getenv('MODELS_DIR', './models'))
    CACHE_DIR = Path(os.getenv('CACHE_DIR', './cache'))
    
    # Create directories
    for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # File paths
    RAW_DATA_FILE = DATA_DIR / 'raw_papers.jsonl'
    PROCESSED_DATA_FILE = DATA_DIR / 'processed_papers.jsonl'
    EMBEDDINGS_FILE = DATA_DIR / 'embeddings.npy'
    FAISS_INDEX_FILE = DATA_DIR / 'faiss_index.bin'
    INDEX_MAPPING_FILE = DATA_DIR / 'index_mapping.json'
    TOPICS_FILE = DATA_DIR / 'topics.json'
    SUMMARIES_FILE = DATA_DIR / 'summaries.jsonl'
    
    # API configuration
    API_HOST = '127.0.0.1'
    API_PORT = 8000
    
    # Streamlit configuration
    STREAMLIT_HOST = '127.0.0.1'
    STREAMLIT_PORT = 8501
    
    # Processing parameters
    CHUNK_SIZE = 100
    TOP_K_RESULTS = 10
    MAX_SUMMARY_LENGTH = 150
    MIN_SUMMARY_LENGTH = 50
    
    # Topic modeling parameters
    MIN_TOPIC_SIZE = 10
    N_NEIGHBORS = 15
    MIN_DIST = 0.0