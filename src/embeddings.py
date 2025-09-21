import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import Config
from src.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self):
        self.model_name = Config.EMBEDDING_MODEL
        self.model = None
        self.embeddings = None
        self.index = None
        self.papers = []
        self.id_to_index = {}
        self.index_to_id = {}
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings"""
        self.load_model()
        
        logger.info(f"Encoding {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def create_embeddings(self, papers: List[Dict]) -> np.ndarray:
        """Create embeddings for papers"""
        logger.info(f"Creating embeddings for {len(papers)} papers")
        
        # Extract texts for embedding
        texts = []
        valid_papers = []
        
        for paper in papers:
            text_content = paper.get('text_content', '')
            if text_content:
                texts.append(text_content)
                valid_papers.append(paper)
        
        if not texts:
            logger.error("No valid texts found for embedding")
            return np.array([])
        
        # Generate embeddings
        embeddings = self.encode_texts(texts)
        
        # Store papers and create mappings
        self.papers = valid_papers
        self.id_to_index = {paper['pmid']: i for i, paper in enumerate(valid_papers)}
        self.index_to_id = {i: paper['pmid'] for i, paper in enumerate(valid_papers)}
        
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, embeddings_file: Path):
        """Save embeddings to file"""
        logger.info(f"Saving embeddings to {embeddings_file}")
        np.save(embeddings_file, embeddings)
    
    def load_embeddings(self, embeddings_file: Path) -> np.ndarray:
        """Load embeddings from file"""
        if embeddings_file.exists():
            logger.info(f"Loading embeddings from {embeddings_file}")
            return np.load(embeddings_file)
        return np.array([])
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for similarity search"""
        logger.info("Building FAISS index")
        
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, index_file: Path):
        """Save FAISS index to file"""
        logger.info(f"Saving FAISS index to {index_file}")
        faiss.write_index(index, str(index_file))
    
    def load_index(self, index_file: Path) -> faiss.Index:
        """Load FAISS index from file"""
        if index_file.exists():
            logger.info(f"Loading FAISS index from {index_file}")
            return faiss.read_index(str(index_file))
        return None
    
    def save_index_mapping(self, mapping_file: Path):
        """Save index to ID mapping"""
        mapping_data = {
            'id_to_index': self.id_to_index,
            'index_to_id': self.index_to_id
        }
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f)
    
    def load_index_mapping(self, mapping_file: Path):
        """Load index to ID mapping"""
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
                self.id_to_index = mapping_data['id_to_index']
                self.index_to_id = {int(k): v for k, v in mapping_data['index_to_id'].items()}
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Perform semantic search"""
        if self.index is None or self.model is None:
            logger.error("Index or model not loaded")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Get results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.papers):
                paper = self.papers[idx]
                results.append((paper, float(score)))
        
        return results
    
    def find_similar_papers(self, pmid: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Find papers similar to a given paper"""
        if pmid not in self.id_to_index:
            logger.error(f"PMID {pmid} not found")
            return []
        
        paper_idx = self.id_to_index[pmid]
        
        # Get embedding for this paper
        paper_embedding = self.embeddings[paper_idx:paper_idx+1]
        
        # Search for similar papers
        scores, indices = self.index.search(paper_embedding.astype(np.float32), top_k + 1)
        
        # Skip the first result (the paper itself)
        results = []
        for score, idx in zip(scores[0][1:], indices[0][1:]):
            if idx < len(self.papers):
                paper = self.papers[idx]
                results.append((paper, float(score)))
        
        return results
    
    def get_paper_by_pmid(self, pmid: str) -> Dict:
        """Get paper by PMID"""
        if pmid in self.id_to_index:
            idx = self.id_to_index[pmid]
            return self.papers[idx]
        return None
    
    def cluster_embeddings(self, n_clusters: int = 50) -> np.ndarray:
        """Cluster embeddings using K-means"""
        from sklearn.cluster import KMeans
        
        logger.info(f"Clustering {len(self.embeddings)} embeddings into {n_clusters} clusters")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # Add cluster labels to papers
        for i, paper in enumerate(self.papers):
            paper['cluster_label'] = int(cluster_labels[i])
        
        return cluster_labels
    
    def setup_embeddings(self) -> bool:
        """Setup embeddings and index"""
        # Check if embeddings already exist
        if (Config.EMBEDDINGS_FILE.exists() and 
            Config.FAISS_INDEX_FILE.exists() and 
            Config.INDEX_MAPPING_FILE.exists()):
            
            logger.info("Loading existing embeddings and index")
            
            # Load embeddings
            self.embeddings = self.load_embeddings(Config.EMBEDDINGS_FILE)
            
            # Load index
            self.index = self.load_index(Config.FAISS_INDEX_FILE)
            
            # Load mapping
            self.load_index_mapping(Config.INDEX_MAPPING_FILE)
            
            # Load papers
            preprocessor = TextPreprocessor()
            self.papers = preprocessor.load_processed_papers(Config.PROCESSED_DATA_FILE)
            
            # Load model for search
            self.load_model()
            
            logger.info("Embeddings and index loaded successfully")
            return True
        
        # Create new embeddings
        logger.info("Creating new embeddings and index")
        
        # Load papers
        preprocessor = TextPreprocessor()
        papers = preprocessor.preprocess_data()
        
        if not papers:
            logger.error("No papers available for embedding")
            return False
        
        # Create embeddings
        embeddings = self.create_embeddings(papers)
        if embeddings.size == 0:
            logger.error("Failed to create embeddings")
            return False
        
        # Save embeddings
        self.embeddings = embeddings
        self.save_embeddings(embeddings, Config.EMBEDDINGS_FILE)
        
        # Build and save index
        self.index = self.build_faiss_index(embeddings)
        self.save_index(self.index, Config.FAISS_INDEX_FILE)
        
        # Save mapping
        self.save_index_mapping(Config.INDEX_MAPPING_FILE)
        
        logger.info("Embeddings and index created successfully")
        return True
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about embeddings"""
        if self.embeddings is None:
            return {}
        
        return {
            'num_embeddings': self.embeddings.shape[0],
            'embedding_dim': self.embeddings.shape[1],
            'model_name': self.model_name,
            'index_type': type(self.index).__name__ if self.index else None,
            'num_papers': len(self.papers)
        }

class SemanticSearchEngine:
    """High-level interface for semantic search"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.ready = False
    
    def initialize(self):
        """Initialize the search engine"""
        logger.info("Initializing semantic search engine")
        self.ready = self.embedding_manager.setup_embeddings()
        return self.ready
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for papers"""
        if not self.ready:
            logger.error("Search engine not initialized")
            return []
        
        top_k = top_k or Config.TOP_K_RESULTS
        
        results = self.embedding_manager.semantic_search(query, top_k)
        
        # Format results
        formatted_results = []
        for paper, score in results:
            result = paper.copy()
            result['similarity_score'] = score
            formatted_results.append(result)
        
        return formatted_results
    
    def find_similar(self, pmid: str, top_k: int = None) -> List[Dict]:
        """Find similar papers to a given paper"""
        if not self.ready:
            logger.error("Search engine not initialized")
            return []
        
        top_k = top_k or Config.TOP_K_RESULTS
        
        results = self.embedding_manager.find_similar_papers(pmid, top_k)
        
        # Format results
        formatted_results = []
        for paper, score in results:
            result = paper.copy()
            result['similarity_score'] = score
            formatted_results.append(result)
        
        return formatted_results
    
    def get_paper(self, pmid: str) -> Dict:
        """Get paper by PMID"""
        if not self.ready:
            return None
        
        return self.embedding_manager.get_paper_by_pmid(pmid)
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        return self.embedding_manager.get_embedding_stats()

if __name__ == "__main__":
    # Test the embedding system
    search_engine = SemanticSearchEngine()
    
    if search_engine.initialize():
        logger.info("Search engine initialized successfully")
        
        # Test search
        test_query = "covid-19 vaccine efficacy"
        results = search_engine.search(test_query, 5)
        
        logger.info(f"Search for '{test_query}' returned {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. {result['title']} (Score: {result['similarity_score']:.3f})")
    else:
        logger.error("Failed to initialize search engine")