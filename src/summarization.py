import json
from typing import List, Dict, Optional
from pathlib import Path
import logging
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import util
import torch
from tqdm import tqdm

from config import Config
from src.embeddings import SemanticSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self):
        self.model_name = Config.SUMMARIZATION_MODEL
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        self.max_length = Config.MAX_SUMMARY_LENGTH
        self.min_length = Config.MIN_SUMMARY_LENGTH
        
    def load_model(self):
        """Load the summarization model"""
        if self.summarizer is None:
            logger.info(f"Loading summarization model: {self.model_name}")
            
            try:
                # Try to load as pipeline first
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Summarization model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Fallback to a more basic model
                logger.info("Falling back to basic summarization model")
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1
                )
    
    def extractive_preprocessing(self, text: str, query: str = None, top_sentences: int = 6) -> str:
        """Extract most relevant sentences before abstractive summarization"""
        if not query:
            # If no query, return first N sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:top_sentences]) + '.'
        
        # Use sentence similarity to find most relevant sentences
        sentences = text.split('. ')
        if len(sentences) <= top_sentences:
            return text
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight model for sentence ranking
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode query and sentences
            query_embedding = model.encode([query])
            sentence_embeddings = model.encode(sentences)
            
            # Calculate similarities
            similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
            
            # Get top sentences
            top_indices = similarities.argsort(descending=True)[:top_sentences]
            top_sentences_list = [sentences[idx] for idx in sorted(top_indices)]
            
            return '. '.join(top_sentences_list) + '.'
            
        except Exception as e:
            logger.warning(f"Error in extractive preprocessing: {e}")
            # Fallback to first N sentences
            return '. '.join(sentences[:top_sentences]) + '.'
    
    def summarize_text(self, text: str, query: str = None) -> Dict:
        """Summarize a single text"""
        self.load_model()
        
        if not text or len(text.strip()) < 50:
            return {
                'summary': '',
                'error': 'Text too short for summarization'
            }
        
        try:
            # Preprocessing: extract relevant sentences
            processed_text = self.extractive_preprocessing(text, query)
            
            # Check text length for tokenizer limits
            if len(processed_text.split()) > 1000:
                # Truncate to fit model limits
                words = processed_text.split()
                processed_text = ' '.join(words[:1000])
            
            # Generate summary
            summary_result = self.summarizer(
                processed_text,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=False,
                truncation=True
            )
            
            summary = summary_result[0]['summary_text']
            
            # Post-process summary
            summary = self._post_process_summary(summary)
            
            return {
                'summary': summary,
                'original_length': len(text.split()),
                'summary_length': len(summary.split()),
                'compression_ratio': len(summary.split()) / len(text.split()) if text else 0
            }
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return {
                'summary': '',
                'error': str(e)
            }
    
    def _post_process_summary(self, summary: str) -> str:
        """Post-process the generated summary"""
        # Remove incomplete sentences at the end
        sentences = summary.split('. ')
        if len(sentences) > 1 and not sentences[-1].endswith('.'):
            sentences = sentences[:-1]
        
        summary = '. '.join(sentences)
        if not summary.endswith('.'):
            summary += '.'
        
        # Clean up common artifacts
        summary = summary.replace(' ..', '.')
        summary = summary.replace('..', '.')
        
        return summary.strip()
    
    def summarize_papers(self, papers: List[Dict], query: str = None) -> List[Dict]:
        """Summarize multiple papers"""
        logger.info(f"Summarizing {len(papers)} papers")
        
        summarized_papers = []
        
        for paper in tqdm(papers, desc="Summarizing papers"):
            text = paper.get('text_content', '')
            
            if not text:
                continue
            
            # Generate summary
            summary_result = self.summarize_text(text, query)
            
            # Add summary to paper data
            paper_with_summary = paper.copy()
            paper_with_summary.update(summary_result)
            
            summarized_papers.append(paper_with_summary)
        
        return summarized_papers
    
    def generate_multi_paper_summary(self, papers: List[Dict], topic: str = None) -> Dict:
        """Generate a summary across multiple papers"""
        if not papers:
            return {'summary': '', 'error': 'No papers provided'}
        
        # Collect all summaries or abstracts
        text_pieces = []
        pmids = []
        
        for paper in papers:
            if paper.get('summary'):
                text_pieces.append(paper['summary'])
            elif paper.get('abstract'):
                text_pieces.append(paper['abstract'])
            elif paper.get('text_content'):
                # Use first 200 words
                words = paper['text_content'].split()[:200]
                text_pieces.append(' '.join(words))
            
            pmids.append(paper['pmid'])
        
        if not text_pieces:
            return {'summary': '', 'error': 'No text content found'}
        
        # Combine texts
        combined_text = ' '.join(text_pieces)
        
        # Add topic context if provided
        if topic:
            combined_text = f"Topic: {topic}. Research findings: {combined_text}"
        
        # Summarize
        result = self.summarize_text(combined_text, topic)
        result['source_papers'] = pmids
        result['num_papers'] = len(papers)
        
        return result
    
    def save_summaries(self, papers: List[Dict], filepath: Path):
        """Save papers with summaries"""
        logger.info(f"Saving {len(papers)} papers with summaries to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    
    def load_summaries(self, filepath: Path) -> List[Dict]:
        """Load papers with summaries"""
        papers = []
        
        if not filepath.exists():
            return papers
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line.strip()))
        
        return papers
    
    def evaluate_summary(self, summary: str, reference: str) -> Dict:
        """Evaluate summary quality"""
        try:
            from rouge_score import rouge_scorer
            from bert_score import score
            
            # ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(reference, summary)
            
            # BERTScore
            P, R, F1 = score([summary], [reference], lang='en', verbose=False)
            
            return {
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure,
                'bertscore_f1': F1.item()
            }
            
        except ImportError:
            logger.warning("ROUGE or BERTScore not available for evaluation")
            return {}
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {}

class SummarizationEngine:
    """High-level interface for summarization"""
    
    def __init__(self):
        self.summarizer = TextSummarizer()
        self.search_engine = None
    
    def initialize(self, search_engine: SemanticSearchEngine = None):
        """Initialize the summarization engine"""
        logger.info("Initializing summarization engine")
        self.search_engine = search_engine
        self.summarizer.load_model()
    
    def summarize_search_results(self, query: str, top_k: int = 5) -> Dict:
        """Summarize search results for a query"""
        if not self.search_engine:
            logger.error("Search engine not available")
            return {}
        
        # Get search results
        results = self.search_engine.search(query, top_k)
        
        if not results:
            return {'summary': 'No relevant papers found', 'papers': []}
        
        # Generate multi-paper summary
        summary_result = self.summarizer.generate_multi_paper_summary(results, query)
        summary_result['papers'] = results
        summary_result['query'] = query
        
        return summary_result
    
    def summarize_paper(self, pmid: str) -> Dict:
        """Summarize a specific paper"""
        if not self.search_engine:
            logger.error("Search engine not available")
            return {}
        
        paper = self.search_engine.get_paper(pmid)
        if not paper:
            return {'error': f'Paper {pmid} not found'}
        
        summary_result = self.summarizer.summarize_text(paper.get('text_content', ''))
        summary_result['paper'] = paper
        
        return summary_result
    
    def create_paper_summaries(self) -> List[Dict]:
        """Create summaries for all papers"""
        # Check if summaries already exist
        if Config.SUMMARIES_FILE.exists():
            logger.info(f"Loading existing summaries from {Config.SUMMARIES_FILE}")
            return self.summarizer.load_summaries(Config.SUMMARIES_FILE)
        
        # Load papers
        from src.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        papers = preprocessor.load_processed_papers(Config.PROCESSED_DATA_FILE)
        
        if not papers:
            logger.error("No papers available for summarization")
            return []
        
        # Create summaries
        summarized_papers = self.summarizer.summarize_papers(papers)
        
        # Save summaries
        self.summarizer.save_summaries(summarized_papers, Config.SUMMARIES_FILE)
        
        return summarized_papers
    
    def get_trending_summaries(self, time_window: str = 'month') -> List[Dict]:
        """Get summaries of trending topics"""
        # This would integrate with topic modeling
        # For now, return recent papers
        if not self.search_engine:
            return []
        
        # This is a placeholder - would be implemented with topic modeling
        recent_papers = []
        return recent_papers

if __name__ == "__main__":
    # Test summarization
    engine = SummarizationEngine()
    engine.initialize()
    
    # Create summaries for all papers
    summaries = engine.create_paper_summaries()
    logger.info(f"Created summaries for {len(summaries)} papers")