import json
import re
import unicodedata
from typing import List, Dict
from pathlib import Path
import logging
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from config import Config
from src.data_ingestion import PubMedIngester

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        self.min_text_length = 50
        self.max_text_length = 10000
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'\[[\d\s,-]+\]', '', text)  # Remove citation numbers
        text = re.sub(r'\([\d\s,-]+\)', '', text)  # Remove parenthetical numbers
        text = re.sub(r'©.*?\d{4}', '', text)      # Remove copyright notices
        text = re.sub(r'doi:.*?\s', '', text)      # Remove DOI strings
        
        # Clean up HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        # Remove extra spaces and trim
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        
        for sentence in sentences:
            cleaned = self.clean_text(sentence)
            if len(cleaned) > 20:  # Minimum sentence length
                cleaned_sentences.append(cleaned)
                
        return cleaned_sentences
    
    def create_chunks(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """Create overlapping chunks for long texts"""
        sentences = self.extract_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (keep last sentence)
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_length = len(current_chunk[0].split()) if current_chunk else 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def validate_paper(self, paper: Dict) -> bool:
        """Validate if paper meets quality criteria"""
        # Check if we have essential fields
        if not paper.get('pmid'):
            return False
            
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        # Must have either title or abstract
        if not title and not abstract:
            return False
            
        # Check text length
        text_content = f"{title} {abstract}".strip()
        if len(text_content) < self.min_text_length:
            return False
            
        if len(text_content) > self.max_text_length:
            return False
            
        return True
    
    def process_paper(self, paper: Dict) -> Dict:
        """Process a single paper"""
        # Clean title and abstract
        title = self.clean_text(paper.get('title', ''))
        abstract = self.clean_text(paper.get('abstract', ''))
        
        # Create combined text content
        text_content = f"{title}. {abstract}".strip()
        if text_content.endswith('.'):
            text_content = text_content[:-1]
        
        # Extract sentences
        sentences = self.extract_sentences(text_content)
        
        # Create chunks if needed
        chunks = self.create_chunks(text_content)
        
        # Process metadata
        authors = paper.get('authors', [])
        mesh_terms = paper.get('mesh_terms', [])
        
        # Clean journal name
        journal = self.clean_text(paper.get('journal', ''))
        
        processed_paper = {
            'pmid': paper['pmid'],
            'title': title,
            'abstract': abstract,
            'text_content': text_content,
            'sentences': sentences,
            'chunks': chunks,
            'authors': authors,
            'journal': journal,
            'pub_date': paper.get('pub_date', 'Unknown'),
            'mesh_terms': mesh_terms,
            'word_count': len(text_content.split()),
            'sentence_count': len(sentences),
            'chunk_count': len(chunks)
        }
        
        return processed_paper
    
    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """Process multiple papers"""
        logger.info(f"Processing {len(papers)} papers")
        
        processed_papers = []
        skipped_count = 0
        
        for paper in tqdm(papers, desc="Processing papers"):
            try:
                # Validate paper
                if not self.validate_paper(paper):
                    skipped_count += 1
                    continue
                
                # Process paper
                processed_paper = self.process_paper(paper)
                processed_papers.append(processed_paper)
                
            except Exception as e:
                logger.warning(f"Error processing paper {paper.get('pmid', 'unknown')}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Processed {len(processed_papers)} papers, skipped {skipped_count}")
        return processed_papers
    
    def save_processed_papers(self, papers: List[Dict], filepath: Path):
        """Save processed papers"""
        logger.info(f"Saving {len(papers)} processed papers to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    
    def load_processed_papers(self, filepath: Path) -> List[Dict]:
        """Load processed papers"""
        papers = []
        
        if not filepath.exists():
            return papers
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line.strip()))
                
        return papers
    
    def get_statistics(self, papers: List[Dict]) -> Dict:
        """Get statistics about processed papers"""
        if not papers:
            return {}
            
        total_papers = len(papers)
        total_words = sum(paper.get('word_count', 0) for paper in papers)
        total_sentences = sum(paper.get('sentence_count', 0) for paper in papers)
        
        # Get publication years
        pub_years = []
        for paper in papers:
            pub_date = paper.get('pub_date', '')
            if pub_date and pub_date != 'Unknown':
                try:
                    year = int(pub_date.split('-')[0])
                    pub_years.append(year)
                except:
                    pass
        
        # Get journal distribution
        journals = {}
        for paper in papers:
            journal = paper.get('journal', 'Unknown')
            journals[journal] = journals.get(journal, 0) + 1
        
        # Get MeSH term distribution
        mesh_count = {}
        for paper in papers:
            for term in paper.get('mesh_terms', []):
                mesh_count[term] = mesh_count.get(term, 0) + 1
        
        stats = {
            'total_papers': total_papers,
            'total_words': total_words,
            'total_sentences': total_sentences,
            'avg_words_per_paper': total_words / total_papers if total_papers > 0 else 0,
            'avg_sentences_per_paper': total_sentences / total_papers if total_papers > 0 else 0,
            'year_range': (min(pub_years), max(pub_years)) if pub_years else None,
            'top_journals': sorted(journals.items(), key=lambda x: x[1], reverse=True)[:10],
            'top_mesh_terms': sorted(mesh_count.items(), key=lambda x: x[1], reverse=True)[:20]
        }
        
        return stats
    
    def preprocess_data(self) -> List[Dict]:
        """Main preprocessing method"""
        # Check if processed data already exists
        if Config.PROCESSED_DATA_FILE.exists():
            logger.info(f"Loading existing processed data from {Config.PROCESSED_DATA_FILE}")
            return self.load_processed_papers(Config.PROCESSED_DATA_FILE)
        
        # Load raw data
        ingester = PubMedIngester()
        raw_papers = ingester.load_papers(Config.RAW_DATA_FILE)
        
        if not raw_papers:
            logger.error("No raw papers found! Run data ingestion first.")
            return []
        
        # Process papers
        processed_papers = self.process_papers(raw_papers)
        
        # Save processed papers
        self.save_processed_papers(processed_papers, Config.PROCESSED_DATA_FILE)
        
        # Print statistics
        stats = self.get_statistics(processed_papers)
        logger.info(f"Processing complete! Statistics: {stats}")
        
        return processed_papers

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    papers = preprocessor.preprocess_data()
    logger.info(f"Preprocessed {len(papers)} papers successfully!")