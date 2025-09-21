import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from tqdm import tqdm

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PubMedIngester:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = Config.NCBI_EMAIL
        self.api_key = Config.NCBI_API_KEY
        self.session = requests.Session()
        
    def search_papers(self, query: str, max_results: int = 5000, 
                     date_from: str = None, date_to: str = None) -> List[str]:
        """Search for paper IDs using PubMed API"""
        logger.info(f"Searching for papers with query: {query}")
        
        # Build search parameters
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'email': self.email,
            'tool': 'biomedical_research_assistant'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
            
        if date_from and date_to:
            params['mindate'] = date_from
            params['maxdate'] = date_to
            params['datetype'] = 'pdat'
        
        # Make search request
        search_url = self.base_url + 'esearch.fcgi'
        response = self.session.get(search_url, params=params)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        pmids = [id_elem.text for id_elem in root.findall('.//Id')]
        
        logger.info(f"Found {len(pmids)} papers")
        return pmids
    
    def fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed information for papers"""
        logger.info(f"Fetching details for {len(pmids)} papers")
        
        papers = []
        batch_size = 100
        
        for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching paper details"):
            batch_pmids = pmids[i:i + batch_size]
            batch_papers = self._fetch_batch(batch_pmids)
            papers.extend(batch_papers)
            
            # Rate limiting
            time.sleep(0.1)
            
        logger.info(f"Successfully fetched {len(papers)} papers")
        return papers
    
    def _fetch_batch(self, pmids: List[str]) -> List[Dict]:
        """Fetch a batch of paper details"""
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'email': self.email,
            'tool': 'biomedical_research_assistant'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        fetch_url = self.base_url + 'efetch.fcgi'
        response = self.session.get(fetch_url, params=params)
        response.raise_for_status()
        
        return self._parse_pubmed_xml(response.content)
    
    def _parse_pubmed_xml(self, xml_content: bytes) -> List[Dict]:
        """Parse PubMed XML response"""
        papers = []
        root = ET.fromstring(xml_content)
        
        for article in root.findall('.//PubmedArticle'):
            try:
                paper = self._extract_paper_info(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue
                
        return papers
    
    def _extract_paper_info(self, article) -> Optional[Dict]:
        """Extract information from a single article"""
        try:
            # Basic info
            pmid = article.find('.//PMID').text
            
            # Title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Abstract
            abstract_parts = []
            abstract_elems = article.findall('.//AbstractText')
            for elem in abstract_elems:
                if elem.text:
                    label = elem.get('Label', '')
                    text = elem.text
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            
            abstract = " ".join(abstract_parts)
            
            # Authors
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            
            # Journal
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Publication date
            pub_date = self._extract_pub_date(article)
            
            # MeSH terms
            mesh_terms = []
            for mesh in article.findall('.//MeshHeading/DescriptorName'):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            # Skip if no title or abstract
            if not title and not abstract:
                return None
                
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'pub_date': pub_date,
                'mesh_terms': mesh_terms,
                'text_content': f"{title}. {abstract}".strip()
            }
            
        except Exception as e:
            logger.warning(f"Error extracting paper info: {e}")
            return None
    
    def _extract_pub_date(self, article) -> str:
        """Extract publication date"""
        try:
            # Try PubDate first
            pub_date = article.find('.//PubDate')
            if pub_date is not None:
                year = pub_date.find('Year')
                month = pub_date.find('Month')
                day = pub_date.find('Day')
                
                if year is not None:
                    date_str = year.text
                    if month is not None:
                        date_str += f"-{month.text}"
                        if day is not None:
                            date_str += f"-{day.text}"
                    return date_str
            
            # Try MedlineDate as fallback
            medline_date = article.find('.//MedlineDate')
            if medline_date is not None:
                return medline_date.text[:10]  # Take first 10 chars
                
            return "Unknown"
            
        except Exception:
            return "Unknown"
    
    def save_papers(self, papers: List[Dict], filepath: Path):
        """Save papers to JSONL file"""
        logger.info(f"Saving {len(papers)} papers to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    
    def load_papers(self, filepath: Path) -> List[Dict]:
        """Load papers from JSONL file"""
        papers = []
        
        if not filepath.exists():
            return papers
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line.strip()))
                
        return papers
    
    def ingest_data(self, query: str = None, max_results: int = None, 
                   date_from: str = None, date_to: str = None) -> List[Dict]:
        """Main ingestion method"""
        query = query or Config.RESEARCH_DOMAIN
        max_results = max_results or Config.MAX_PAPERS
        date_from = date_from or Config.DATE_FROM
        date_to = date_to or Config.DATE_TO
        
        # Check if data already exists
        if Config.RAW_DATA_FILE.exists():
            logger.info(f"Loading existing data from {Config.RAW_DATA_FILE}")
            return self.load_papers(Config.RAW_DATA_FILE)
        
        # Search for papers
        pmids = self.search_papers(query, max_results, date_from, date_to)
        
        if not pmids:
            logger.error("No papers found!")
            return []
        
        # Fetch paper details
        papers = self.fetch_paper_details(pmids)
        
        # Save raw data
        self.save_papers(papers, Config.RAW_DATA_FILE)
        
        return papers

if __name__ == "__main__":
    ingester = PubMedIngester()
    papers = ingester.ingest_data()
    logger.info(f"Ingested {len(papers)} papers successfully!")