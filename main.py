#!/usr/bin/env python3
"""
Biomedical Research Assistant - Main Execution Script
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from src.data_ingestion import PubMedIngester
from src.preprocessing import TextPreprocessor
from src.embeddings import SemanticSearchEngine
from src.summarization import SummarizationEngine
from src.topic_modeling import TopicModelingEngine
from src.api import run_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_data_pipeline():
    """Set up the complete data processing pipeline"""
    logger.info("=== Starting Data Pipeline Setup ===")
    
    try:
        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        ingester = PubMedIngester()
        papers = ingester.ingest_data()
        
        if not papers:
            logger.error("No papers were ingested. Check your configuration.")
            return False
        
        logger.info(f"Successfully ingested {len(papers)} papers")
        
        # Step 2: Text Preprocessing
        logger.info("Step 2: Text Preprocessing")
        preprocessor = TextPreprocessor()
        processed_papers = preprocessor.preprocess_data()
        
        if not processed_papers:
            logger.error("No papers were processed. Check preprocessing step.")
            return False
        
        logger.info(f"Successfully processed {len(processed_papers)} papers")
        
        # Step 3: Create Embeddings and Search Index
        logger.info("Step 3: Creating Embeddings and Search Index")
        search_engine = SemanticSearchEngine()
        
        if not search_engine.initialize():
            logger.error("Failed to initialize search engine")
            return False
        
        logger.info("Search engine initialized successfully")
        
        # Step 4: Generate Summaries
        logger.info("Step 4: Generating Summaries")
        summarization_engine = SummarizationEngine()
        summarization_engine.initialize(search_engine)
        summaries = summarization_engine.create_paper_summaries()
        
        logger.info(f"Generated summaries for {len(summaries)} papers")
        
        # Step 5: Topic Modeling and Trend Analysis
        logger.info("Step 5: Topic Modeling and Trend Analysis")
        topic_engine = TopicModelingEngine()
        
        if not topic_engine.initialize_topic_modeling():
            logger.error("Failed to initialize topic modeling")
            return False
        
        logger.info("Topic modeling initialized successfully")
        
        # Print final statistics
        stats = search_engine.get_stats()
        logger.info("=== Pipeline Setup Complete ===")
        logger.info(f"Total papers: {stats.get('num_papers', 0)}")
        logger.info(f"Embeddings created: {stats.get('num_embeddings', 0)}")
        logger.info(f"Model used: {stats.get('model_name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data pipeline setup: {e}")
        return False

def run_search_demo():
    """Run a search demonstration"""
    logger.info("=== Running Search Demo ===")
    
    try:
        # Initialize search engine
        search_engine = SemanticSearchEngine()
        if not search_engine.initialize():
            logger.error("Failed to initialize search engine")
            return
        
        # Demo queries
        demo_queries = [
            "COVID-19 vaccine efficacy",
            "cancer immunotherapy",
            "diabetes treatment",
            "alzheimer disease biomarkers"
        ]
        
        for query in demo_queries:
            logger.info(f"\nSearching for: '{query}'")
            results = search_engine.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')[:80] + "..."
                score = result.get('similarity_score', 0)
                logger.info(f"  {i}. {title} (Score: {score:.3f})")
    
    except Exception as e:
        logger.error(f"Error in search demo: {e}")

def run_summarization_demo():
    """Run a summarization demonstration"""
    logger.info("=== Running Summarization Demo ===")
    
    try:
        # Initialize engines
        search_engine = SemanticSearchEngine()
        if not search_engine.initialize():
            logger.error("Failed to initialize search engine")
            return
        
        summarization_engine = SummarizationEngine()
        summarization_engine.initialize(search_engine)
        
        # Demo query
        query = "COVID-19 vaccine effectiveness"
        logger.info(f"Generating summary for: '{query}'")
        
        summary_result = summarization_engine.summarize_search_results(query, top_k=5)
        
        if summary_result.get('summary'):
            logger.info(f"Summary: {summary_result['summary']}")
            logger.info(f"Based on {summary_result.get('num_papers', 0)} papers")
        else:
            logger.error("Failed to generate summary")
    
    except Exception as e:
        logger.error(f"Error in summarization demo: {e}")

def run_topic_analysis_demo():
    """Run topic analysis demonstration"""
    logger.info("=== Running Topic Analysis Demo ===")
    
    try:
        # Initialize topic engine
        topic_engine = TopicModelingEngine()
        if not topic_engine.initialize_topic_modeling():
            logger.error("Failed to initialize topic modeling")
            return
        
        # Get trending topics
        trending_topics = topic_engine.get_trending_topics(5)
        
        logger.info("Top 5 Trending Topics:")
        for i, topic in enumerate(trending_topics, 1):
            label = topic['topic_label']
            growth_score = topic['growth_score']
            logger.info(f"  {i}. {label} (Growth Score: {growth_score:.3f})")
        
        # General trends
        general_trends = topic_engine.analyze_general_trends()
        
        time_trends = general_trends.get('time_trends', {})
        if time_trends:
            growth_rate = time_trends.get('annual_growth_rate', 0)
            logger.info(f"\nAnnual publication growth rate: {growth_rate*100:.1f}%")
        
        journal_trends = general_trends.get('journal_trends', {})
        if journal_trends:
            top_journals = journal_trends.get('top_journals', [])[:3]
            logger.info("\nTop 3 Journals:")
            for journal in top_journals:
                logger.info(f"  - {journal['journal']}: {journal['count']} papers")
    
    except Exception as e:
        logger.error(f"Error in topic analysis demo: {e}")

def check_requirements():
    """Check if all requirements are met"""
    logger.info("Checking requirements...")
    
    # Check if .env file exists
    if not Path('.env').exists():
        logger.error("❌ .env file not found. Please copy .env.template to .env and configure it.")
        return False
    
    # Check NCBI email configuration
    if not Config.NCBI_EMAIL:
        logger.error("❌ NCBI_EMAIL not configured in .env file.")
        return False
    
    # Check data directories
    for directory in [Config.DATA_DIR, Config.MODELS_DIR, Config.CACHE_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
    
    logger.info("✅ All requirements met!")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Biomedical Research Assistant")
    parser.add_argument(
        'command', 
        choices=['setup', 'server', 'dashboard', 'demo-search', 'demo-summary', 'demo-topics', 'check'],
        help='Command to execute'
    )
    parser.add_argument('--host', default=Config.API_HOST, help='Host for server')
    parser.add_argument('--port', type=int, default=Config.API_PORT, help='Port for server')
    
    args = parser.parse_args()
    
    # Check requirements first
    if not check_requirements():
        sys.exit(1)
    
    if args.command == 'check':
        logger.info("✅ Configuration check passed!")
        
    elif args.command == 'setup':
        logger.info("Setting up the complete data processing pipeline...")
        if setup_data_pipeline():
            logger.info("✅ Setup completed successfully!")
            logger.info("You can now run 'python main.py server' to start the API")
            logger.info("Or run 'python main.py dashboard' to start the Streamlit app")
        else:
            logger.error("❌ Setup failed!")
            sys.exit(1)
    
    elif args.command == 'server':
        logger.info(f"Starting API server on {args.host}:{args.port}")
        # Update config with provided host/port
        Config.API_HOST = args.host
        Config.API_PORT = args.port
        run_server()
    
    elif args.command == 'dashboard':
        logger.info("Starting Streamlit dashboard...")
        import subprocess
        import os
        
        # Set environment variables for Streamlit
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent / "src")
        
        # Run Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            'src/streamlit_app.py',
            '--server.address', Config.STREAMLIT_HOST,
            '--server.port', str(Config.STREAMLIT_PORT),
            '--browser.gatherUsageStats', 'false'
        ]
        
        subprocess.run(cmd, env=env)
    
    elif args.command == 'demo-search':
        run_search_demo()
    
    elif args.command == 'demo-summary':
        run_summarization_demo()
    
    elif args.command == 'demo-topics':
        run_topic_analysis_demo()

if __name__ == "__main__":
    main()