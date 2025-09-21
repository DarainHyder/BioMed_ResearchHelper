import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import re

from config import Config
from src.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModeler:
    def __init__(self):
        self.model = None
        self.topics = None
        self.probabilities = None
        self.topic_info = None
        self.papers = []
        self.embeddings = None
        
    def create_model(self) -> BERTopic:
        """Create BERTopic model with biomedical-optimized parameters"""
        logger.info("Creating BERTopic model")
        
        # Vectorizer for better biomedical term extraction
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_features=1000
        )
        
        # Use KeyBERT for better topic representation
        representation_model = KeyBERTInspired()
        
        # Create BERTopic model
        topic_model = BERTopic(
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            min_topic_size=Config.MIN_TOPIC_SIZE,
            n_gram_range=(1, 2),
            calculate_probabilities=True,
            verbose=True
        )
        
        return topic_model
    
    def fit_topics(self, papers: List[Dict], embeddings: np.ndarray = None) -> BERTopic:
        """Fit topic model on papers"""
        logger.info(f"Fitting topic model on {len(papers)} papers")
        
        self.papers = papers
        self.embeddings = embeddings
        
        # Extract texts
        texts = []
        for paper in papers:
            text = paper.get('text_content', '')
            if text:
                texts.append(text)
        
        if not texts:
            logger.error("No texts available for topic modeling")
            return None
        
        # Create model
        self.model = self.create_model()
        
        # Fit model
        if embeddings is not None:
            logger.info("Using pre-computed embeddings")
            self.topics, self.probabilities = self.model.fit_transform(texts, embeddings)
        else:
            logger.info("Computing embeddings during fitting")
            self.topics, self.probabilities = self.model.fit_transform(texts)
        
        # Get topic info
        self.topic_info = self.model.get_topic_info()
        
        logger.info(f"Found {len(self.model.get_topics())} topics")
        return self.model
    
    def get_topic_trends(self, time_window: str = 'month') -> Dict:
        """Analyze topic trends over time"""
        logger.info(f"Analyzing topic trends with {time_window} window")
        
        if not self.model or not self.papers:
            logger.error("Model not fitted or no papers available")
            return {}
        
        # Create dataframe with papers and topics
        data = []
        for i, paper in enumerate(self.papers):
            if i < len(self.topics):
                pub_date = paper.get('pub_date', 'Unknown')
                topic_id = self.topics[i]
                
                # Parse date
                try:
                    if pub_date != 'Unknown':
                        date_obj = datetime.strptime(pub_date.split('-')[0], '%Y')
                    else:
                        continue
                except:
                    continue
                
                data.append({
                    'pmid': paper['pmid'],
                    'topic': topic_id,
                    'date': date_obj,
                    'year': date_obj.year,
                    'month': date_obj.month,
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', '')
                })
        
        if not data:
            logger.error("No valid date data for trend analysis")
            return {}
        
        df = pd.DataFrame(data)
        
        # Group by time window
        if time_window == 'month':
            df['time_period'] = df['date'].dt.to_period('M')
        elif time_window == 'year':
            df['time_period'] = df['date'].dt.to_period('Y')
        else:  # quarter
            df['time_period'] = df['date'].dt.to_period('Q')
        
        # Calculate topic counts per time period
        topic_trends = df.groupby(['time_period', 'topic']).size().reset_index(name='count')
        
        # Calculate growth rates
        trends_with_growth = []
        for topic_id in topic_trends['topic'].unique():
            if topic_id == -1:  # Skip outlier topic
                continue
                
            topic_data = topic_trends[topic_trends['topic'] == topic_id].sort_values('time_period')
            
            if len(topic_data) < 2:
                continue
            
            # Calculate growth rate
            recent_count = topic_data['count'].iloc[-1]
            previous_count = topic_data['count'].iloc[-2] if len(topic_data) > 1 else 1
            growth_rate = (recent_count - previous_count) / max(previous_count, 1)
            
            # Calculate average count for volume weighting
            avg_count = topic_data['count'].mean()
            
            # Growth score: growth rate weighted by volume
            growth_score = growth_rate * np.log(avg_count + 1)
            
            # Get topic keywords
            topic_words = self.model.get_topic(topic_id)
            topic_label = ', '.join([word for word, _ in topic_words[:3]])
            
            trends_with_growth.append({
                'topic_id': topic_id,
                'topic_label': topic_label,
                'growth_rate': growth_rate,
                'growth_score': growth_score,
                'recent_count': recent_count,
                'avg_count': avg_count,
                'time_series': topic_data.to_dict('records')
            })
        
        # Sort by growth score
        trends_with_growth.sort(key=lambda x: x['growth_score'], reverse=True)
        
        return {
            'trends': trends_with_growth,
            'time_window': time_window,
            'total_topics': len(trends_with_growth),
            'date_range': (df['date'].min().strftime('%Y-%m-%d'), 
                          df['date'].max().strftime('%Y-%m-%d'))
        }
    
    def get_topic_details(self, topic_id: int, num_papers: int = 10) -> Dict:
        """Get detailed information about a specific topic"""
        if not self.model:
            return {}
        
        # Get topic words
        topic_words = self.model.get_topic(topic_id)
        
        # Get representative documents
        topic_papers = []
        for i, paper in enumerate(self.papers):
            if i < len(self.topics) and self.topics[i] == topic_id:
                paper_with_prob = paper.copy()
                if i < len(self.probabilities):
                    paper_with_prob['topic_probability'] = float(self.probabilities[i][topic_id + 1])  # +1 because -1 is first
                topic_papers.append(paper_with_prob)
        
        # Sort by probability if available
        if topic_papers and 'topic_probability' in topic_papers[0]:
            topic_papers.sort(key=lambda x: x.get('topic_probability', 0), reverse=True)
        
        # Get top papers
        top_papers = topic_papers[:num_papers]
        
        # Extract MeSH terms from papers in this topic
        mesh_terms = []
        for paper in topic_papers:
            mesh_terms.extend(paper.get('mesh_terms', []))
        
        mesh_counter = Counter(mesh_terms)
        
        return {
            'topic_id': topic_id,
            'keywords': [{'word': word, 'score': float(score)} for word, score in topic_words],
            'num_papers': len(topic_papers),
            'top_papers': top_papers,
            'top_mesh_terms': [{'term': term, 'count': count} for term, count in mesh_counter.most_common(10)],
            'topic_label': ', '.join([word for word, _ in topic_words[:3]])
        }
    
    def visualize_topics(self) -> Dict:
        """Create topic visualizations"""
        if not self.model:
            return {}
        
        logger.info("Creating topic visualizations")
        
        try:
            # Topic visualization
            fig_topics = self.model.visualize_topics()
            
            # Topic hierarchy
            fig_hierarchy = self.model.visualize_hierarchy()
            
            # Topic heatmap
            fig_heatmap = self.model.visualize_heatmap()
            
            return {
                'topics_plot': fig_topics.to_html(),
                'hierarchy_plot': fig_hierarchy.to_html(),
                'heatmap_plot': fig_heatmap.to_html()
            }
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return {}
    
    def create_trend_visualizations(self, trends_data: Dict) -> Dict:
        """Create trend visualizations"""
        if not trends_data or not trends_data.get('trends'):
            return {}
        
        logger.info("Creating trend visualizations")
        
        # Prepare data for plotting
        plot_data = []
        for trend in trends_data['trends'][:10]:  # Top 10 topics
            for point in trend['time_series']:
                plot_data.append({
                    'topic': trend['topic_label'],
                    'time_period': str(point['time_period']),
                    'count': point['count'],
                    'growth_score': trend['growth_score']
                })
        
        if not plot_data:
            return {}
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create line plot for trends
        fig_trends = px.line(
            df_plot,
            x='time_period',
            y='count',
            color='topic',
            title='Topic Trends Over Time',
            labels={'count': 'Number of Papers', 'time_period': 'Time Period'}
        )
        fig_trends.update_layout(xaxis_tickangle=-45)
        
        # Create bar plot for growth scores
        growth_data = trends_data['trends'][:10]
        fig_growth = px.bar(
            x=[t['topic_label'] for t in growth_data],
            y=[t['growth_score'] for t in growth_data],
            title='Topic Growth Scores',
            labels={'x': 'Topic', 'y': 'Growth Score'}
        )
        fig_growth.update_layout(xaxis_tickangle=-45)
        
        return {
            'trends_plot': fig_trends.to_html(),
            'growth_plot': fig_growth.to_html()
        }
    
    def save_model(self, model_path: Path):
        """Save the trained model"""
        if self.model:
            logger.info(f"Saving topic model to {model_path}")
            self.model.save(str(model_path))
    
    def load_model(self, model_path: Path):
        """Load a trained model"""
        if model_path.exists():
            logger.info(f"Loading topic model from {model_path}")
            self.model = BERTopic.load(str(model_path))
            return True
        return False
    
    def save_topics_data(self, topics_data: Dict, filepath: Path):
        """Save topics analysis data"""
        logger.info(f"Saving topics data to {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(topics_data, f, ensure_ascii=False, indent=2, default=str)
    
    def load_topics_data(self, filepath: Path) -> Dict:
        """Load topics analysis data"""
        if filepath.exists():
            logger.info(f"Loading topics data from {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

class TrendAnalyzer:
    """Advanced trend analysis functionality"""
    
    def __init__(self):
        self.papers = []
        
    def analyze_research_trends(self, papers: List[Dict]) -> Dict:
        """Analyze general research trends without topic modeling"""
        logger.info("Analyzing research trends")
        
        self.papers = papers
        
        # Time-based analysis
        time_trends = self._analyze_time_trends()
        
        # Journal analysis
        journal_trends = self._analyze_journal_trends()
        
        # MeSH term trends
        mesh_trends = self._analyze_mesh_trends()
        
        # Author collaboration trends
        collaboration_trends = self._analyze_collaboration_trends()
        
        return {
            'time_trends': time_trends,
            'journal_trends': journal_trends,
            'mesh_trends': mesh_trends,
            'collaboration_trends': collaboration_trends,
            'total_papers': len(papers)
        }
    
    def _analyze_time_trends(self) -> Dict:
        """Analyze publication trends over time"""
        # Group papers by year
        yearly_counts = defaultdict(int)
        monthly_counts = defaultdict(int)
        
        for paper in self.papers:
            pub_date = paper.get('pub_date', 'Unknown')
            if pub_date != 'Unknown':
                try:
                    year = pub_date.split('-')[0]
                    yearly_counts[year] += 1
                    
                    if len(pub_date.split('-')) >= 2:
                        month_key = pub_date[:7]  # YYYY-MM
                        monthly_counts[month_key] += 1
                except:
                    continue
        
        # Calculate growth rates
        years = sorted(yearly_counts.keys())
        if len(years) >= 2:
            recent_year = yearly_counts[years[-1]]
            previous_year = yearly_counts[years[-2]] if len(years) > 1 else 1
            annual_growth = (recent_year - previous_year) / max(previous_year, 1)
        else:
            annual_growth = 0
        
        return {
            'yearly_counts': dict(yearly_counts),
            'monthly_counts': dict(monthly_counts),
            'annual_growth_rate': annual_growth,
            'peak_year': max(yearly_counts.items(), key=lambda x: x[1])[0] if yearly_counts else None,
            'total_years': len(years)
        }
    
    def _analyze_journal_trends(self) -> Dict:
        """Analyze journal publication trends"""
        journal_counts = defaultdict(int)
        journal_years = defaultdict(list)
        
        for paper in self.papers:
            journal = paper.get('journal', 'Unknown')
            pub_date = paper.get('pub_date', 'Unknown')
            
            journal_counts[journal] += 1
            
            if pub_date != 'Unknown':
                try:
                    year = int(pub_date.split('-')[0])
                    journal_years[journal].append(year)
                except:
                    continue
        
        # Top journals
        top_journals = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Journal diversity (number of unique journals)
        journal_diversity = len([j for j in journal_counts if journal_counts[j] > 1])
        
        return {
            'top_journals': [{'journal': j, 'count': c} for j, c in top_journals],
            'total_journals': len(journal_counts),
            'journal_diversity': journal_diversity
        }
    
    def _analyze_mesh_trends(self) -> Dict:
        """Analyze MeSH term trends"""
        mesh_counts = defaultdict(int)
        mesh_yearly = defaultdict(lambda: defaultdict(int))
        
        for paper in self.papers:
            mesh_terms = paper.get('mesh_terms', [])
            pub_date = paper.get('pub_date', 'Unknown')
            
            year = None
            if pub_date != 'Unknown':
                try:
                    year = pub_date.split('-')[0]
                except:
                    pass
            
            for term in mesh_terms:
                mesh_counts[term] += 1
                if year:
                    mesh_yearly[term][year] += 1
        
        # Top MeSH terms
        top_mesh = sorted(mesh_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Growing MeSH terms (terms with increasing usage)
        growing_mesh = []
        for term, yearly_data in mesh_yearly.items():
            years = sorted(yearly_data.keys())
            if len(years) >= 2:
                recent_count = yearly_data[years[-1]]
                earlier_count = sum(yearly_data[year] for year in years[:-1]) / max(len(years) - 1, 1)
                growth = (recent_count - earlier_count) / max(earlier_count, 1)
                
                if recent_count >= 3:  # Only consider terms with reasonable frequency
                    growing_mesh.append({
                        'term': term,
                        'growth_rate': growth,
                        'recent_count': recent_count,
                        'total_count': mesh_counts[term]
                    })
        
        growing_mesh.sort(key=lambda x: x['growth_rate'], reverse=True)
        
        return {
            'top_mesh_terms': [{'term': t, 'count': c} for t, c in top_mesh],
            'growing_mesh_terms': growing_mesh[:10],
            'total_unique_terms': len(mesh_counts)
        }
    
    def _analyze_collaboration_trends(self) -> Dict:
        """Analyze author collaboration trends"""
        author_counts = defaultdict(int)
        collaboration_sizes = []
        
        for paper in self.papers:
            authors = paper.get('authors', [])
            num_authors = len(authors)
            
            if num_authors > 0:
                collaboration_sizes.append(num_authors)
                
                for author in authors:
                    author_counts[author] += 1
        
        if not collaboration_sizes:
            return {}
        
        # Calculate collaboration statistics
        avg_collaboration = np.mean(collaboration_sizes)
        max_collaboration = max(collaboration_sizes)
        
        # Most prolific authors
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'avg_authors_per_paper': avg_collaboration,
            'max_authors_per_paper': max_collaboration,
            'total_unique_authors': len(author_counts),
            'top_authors': [{'author': a, 'paper_count': c} for a, c in top_authors],
            'collaboration_distribution': {
                'single_author': sum(1 for x in collaboration_sizes if x == 1),
                'small_team': sum(1 for x in collaboration_sizes if 2 <= x <= 5),
                'large_team': sum(1 for x in collaboration_sizes if x > 5)
            }
        }

class TopicModelingEngine:
    """High-level interface for topic modeling and trend analysis"""
    
    def __init__(self):
        self.topic_modeler = TopicModeler()
        self.trend_analyzer = TrendAnalyzer()
        self.topics_data = {}
    
    def initialize_topic_modeling(self) -> bool:
        """Initialize topic modeling with embeddings"""
        logger.info("Initializing topic modeling")
        
        # Check if topics already exist
        if Config.TOPICS_FILE.exists():
            logger.info("Loading existing topics data")
            self.topics_data = self.topic_modeler.load_topics_data(Config.TOPICS_FILE)
            
            # Try to load model
            model_path = Config.MODELS_DIR / 'bertopic_model'
            if self.topic_modeler.load_model(model_path):
                return True
        
        # Create new topic model
        logger.info("Creating new topic model")
        
        # Load papers and embeddings
        preprocessor = TextPreprocessor()
        papers = preprocessor.load_processed_papers(Config.PROCESSED_DATA_FILE)
        
        if not papers:
            logger.error("No papers available for topic modeling")
            return False
        
        # Load embeddings
        embeddings = None
        if Config.EMBEDDINGS_FILE.exists():
            embeddings = np.load(Config.EMBEDDINGS_FILE)
            logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        
        # Fit topic model
        model = self.topic_modeler.fit_topics(papers, embeddings)
        if not model:
            return False
        
        # Analyze trends
        trends_data = self.topic_modeler.get_topic_trends()
        
        # Save model and data
        model_path = Config.MODELS_DIR / 'bertopic_model'
        self.topic_modeler.save_model(model_path)
        
        # Combine all data
        self.topics_data = {
            'trends': trends_data,
            'topic_info': self.topic_modeler.topic_info.to_dict('records'),
            'num_topics': len(self.topic_modeler.model.get_topics())
        }
        
        self.topic_modeler.save_topics_data(self.topics_data, Config.TOPICS_FILE)
        
        logger.info("Topic modeling initialized successfully")
        return True
    
    def get_trending_topics(self, top_k: int = 10) -> List[Dict]:
        """Get top trending topics"""
        if not self.topics_data or 'trends' not in self.topics_data:
            return []
        
        trends = self.topics_data['trends'].get('trends', [])
        return trends[:top_k]
    
    def get_topic_details(self, topic_id: int) -> Dict:
        """Get detailed information about a topic"""
        if not self.topic_modeler.model:
            return {}
        
        return self.topic_modeler.get_topic_details(topic_id)
    
    def analyze_general_trends(self) -> Dict:
        """Analyze general research trends"""
        preprocessor = TextPreprocessor()
        papers = preprocessor.load_processed_papers(Config.PROCESSED_DATA_FILE)
        
        if not papers:
            return {}
        
        return self.trend_analyzer.analyze_research_trends(papers)
    
    def create_visualizations(self) -> Dict:
        """Create all visualizations"""
        visualizations = {}
        
        # Topic visualizations
        if self.topic_modeler.model:
            topic_viz = self.topic_modeler.visualize_topics()
            visualizations.update(topic_viz)
        
        # Trend visualizations
        if self.topics_data and 'trends' in self.topics_data:
            trend_viz = self.topic_modeler.create_trend_visualizations(self.topics_data['trends'])
            visualizations.update(trend_viz)
        
        return visualizations
    
    def get_topic_timeline(self, topic_id: int) -> Dict:
        """Get timeline data for a specific topic"""
        if not self.topics_data or 'trends' not in self.topics_data:
            return {}
        
        trends = self.topics_data['trends'].get('trends', [])
        
        for trend in trends:
            if trend['topic_id'] == topic_id:
                return {
                    'topic_id': topic_id,
                    'topic_label': trend['topic_label'],
                    'timeline': trend['time_series'],
                    'growth_rate': trend['growth_rate'],
                    'growth_score': trend['growth_score']
                }
        
        return {}

if __name__ == "__main__":
    # Test topic modeling
    engine = TopicModelingEngine()
    
    if engine.initialize_topic_modeling():
        logger.info("Topic modeling initialized successfully")
        
        # Get trending topics
        trending = engine.get_trending_topics(5)
        logger.info(f"Found {len(trending)} trending topics")
        
        for topic in trending:
            logger.info(f"Topic: {topic['topic_label']} (Growth Score: {topic['growth_score']:.3f})")
    else:
        logger.error("Failed to initialize topic modeling")