import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from typing import List, Dict
import time

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Biomedical Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = f"http://{Config.API_HOST}:{Config.API_PORT}"

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.search-box {
    font-size: 1.1rem;
    padding: 0.5rem;
    border-radius: 0.5rem;
}
.paper-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
.warning-banner {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #ffeaa7;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def make_api_request(endpoint: str, params: dict = None) -> dict:
    """Make API request with caching"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return {}

def check_api_health() -> bool:
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        data = response.json()
        return data.get("status") == "healthy"
    except:
        return False

def display_paper_card(paper: dict, show_similarity: bool = False):
    """Display a paper in a card format"""
    with st.container():
        st.markdown('<div class="paper-card">', unsafe_allow_html=True)
        
        # Title
        title = paper.get('title', 'No title available')
        st.markdown(f"**{title}**")
        
        # Metadata
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.text(f"PMID: {paper.get('pmid', 'N/A')}")
        with col2:
            st.text(f"Journal: {paper.get('journal', 'Unknown')[:30]}...")
        with col3:
            st.text(f"Date: {paper.get('pub_date', 'Unknown')}")
        
        # Similarity score if available
        if show_similarity and 'similarity_score' in paper:
            st.progress(float(paper['similarity_score']))
            st.text(f"Similarity: {paper['similarity_score']:.3f}")
        
        # Abstract
        abstract = paper.get('abstract', 'No abstract available')
        if abstract:
            with st.expander("View Abstract"):
                st.text(abstract[:500] + "..." if len(abstract) > 500 else abstract)
        
        # Summary if available
        if paper.get('summary'):
            st.markdown("**Summary:**")
            st.text(paper['summary'])
        
        # MeSH terms
        mesh_terms = paper.get('mesh_terms', [])
        if mesh_terms:
            st.markdown("**MeSH Terms:**")
            st.text(", ".join(mesh_terms[:5]) + ("..." if len(mesh_terms) > 5 else ""))
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(f"View Details", key=f"details_{paper['pmid']}"):
                st.session_state.selected_paper = paper['pmid']
        with col2:
            if st.button(f"Find Similar", key=f"similar_{paper['pmid']}"):
                st.session_state.find_similar_pmid = paper['pmid']
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Biomedical Research Assistant</h1>', unsafe_allow_html=True)
    
    # Medical disclaimer
    st.markdown("""
    <div class="warning-banner">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research purposes only and does not provide medical advice. 
    Always consult with healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not available. Please ensure the backend server is running.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        
        # Get system stats
        stats = make_api_request("/stats")
        if stats:
            st.subheader("📊 System Statistics")
            
            search_stats = stats.get('search', {})
            if search_stats:
                st.metric("Total Papers", search_stats.get('num_papers', 0))
                st.metric("Embeddings", search_stats.get('num_embeddings', 0))
            
            trends_stats = stats.get('trends', {})
            if trends_stats:
                st.metric("Unique Authors", trends_stats.get('collaboration_trends', {}).get('total_unique_authors', 0))
                st.metric("Unique Journals", trends_stats.get('journal_trends', {}).get('total_journals', 0))
        
        st.divider()
        
        # Page selection
        page = st.selectbox(
            "Choose a page:",
            ["🔍 Search & Explore", "📈 Trending Topics", "📊 Research Trends", "🎯 Paper Details"]
        )
    
    # Main content based on selected page
    if page == "🔍 Search & Explore":
        search_page()
    elif page == "📈 Trending Topics":
        trending_topics_page()
    elif page == "📊 Research Trends":
        research_trends_page()
    elif page == "🎯 Paper Details":
        paper_details_page()

def search_page():
    """Search and exploration page"""
    st.header("🔍 Search & Explore Research Papers")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter your research query:",
            placeholder="e.g., COVID-19 vaccine efficacy, cancer immunotherapy, diabetes treatment",
            key="search_query"
        )
    
    with col2:
        top_k = st.selectbox("Results:", [5, 10, 20, 50], index=1)
    
    # Search options
    col1, col2 = st.columns([1, 1])
    with col1:
        search_type = st.radio("Search Type:", ["Semantic Search", "Summarized Results"])
    with col2:
        if search_type == "Summarized Results":
            summary_papers = st.slider("Papers to summarize:", 3, 10, 5)
    
    # Search button
    if st.button("🔍 Search", type="primary") or search_query:
        if search_query:
            if search_type == "Semantic Search":
                # Semantic search
                with st.spinner("Searching..."):
                    results = make_api_request("/search", {"q": search_query, "top_k": top_k})
                
                if results and results.get('results'):
                    st.success(f"Found {results['total_results']} papers")
                    
                    # Display results
                    for i, paper in enumerate(results['results']):
                        st.markdown(f"### Result {i+1}")
                        display_paper_card(paper, show_similarity=True)
                else:
                    st.warning("No papers found for your query.")
            
            else:
                # Summarized results
                with st.spinner("Generating summary..."):
                    summary_result = make_api_request("/summarize", 
                                                    {"q": search_query, "top_k": summary_papers})
                
                if summary_result:
                    st.subheader("📄 Research Summary")
                    
                    if summary_result.get('summary'):
                        st.markdown("**Summary:**")
                        st.write(summary_result['summary'])
                        
                        # Show source papers
                        if summary_result.get('papers'):
                            st.subheader("📚 Source Papers")
                            for paper in summary_result['papers']:
                                display_paper_card(paper)
                    else:
                        st.error("Failed to generate summary.")
    
    # Handle similar paper search
    if 'find_similar_pmid' in st.session_state:
        pmid = st.session_state.find_similar_pmid
        del st.session_state.find_similar_pmid
        
        st.subheader(f"Papers Similar to PMID: {pmid}")
        
        with st.spinner("Finding similar papers..."):
            similar_results = make_api_request(f"/paper/{pmid}/similar", {"top_k": 10})
        
        if similar_results and similar_results.get('similar_papers'):
            for paper in similar_results['similar_papers']:
                display_paper_card(paper, show_similarity=True)
        else:
            st.warning("No similar papers found.")

def trending_topics_page():
    """Trending topics analysis page"""
    st.header("📈 Trending Research Topics")
    
    # Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.selectbox("Number of topics:", [5, 10, 15, 20], index=1)
    
    # Get trending topics
    with st.spinner("Loading trending topics..."):
        trending_data = make_api_request("/topics/trending", {"top_k": top_k})
    
    if trending_data and trending_data.get('trending_topics'):
        topics = trending_data['trending_topics']
        
        st.subheader("🔥 Top Trending Topics")
        
        # Display topics in cards
        for i, topic in enumerate(topics):
            with st.expander(f"{i+1}. {topic['topic_label']} (Growth Score: {topic['growth_score']:.3f})"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.metric("Growth Rate", f"{topic['growth_rate']*100:.1f}%")
                    st.metric("Recent Papers", topic['recent_count'])
                
                with col2:
                    st.metric("Average Papers", f"{topic['avg_count']:.1f}")
                    st.metric("Growth Score", f"{topic['growth_score']:.3f}")
                
                # Show detailed topic information
                if st.button(f"View Topic Details", key=f"topic_details_{topic['topic_id']}"):
                    st.session_state.selected_topic = topic['topic_id']
        
        # Topic trends visualization
        if len(topics) > 0:
            st.subheader("📊 Topic Growth Visualization")
            
            # Prepare data for plotting
            topic_names = [topic['topic_label'][:30] + "..." if len(topic['topic_label']) > 30 
                          else topic['topic_label'] for topic in topics]
            growth_scores = [topic['growth_score'] for topic in topics]
            
            fig = px.bar(
                x=topic_names,
                y=growth_scores,
                title="Topic Growth Scores",
                labels={'x': 'Topic', 'y': 'Growth Score'}
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No trending topics data available.")
    
    # Handle topic detail view
    if 'selected_topic' in st.session_state:
        topic_id = st.session_state.selected_topic
        del st.session_state.selected_topic
        
        with st.spinner("Loading topic details..."):
            topic_details = make_api_request(f"/topics/{topic_id}")
        
        if topic_details:
            st.subheader(f"Topic Details: {topic_details.get('topic_label', 'Unknown')}")
            
            # Topic information
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Number of Papers", topic_details.get('num_papers', 0))
            with col2:
                st.metric("Topic ID", topic_id)
            
            # Keywords
            keywords = topic_details.get('keywords', [])
            if keywords:
                st.subheader("🔑 Key Terms")
                keyword_df = pd.DataFrame(keywords)
                st.dataframe(keyword_df, use_container_width=True)
            
            # Top papers
            top_papers = topic_details.get('top_papers', [])
            if top_papers:
                st.subheader("📚 Representative Papers")
                for paper in top_papers[:5]:
                    display_paper_card(paper)
            
            # MeSH terms
            mesh_terms = topic_details.get('top_mesh_terms', [])
            if mesh_terms:
                st.subheader("🏷️ Related MeSH Terms")
                mesh_df = pd.DataFrame(mesh_terms)
                st.dataframe(mesh_df, use_container_width=True)

def research_trends_page():
    """General research trends page"""
    st.header("📊 Research Trends Analysis")
    
    # Get general trends data
    with st.spinner("Loading research trends..."):
        trends_data = make_api_request("/trends/general")
    
    if not trends_data:
        st.error("Failed to load trends data.")
        return
    
    # Time trends
    time_trends = trends_data.get('time_trends', {})
    if time_trends:
        st.subheader("📅 Publication Trends Over Time")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Annual Growth Rate", f"{time_trends.get('annual_growth_rate', 0)*100:.1f}%")
        with col2:
            st.metric("Peak Year", time_trends.get('peak_year', 'Unknown'))
        with col3:
            st.metric("Years Covered", time_trends.get('total_years', 0))
        
        # Yearly publication chart
        yearly_data = time_trends.get('yearly_counts', {})
        if yearly_data:
            years = list(yearly_data.keys())
            counts = list(yearly_data.values())
            
            fig = px.line(
                x=years,
                y=counts,
                title="Publications by Year",
                labels={'x': 'Year', 'y': 'Number of Publications'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Journal trends
    journal_trends = trends_data.get('journal_trends', {})
    if journal_trends:
        st.subheader("📰 Journal Analysis")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Total Journals", journal_trends.get('total_journals', 0))
        with col2:
            st.metric("Journal Diversity", journal_trends.get('journal_diversity', 0))
        
        # Top journals
        top_journals = journal_trends.get('top_journals', [])
        if top_journals:
            st.subheader("🏆 Top Journals")
            journal_df = pd.DataFrame(top_journals)
            
            # Create bar chart
            fig = px.bar(
                journal_df.head(10),
                x='count',
                y='journal',
                orientation='h',
                title="Top 10 Journals by Publication Count"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # MeSH term trends
    mesh_trends = trends_data.get('mesh_trends', {})
    if mesh_trends:
        st.subheader("🏷️ MeSH Term Analysis")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Unique MeSH Terms", mesh_trends.get('total_unique_terms', 0))
        
        # Top MeSH terms
        top_mesh = mesh_trends.get('top_mesh_terms', [])
        if top_mesh:
            st.subheader("🔥 Most Frequent MeSH Terms")
            mesh_df = pd.DataFrame(top_mesh[:15])
            
            fig = px.bar(
                mesh_df,
                x='count',
                y='term',
                orientation='h',
                title="Top 15 MeSH Terms"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Growing MeSH terms
        growing_mesh = mesh_trends.get('growing_mesh_terms', [])
        if growing_mesh:
            st.subheader("📈 Rapidly Growing MeSH Terms")
            growing_df = pd.DataFrame(growing_mesh[:10])
            st.dataframe(growing_df, use_container_width=True)
    
    # Collaboration trends
    collab_trends = trends_data.get('collaboration_trends', {})
    if collab_trends:
        st.subheader("🤝 Collaboration Analysis")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Unique Authors", collab_trends.get('total_unique_authors', 0))
        with col2:
            st.metric("Avg Authors/Paper", f"{collab_trends.get('avg_authors_per_paper', 0):.1f}")
        with col3:
            st.metric("Max Authors/Paper", collab_trends.get('max_authors_per_paper', 0))
        
        # Collaboration distribution
        collab_dist = collab_trends.get('collaboration_distribution', {})
        if collab_dist:
            st.subheader("👥 Team Size Distribution")
            
            categories = ['Single Author', 'Small Team (2-5)', 'Large Team (6+)']
            values = [
                collab_dist.get('single_author', 0),
                collab_dist.get('small_team', 0),
                collab_dist.get('large_team', 0)
            ]
            
            fig = px.pie(
                values=values,
                names=categories,
                title="Distribution of Team Sizes"
            )
            st.plotly_chart(fig, use_container_width=True)

def paper_details_page():
    """Paper details page"""
    st.header("🎯 Paper Details & Analysis")
    
    # PMID input
    pmid_input = st.text_input("Enter PMID:", placeholder="e.g., 34743777")
    
    if pmid_input:
        # Get paper details
        with st.spinner("Loading paper details..."):
            paper = make_api_request(f"/paper/{pmid_input}")
        
        if paper:
            st.success("Paper found!")
            display_paper_card(paper)
            
            # Additional analysis options
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary_result = make_api_request(f"/paper/{pmid_input}/summary")
                    
                    if summary_result and summary_result.get('summary'):
                        st.subheader("📄 Generated Summary")
                        st.write(summary_result['summary'])
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.metric("Original Words", summary_result.get('original_length', 0))
                        with col2:
                            st.metric("Summary Words", summary_result.get('summary_length', 0))
                        with col3:
                            ratio = summary_result.get('compression_ratio', 0)
                            st.metric("Compression", f"{ratio:.2f}")
            
            with col2:
                if st.button("Find Similar Papers"):
                    with st.spinner("Finding similar papers..."):
                        similar_results = make_api_request(f"/paper/{pmid_input}/similar")
                    
                    if similar_results and similar_results.get('similar_papers'):
                        st.subheader("🔗 Similar Papers")
                        for similar_paper in similar_results['similar_papers'][:5]:
                            display_paper_card(similar_paper, show_similarity=True)
        
        else:
            st.error(f"Paper with PMID {pmid_input} not found.")
    
    else:
        st.info("Enter a PMID to view paper details and analysis.")

if __name__ == "__main__":
    main()