# 🔬 Biomedical Research Assistant

A comprehensive AI-powered system for biomedical literature analysis, semantic search, summarization, and trend discovery.

## ✨ Features

- **🔍 Semantic Search**: AI-powered search through biomedical literature
- **📄 Auto-Summarization**: Generate concise summaries of research papers
- **📈 Trend Analysis**: Discover trending topics and research patterns
- **🎯 Topic Modeling**: Identify and analyze research themes
- **🌐 Web Dashboard**: Interactive Streamlit interface
- **🚀 REST API**: FastAPI backend for integration
- **📊 Visualizations**: Charts and graphs for trend analysis

## 🎯 Use Cases

- **Researchers**: Quickly find relevant papers and identify research gaps
- **Clinicians**: Stay updated with latest medical research
- **Students**: Understand research trends and topics
- **Data Scientists**: Analyze patterns in biomedical literature
- **Institutions**: Monitor research output and collaborations

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd biomedical-research-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Settings

```bash
# Copy configuration template
copy .env.template .env  # Windows
# cp .env.template .env  # Linux/Mac

# Edit .env with your email (required for PubMed API)
# NCBI_EMAIL=your.email@example.com
```

### 3. Run Setup

```bash
# Check configuration
python main.py check

# Set up data pipeline (30-60 minutes first time)
python main.py setup
```

### 4. Start Using

```bash
# Start API server
python main.py server

# Start web dashboard (in another terminal)
python main.py dashboard
```

Open your browser to `http://localhost:8501` for the dashboard!

## 📖 Detailed Documentation

See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for complete setup guide.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   AI Processing  │    │   User Interface│
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • PubMed API    │───▶│ • Text Cleaning  │───▶│ • Web Dashboard │
│ • Research Papers│    │ • Embeddings     │    │ • REST API      │
│ • Metadata      │    │ • Summarization  │    │ • Visualizations│
│ • MeSH Terms    │    │ • Topic Modeling │    │ • Search Results│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Components

### Data Pipeline
- **Data Ingestion**: Fetches papers from PubMed using Entrez API
- **Preprocessing**: Cleans and structures text data
- **Embeddings**: Creates semantic vectors using Sentence Transformers
- **Indexing**: Builds FAISS index for fast similarity search

### AI Models
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Summarization**: `facebook/bart-large-cnn`
- **Topic Modeling**: BERTopic with biomedical optimizations

### Applications
- **API Server**: FastAPI with auto-generated documentation
- **Web Dashboard**: Streamlit with interactive visualizations
- **CLI Tools**: Command-line interface for all operations

## 📊 Example Queries

- "COVID-19 vaccine efficacy clinical trials"
- "cancer immunotherapy checkpoint inhibitors"
- "alzheimer disease biomarkers tau protein"
- "diabetes treatment metformin mechanism"
- "machine learning medical imaging"

## 🔧 Configuration Options

### Research Domain
```env
RESEARCH_DOMAIN=covid immunotherapy
MAX_PAPERS=5000
DATE_FROM=2020/01/01
DATE_TO=2024/12/31
```

### Model Selection
```env
# Standard (fast)
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Biomedical (accurate)
EMBEDDING_MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
```

### Performance Tuning
```env
# For testing
MAX_PAPERS=1000
TOP_K_RESULTS=10

# For production
MAX_PAPERS=10000
TOP_K_RESULTS=20
```

## 📈 Performance

| Dataset Size | Setup Time | Search Speed | Memory Usage |
|--------------|------------|--------------|--------------|
| 1K papers    | 5-10 min   | <100ms       | 2-4 GB       |
| 5K papers    | 20-30 min  | <200ms       | 4-8 GB       |
| 10K+ papers  | 45-60 min  | <300ms       | 8-16 GB      |

## 🤖 API Endpoints

### Search
```http
GET /search?q=covid+vaccine&top_k=10
POST /search
```

### Summarization
```http
GET /summarize?q=covid+vaccine&top_k=5
GET /paper/{pmid}/summary
```

### Topics & Trends
```http
GET /topics/trending?top_k=10
GET /topics/{topic_id}
GET /trends/general
```

### Paper Details
```http
GET /paper/{pmid}
GET /paper/{pmid}/similar
```

## 🎨 Dashboard Features

### 🔍 Search & Explore
- Semantic search with similarity scores
- Multi-paper summarization
- Similar paper recommendations
- Interactive result filtering

### 📈 Trending Topics
- Real-time trending topic identification
- Growth rate analysis
- Topic evolution over time
- Representative paper extraction

### 📊 Research Trends
- Publication trends by year/month
- Journal analysis and rankings
- Author collaboration patterns
- MeSH term frequency analysis

### 🎯 Paper Analysis
- Individual paper summaries
- Citation-style information
- Related paper discovery
- Metadata extraction

## 🔐 Privacy & Ethics

- **No Personal Data**: Only public research metadata is processed
- **Medical Disclaimer**: For research purposes only, not medical advice
- **Rate Limiting**: Respects PubMed API rate limits
- **Open Source**: Transparent algorithms and processing

## 🛡️ System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 5GB storage
- Internet connection

**Recommended:**
- Python 3.9+
- 16GB+ RAM
- 20GB+ storage
- GPU (optional, for faster processing)

## 🚨 Important Notes

⚠️ **Medical Disclaimer**: This tool is for research and educational purposes only. It does not provide medical advice, diagnosis, or treatment recommendations. Always consult qualified healthcare professionals for medical decisions.

📚 **Data Source**: All data comes from publicly available research papers via PubMed/NCBI APIs.

🔄 **Updates**: The system processes research papers available up to your search date range. For the most current research, regularly update your dataset.

## 📞 Support & Contributing

- **Issues**: Report bugs and request features via GitHub issues
- **Documentation**: See setup guide and API documentation
- **Community**: Join discussions and share improvements
- **Contributing**: Pull requests welcome for new features and fixes

## 📜 License

This project is open source. See LICENSE file for details.

## 🙏 Acknowledgments

- **NCBI/PubMed** for providing access to biomedical literature
- **Hugging Face** for transformer models and libraries
- **Streamlit & FastAPI** for web framework components
- **Scientific Community** for open access research

---

**Built with ❤️ for the research community**

*Empowering discovery through AI-driven literature analysis*