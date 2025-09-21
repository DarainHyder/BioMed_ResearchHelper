# Biomedical Research Assistant - Complete Setup Guide

This guide will walk you through setting up and running the complete biomedical research assistant project on Windows.

## 📋 Prerequisites

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **Git** (optional, for cloning)
- **At least 8GB RAM** (16GB+ recommended for large datasets)
- **10GB+ free disk space**
- **Internet connection** for downloading models and data

## 🚀 Quick Start (5-Minute Setup)

### Step 1: Download and Extract Files

1. Download all the project files to a folder (e.g., `C:\biomedical-research-assistant`)
2. Your folder structure should look like this:

```
biomedical-research-assistant/
├── main.py
├── config.py
├── requirements.txt
├── .env.template
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── summarization.py
│   ├── topic_modeling.py
│   ├── api.py
│   └── streamlit_app.py
├── data/ (will be created automatically)
├── models/ (will be created automatically)
└── cache/ (will be created automatically)
```

### Step 2: Create Virtual Environment

Open Command Prompt or PowerShell as Administrator and run:

```bash
# Navigate to project directory
cd C:\biomedical-research-assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note:** This may take 10-15 minutes as it downloads large ML models.

### Step 4: Configure Environment

1. Copy the environment template:
```bash
copy .env.template .env
```

2. Edit `.env` file with your details:
```bash
# Open .env in notepad
notepad .env
```

3. Configure the required settings:
```env
# Your email for NCBI API (REQUIRED)
NCBI_EMAIL=your.email@example.com

# Optional: NCBI API key for higher rate limits
# Get from: https://www.ncbi.nlm.nih.gov/account/settings/
NCBI_API_KEY=your_api_key_here

# Research domain (customize as needed)
RESEARCH_DOMAIN=covid immunotherapy
MAX_PAPERS=2000
DATE_FROM=2020/01/01
DATE_TO=2024/12/31
```

**Important:** Replace `your.email@example.com` with your actual email address. This is required by NCBI.

### Step 5: Check Configuration

```bash
python main.py check
```

You should see: "✅ Configuration check passed!"

## 🏃‍♂️ Running the System

### Option A: Complete Setup (Recommended for First Time)

This will download papers, process them, create embeddings, and set up everything:

```bash
python main.py setup
```

**Expected time:** 30-60 minutes depending on your settings and internet speed.

**What happens during setup:**
1. Downloads papers from PubMed (5-10 minutes)
2. Processes and cleans text (5-10 minutes)
3. Creates embeddings using AI models (15-30 minutes)
4. Generates summaries (10-20 minutes)
5. Builds topic models and trend analysis (10-15 minutes)

### Option B: Start with Demo (Quick Test)

If you want to test with minimal setup:

1. First, run a quick ingestion:
```bash
# Edit .env and set MAX_PAPERS=100 for testing
python main.py setup
```

## 🖥️ Using the System

### 1. Start the API Server

```bash
python main.py server
```

The API will be available at: `http://localhost:8000`

**API Endpoints:**
- `http://localhost:8000/docs` - Interactive API documentation
- `http://localhost:8000/health` - Health check
- `http://localhost:8000/search?q=covid vaccine` - Search example

### 2. Start the Web Dashboard

Open a **new** Command Prompt/PowerShell window:

```bash
# Navigate to project directory
cd C:\biomedical-research-assistant

# Activate virtual environment
venv\Scripts\activate

# Start dashboard
python main.py dashboard
```

The dashboard will open automatically in your browser at: `http://localhost:8501`

## 🎮 Demo Commands

Test individual components:

```bash
# Test search functionality
python main.py demo-search

# Test summarization
python main.py demo-summary

# Test topic analysis
python main.py demo-topics
```

## 📊 Using the Web Dashboard

### Search & Explore Page
1. **Semantic Search**: Enter research queries like "COVID-19 vaccine efficacy"
2. **Summarized Results**: Get AI-generated summaries of multiple papers
3. **View paper details, abstracts, and similar papers**

### Trending Topics Page
1. **View hot research topics** with growth scores
2. **Analyze topic trends** over time
3. **Explore topic details** and related papers

### Research Trends Page
1. **Publication trends** over time
2. **Top journals** and collaboration patterns
3. **MeSH term analysis** and growing research areas

### Paper Details Page
1. **Enter PMIDs** to get detailed paper information
2. **Generate summaries** for specific papers
3. **Find similar papers** using AI similarity

## 🛠️ Customization

### Change Research Domain

Edit `.env` file:
```env
RESEARCH_DOMAIN=diabetes treatment machine learning
MAX_PAPERS=5000
DATE_FROM=2018/01/01
DATE_TO=2024/12/31
```

Then run setup again:
```bash
python main.py setup
```

### Modify Models

Edit `config.py`:
```python
# Use biomedical-specific embeddings
EMBEDDING_MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

# Use different summarization model
SUMMARIZATION_MODEL = 'microsoft/DialoGPT-medium'
```

### Adjust Processing Parameters

Edit `config.py`:
```python
# Process more papers
MAX_PAPERS = 10000

# Get more search results
TOP_K_RESULTS = 20

# Longer summaries
MAX_SUMMARY_LENGTH = 200
```

## 🔧 Troubleshooting

### Common Issues

**1. "NCBI_EMAIL is required" Error**
- Edit `.env` file and set your email address
- Make sure there are no spaces around the `=` sign

**2. "API is not available" in Dashboard**
- Make sure the API server is running: `python main.py server`
- Check if port 8000 is blocked by firewall

**3. Out of Memory Errors**
- Reduce `MAX_PAPERS` in `.env` file
- Close other applications
- Consider using a machine with more RAM

**4. Slow Performance**
- Set `MAX_PAPERS=1000` or lower for testing
- Use GPU if available (will be detected automatically)
- Consider using lighter embedding models

**5. ImportError or Module Not Found**
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- Make sure virtual environment is activated
- Try: `pip install --upgrade transformers sentence-transformers`

**6. "No papers found" Error**
- Check your internet connection
- Verify NCBI email is valid
- Try a broader search term in `RESEARCH_DOMAIN`

### Performance Optimization

**For Faster Processing:**
```env
MAX_PAPERS=1000
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**For Better Quality:**
```env
MAX_PAPERS=5000
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

**For Biomedical Accuracy:**
```env
EMBEDDING_MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
```

## 📁 File Structure Explained

```
biomedical-research-assistant/
├── main.py                    # Main execution script
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .env                      # Your configuration (create from template)
├── .env.template             # Configuration template
├── src/
│   ├── data_ingestion.py     # PubMed data collection
│   ├── preprocessing.py      # Text cleaning and processing
│   ├── embeddings.py         # AI embeddings and search
│   ├── summarization.py      # AI summarization
│   ├── topic_modeling.py     # Topic analysis and trends
│   ├── api.py               # FastAPI backend server
│   └── streamlit_app.py     # Web dashboard
├── data/                     # Data files (auto-created)
│   ├── raw_papers.jsonl     # Downloaded papers
│   ├── processed_papers.jsonl # Cleaned papers
│   ├── embeddings.npy       # AI embeddings
│   ├── faiss_index.bin      # Search index
│   ├── summaries.jsonl      # Generated summaries
│   └── topics.json          # Topic analysis results
├── models/                   # AI models cache (auto-created)
└── cache/                    # Temporary files (auto-created)
```

## 🎯 Next Steps

1. **Start Small**: Begin with 500-1000 papers to test everything works
2. **Scale Up**: Once comfortable, increase to 5000+ papers for better analysis
3. **Customize**: Modify research domains and parameters for your specific needs
4. **Integrate**: Use the API endpoints to integrate with your own applications
5. **Extend**: Add new features by modifying the source code

## 📚 Additional Resources

- **PubMed API Documentation**: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **Sentence Transformers**: https://www.sbert.net/
- **BERTopic Documentation**: https://maartengr.github.io/BERTopic/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs** - Look for error messages in the console
2. **Verify configuration** - Run `python main.py check`
3. **Test components individually** - Use the demo commands
4. **Check system resources** - Ensure adequate RAM and disk space
5. **Try with smaller datasets** - Reduce MAX_PAPERS for testing

## 🔐 Important Notes

- **Medical Disclaimer**: This tool is for research purposes only. Always consult healthcare professionals for medical decisions.
- **Rate Limits**: NCBI has rate limits. Get an API key for higher limits.
- **Privacy**: No patient data is stored. Only public research paper metadata is used.
- **Resources**: Processing large datasets requires significant computational resources.

---

**Happy Researching! 🔬📊**