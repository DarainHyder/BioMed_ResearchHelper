from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import logging
from pydantic import BaseModel
import uvicorn

from config import Config
from src.embeddings import SemanticSearchEngine
from src.summarization import SummarizationEngine
from src.topic_modeling import TopicModelingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    total_results: int

class SummaryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class TopicDetailsResponse(BaseModel):
    topic_id: int
    topic_label: str
    keywords: List[Dict]
    num_papers: int
    top_papers: List[Dict]

class TrendingTopicsResponse(BaseModel):
    trending_topics: List[Dict]
    time_window: str
    total_topics: int

# Initialize FastAPI app
app = FastAPI(
    title="Biomedical Research Assistant API",
    description="API for semantic search, summarization, and trend analysis of biomedical literature",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for engines
search_engine = None
summarization_engine = None
topic_engine = None
initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize all engines on startup"""
    global search_engine, summarization_engine, topic_engine, initialized
    
    logger.info("Initializing engines...")
    
    try:
        # Initialize search engine
        search_engine = SemanticSearchEngine()
        if not search_engine.initialize():
            logger.error("Failed to initialize search engine")
            # Without search engine, we can't do anything, so we must abort
            initialized = False
            return
    except Exception as e:
        logger.error(f"Critical error during search engine initialization: {e}")
        initialized = False
        return
        
    try:
        # Initialize summarization engine
        summarization_engine = SummarizationEngine()
        summarization_engine.initialize(search_engine)
    except Exception as e:
        logger.error(f"Failed to initialize summarization engine (Continuing without it): {e}")
        summarization_engine = None
        
    try:
        # Initialize topic modeling engine
        topic_engine = TopicModelingEngine()
        topic_engine.initialize_topic_modeling()
    except Exception as e:
        logger.error(f"Failed to initialize topic engine (Continuing without it): {e}")
        topic_engine = None
        
    initialized = True
    logger.info("Engines initialization phase complete")

def check_initialization():
    """Check if engines are initialized"""
    if not initialized:
        raise HTTPException(status_code=503, detail="System not initialized. Please wait and try again.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Biomedical Research Assistant API", "status": "running" if initialized else "initializing"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if initialized else "initializing",
        "engines": {
            "search": search_engine is not None,
            "summarization": summarization_engine is not None,
            "topics": topic_engine is not None
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    check_initialization()
    
    stats = {}
    
    if search_engine:
        stats["search"] = search_engine.get_stats()
    
    if topic_engine:
        general_trends = topic_engine.analyze_general_trends()
        stats["trends"] = general_trends
    
    return stats

@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Perform semantic search"""
    check_initialization()
    
    try:
        results = search_engine.search(request.query, request.top_k)
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of results to return")
):
    """GET endpoint for search (for easier testing)"""
    request = SearchRequest(query=q, top_k=top_k)
    return await semantic_search(request)

@app.get("/paper/{pmid}")
async def get_paper(pmid: str):
    """Get paper details by PMID"""
    check_initialization()
    
    try:
        paper = search_engine.get_paper(pmid)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {pmid} not found")
        
        return paper
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/paper/{pmid}/similar")
async def get_similar_papers(
    pmid: str,
    top_k: int = Query(10, description="Number of similar papers to return")
):
    """Get papers similar to a given paper"""
    check_initialization()
    
    try:
        similar_papers = search_engine.find_similar(pmid, top_k)
        return {
            "pmid": pmid,
            "similar_papers": similar_papers,
            "total_results": len(similar_papers)
        }
        
    except Exception as e:
        logger.error(f"Error finding similar papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_search_results(request: SummaryRequest):
    """Summarize search results for a query"""
    check_initialization()
    
    try:
        summary_result = summarization_engine.summarize_search_results(
            request.query, 
            request.top_k
        )
        
        return summary_result
        
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summarize")
async def summarize_get(
    q: str = Query(..., description="Query to summarize"),
    top_k: int = Query(5, description="Number of papers to include in summary")
):
    """GET endpoint for summarization"""
    request = SummaryRequest(query=q, top_k=top_k)
    return await summarize_search_results(request)

@app.get("/paper/{pmid}/summary")
async def summarize_paper(pmid: str):
    """Get summary for a specific paper"""
    check_initialization()
    
    try:
        summary_result = summarization_engine.summarize_paper(pmid)
        return summary_result
        
    except Exception as e:
        logger.error(f"Error summarizing paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics/trending", response_model=TrendingTopicsResponse)
async def get_trending_topics(
    top_k: int = Query(10, description="Number of trending topics to return")
):
    """Get trending topics"""
    check_initialization()
    
    try:
        trending_topics = topic_engine.get_trending_topics(top_k)
        
        return TrendingTopicsResponse(
            trending_topics=trending_topics,
            time_window="month",
            total_topics=len(trending_topics)
        )
        
    except Exception as e:
        logger.error(f"Error getting trending topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics/{topic_id}", response_model=TopicDetailsResponse)
async def get_topic_details(topic_id: int):
    """Get detailed information about a specific topic"""
    check_initialization()
    
    try:
        topic_details = topic_engine.get_topic_details(topic_id)
        
        if not topic_details:
            raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")
        
        return TopicDetailsResponse(**topic_details)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics/{topic_id}/timeline")
async def get_topic_timeline(topic_id: int):
    """Get timeline data for a specific topic"""
    check_initialization()
    
    try:
        timeline_data = topic_engine.get_topic_timeline(topic_id)
        
        if not timeline_data:
            raise HTTPException(status_code=404, detail=f"Timeline for topic {topic_id} not found")
        
        return timeline_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trends/general")
async def get_general_trends():
    """Get general research trends"""
    check_initialization()
    
    try:
        trends = topic_engine.analyze_general_trends()
        return trends
        
    except Exception as e:
        logger.error(f"Error analyzing general trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations")
async def get_visualizations():
    """Get visualization HTML"""
    check_initialization()
    
    try:
        visualizations = topic_engine.create_visualizations()
        return visualizations
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        "src.api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()