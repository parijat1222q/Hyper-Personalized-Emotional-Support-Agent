import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from neo4j import GraphDatabase
import redis
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AI Worker", version="1.0.0")

# Environment configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
NEO4J_URL = os.getenv("NEO4J_URI", os.getenv("NEO4J_URL", "bolt://neo4j-graph:7687"))

# Parse NEO4J_AUTH environment variable (format: "user/password")
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/password")
if "/" in NEO4J_AUTH:
    NEO4J_USER, NEO4J_PASSWORD = NEO4J_AUTH.split("/", 1)
else:
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

REDIS_HOST = os.getenv("REDIS_HOST", "redis-memory")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-vector")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E4B-it")

# Initialize clients
def init_neo4j():
    """Initialize Neo4j driver"""
    try:
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info("✓ Neo4j connected")
        return driver
    except Exception as e:
        logger.error(f"✗ Neo4j connection failed: {e}")
        return None

def init_redis():
    """Initialize Redis client"""
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_client.ping()
        logger.info("✓ Redis connected")
        return redis_client
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        return None

def init_qdrant():
    """Initialize Qdrant client"""
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("✓ Qdrant connected")
        return qdrant_client
    except Exception as e:
        logger.error(f"✗ Qdrant connection failed: {e}")
        return None

def init_hf_client():
    """Initialize Hugging Face InferenceClient"""
    try:
        if not HF_TOKEN:
            logger.warning("⚠ HF_TOKEN not set - using anonymous access")
            return InferenceClient(model=MODEL_NAME)
        return InferenceClient(model=MODEL_NAME, token=HF_TOKEN)
    except Exception as e:
        logger.error(f"✗ HF InferenceClient initialization failed: {e}")
        return None

# Initialize all clients at startup
neo4j_driver = init_neo4j()
redis_client = init_redis()
qdrant_client = init_qdrant()
hf_client = init_hf_client()

# Request/Response models
class AnalyzeRequest(BaseModel):
    user_id: str
    resolved_text: str
    context: Optional[str] = None

class KnowledgeFetchRequest(BaseModel):
    query: str
    source: str = "reddit"
    limit: int = 5

class SemanticTriple(BaseModel):
    subject: str
    predicate: str
    object: str

class AnalyzeResponse(BaseModel):
    status: str
    extracted_triples: List[Dict[str, str]]
    error: Optional[str] = None

class KnowledgeFetchResponse(BaseModel):
    status: str
    fetched_items: List[Dict[str, Any]]
    extracted_triples: List[Dict[str, str]]
    error: Optional[str] = None

# LLM Semantic Triple Extraction
def extract_semantic_triples_lvm(text: str) -> List[Dict[str, str]]:
    """Extract semantic triples from text using LLM"""
    if not hf_client:
        logger.warning("⚠ HF Client not available, using mock extraction")
        return mock_extract_triples(text)
    
    try:
        system_prompt = """You are a semantic triple extraction expert for mental health conversations.
Extract subject-predicate-object triples from the given text. Return ONLY valid JSON array format.
Format: [{"subject": "...", "predicate": "...", "object": "..."}, ...]
Focus on: emotions, behaviors, triggers, symptoms, relationships, and experiences."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract triples from: {text}"}
        ]
        
        response = hf_client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON from response
        try:
            # Try to extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                triples = json.loads(json_str)
                if isinstance(triples, list):
                    return triples[:10]  # Limit to 10 triples
        except json.JSONDecodeError:
            logger.warning(f"⚠ Failed to parse JSON from LLM response: {response_text}")
        
        # Fallback if LLM extraction fails
        return mock_extract_triples(text)
        
    except Exception as e:
        logger.error(f"✗ LLM extraction error: {e}")
        return mock_extract_triples(text)

def mock_extract_triples(text: str) -> List[Dict[str, str]]:
    """Fallback mock extraction"""
    logger.info("Using mock semantic triple extraction")
    
    triples = []
    text_lower = text.lower()
    
    # Keyword-based mock extraction
    if any(w in text_lower for w in ['anxious', 'anxiety', 'worried', 'stress', 'tense']):
        triples.append({
            "subject": "User",
            "predicate": "EXPERIENCING",
            "object": "Anxiety"
        })
    
    if any(w in text_lower for w in ['sad', 'depressed', 'down', 'hopeless', 'worthless']):
        triples.append({
            "subject": "User",
            "predicate": "EXPERIENCING",
            "object": "Depression"
        })
    
    if any(w in text_lower for w in ['tired', 'exhausted', 'fatigued', 'sleep']):
        triples.append({
            "subject": "User",
            "predicate": "EXPERIENCING",
            "object": "Fatigue"
        })
    
    if any(w in text_lower for w in ['work', 'job', 'boss', 'colleague', 'deadline']):
        triples.append({
            "subject": "User",
            "predicate": "STRESSED_BY",
            "object": "Work"
        })
    
    if any(w in text_lower for w in ['family', 'parent', 'mother', 'father', 'sibling']):
        triples.append({
            "subject": "User",
            "predicate": "CONCERNED_ABOUT",
            "object": "Family"
        })
    
    return triples if triples else [{
        "subject": "User",
        "predicate": "COMMUNICATING",
        "object": "Emotional concerns"
    }]

def store_triples_neo4j(user_id: str, triples: List[Dict[str, str]]) -> bool:
    """Store semantic triples in Neo4j"""
    if not neo4j_driver:
        logger.warning("⚠ Neo4j not available")
        return False
    
    try:
        with neo4j_driver.session() as session:
            for triple in triples:
                query = """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:RELATIONSHIP {
                    type: $predicate,
                    user_id: $user_id,
                    timestamp: datetime()
                }]->(o)
                RETURN r
                """
                
                session.run(
                    query,
                    subject=triple.get("subject", "Unknown"),
                    predicate=triple.get("predicate", "UNKNOWN"),
                    object=triple.get("object", "Unknown"),
                    user_id=user_id
                )
            
            logger.info(f"✓ Stored {len(triples)} triples for user {user_id}")
            return True
            
    except Exception as e:
        logger.error(f"✗ Neo4j storage error: {e}")
        return False

def cache_analysis_redis(user_id: str, analysis: Dict[str, Any]) -> bool:
    """Cache analysis results in Redis"""
    if not redis_client:
        logger.warning("⚠ Redis not available")
        return False
    
    try:
        cache_key = f"analysis:{user_id}:{datetime.now().isoformat()}"
        redis_client.setex(
            cache_key,
            ex=3600,  # 1 hour expiry
            value=json.dumps(analysis)
        )
        logger.info(f"✓ Cached analysis for {user_id}")
        return True
    except Exception as e:
        logger.error(f"✗ Redis caching error: {e}")
        return False

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "neo4j": neo4j_driver is not None,
        "redis": redis_client is not None,
        "qdrant": qdrant_client is not None,
        "hf_client": hf_client is not None,
        "model": MODEL_NAME
    }

@app.post("/api/memory/distill", response_model=AnalyzeResponse)
async def distill_memory(request: AnalyzeRequest):
    """
    Extract semantic triples from user input and store in Neo4j
    """
    try:
        logger.info(f"Processing distill request for user {request.user_id}")
        
        # Combine resolved_text and context
        text_to_analyze = request.resolved_text
        if request.context:
            text_to_analyze = f"{request.context} {request.resolved_text}"
        
        # Extract semantic triples using LLM
        triples = extract_semantic_triples_lvm(text_to_analyze)
        
        if not triples:
            raise ValueError("No triples extracted from text")
        
        # Store in Neo4j
        store_success = store_triples_neo4j(request.user_id, triples)
        
        # Cache in Redis
        cache_analysis_redis(
            request.user_id,
            {
                "triples": triples,
                "text": text_to_analyze,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return AnalyzeResponse(
            status="success" if store_success else "partial",
            extracted_triples=triples,
            error=None
        )
        
    except Exception as e:
        logger.error(f"✗ Distill error: {e}")
        return AnalyzeResponse(
            status="error",
            extracted_triples=[],
            error=str(e)
        )

@app.post("/api/knowledge/fetch", response_model=KnowledgeFetchResponse)
async def fetch_knowledge(request: KnowledgeFetchRequest):
    """
    Fetch knowledge from Reddit/forums and convert to semantic triples
    """
    try:
        logger.info(f"Fetching knowledge for query: {request.query}")
        
        fetched_items = []
        
        if request.source.lower() == "reddit":
            fetched_items = fetch_reddit_data(request.query, request.limit)
        else:
            raise ValueError(f"Unknown source: {request.source}")
        
        if not fetched_items:
            return KnowledgeFetchResponse(
                status="no_data",
                fetched_items=[],
                extracted_triples=[],
                error="No data fetched from source"
            )
        
        # Extract triples from fetched content
        all_triples = []
        for item in fetched_items:
            content = item.get("title", "") + " " + item.get("text", "")
            triples = extract_semantic_triples_lvm(content)
            all_triples.extend(triples)
        
        # Store in Neo4j with knowledge_source tag
        if all_triples and neo4j_driver:
            try:
                with neo4j_driver.session() as session:
                    for triple in all_triples:
                        query = """
                        MERGE (s:Entity {name: $subject})
                        MERGE (o:Entity {name: $object})
                        MERGE (s)-[r:KNOWLEDGE {
                            type: $predicate,
                            source: $source,
                            timestamp: datetime()
                        }]->(o)
                        RETURN r
                        """
                        
                        session.run(
                            query,
                            subject=triple.get("subject", "Unknown"),
                            predicate=triple.get("predicate", "UNKNOWN"),
                            object=triple.get("object", "Unknown"),
                            source=request.source
                        )
                logger.info(f"✓ Stored {len(all_triples)} knowledge triples from {request.source}")
            except Exception as e:
                logger.error(f"✗ Neo4j knowledge storage error: {e}")
        
        return KnowledgeFetchResponse(
            status="success",
            fetched_items=fetched_items,
            extracted_triples=all_triples,
            error=None
        )
        
    except Exception as e:
        logger.error(f"✗ Knowledge fetch error: {e}")
        return KnowledgeFetchResponse(
            status="error",
            fetched_items=[],
            extracted_triples=[],
            error=str(e)
        )

def fetch_reddit_data(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch public data from Reddit search API"""
    try:
        logger.info(f"Fetching Reddit posts for: {query}")
        
        headers = {
            "User-Agent": "OmniMind-AI-Mental-Health/1.0 (Educational Purpose)"
        }
        
        # Use Reddit search JSON endpoint
        url = f"https://www.reddit.com/search.json"
        params = {
            "q": query,
            "sort": "new",
            "limit": limit,
            "type": "link"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = []
        
        if "data" in data and "children" in data["data"]:
            for post in data["data"]["children"][:limit]:
                post_data = post.get("data", {})
                items.append({
                    "id": post_data.get("id", ""),
                    "title": post_data.get("title", ""),
                    "text": post_data.get("selftext", ""),
                    "url": post_data.get("url", ""),
                    "subreddit": post_data.get("subreddit", ""),
                    "score": post_data.get("score", 0),
                    "created_utc": post_data.get("created_utc", 0)
                })
        
        logger.info(f"✓ Fetched {len(items)} Reddit items")
        return items
        
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Reddit fetch error: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"✗ Reddit JSON parse error: {e}")
        return []
    except Exception as e:
        logger.error(f"✗ Unexpected error fetching Reddit: {e}")
        return []

@app.post("/api/analyze")
async def analyze_input(request: AnalyzeRequest):
    """Generic analysis endpoint"""
    try:
        logger.info(f"Analysis request for user: {request.user_id}")
        return await distill_memory(request)
    except Exception as e:
        logger.error(f"✗ Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Worker",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/api/memory/distill",
            "/api/knowledge/fetch",
            "/api/analyze"
        ]
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if neo4j_driver:
        neo4j_driver.close()
        logger.info("Neo4j driver closed")
    if redis_client:
        redis_client.close()
        logger.info("Redis client closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
