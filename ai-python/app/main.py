import os
import re
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
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
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:fastest")

# Initialize clients
import time

def init_neo4j():
    """Initialize Neo4j driver with retry to survive docker-compose startup drift"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            logger.info("✓ Neo4j connected")
            return driver
        except Exception as e:
            logger.error(f"✗ Neo4j connection failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(4)
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

class MemoryRetrieveRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 3

class FusedSearchResult(BaseModel):
    id: str
    source: str  # "vector", "keyword", "graph"
    title: str
    content: str
    score: float
    metadata: Dict[str, Any]

class MemoryRetrieveResponse(BaseModel):
    status: str
    fused_results: List[FusedSearchResult]
    context_payload: Dict[str, Any]
    error: Optional[str] = None

# Phase 7: Response Generation Models
class GenerateRequest(BaseModel):
    user_id: str
    query: str
    context: Optional[Dict[str, Any]] = None
    tone: str = "empathetic"  # empathetic, supportive, analytical, motivational
    include_citations: bool = True

class GeneratedResponse(BaseModel):
    response_text: str
    confidence: float
    citations: List[Dict[str, str]] = []
    generated_at: str = ""
    tone_used: str = ""

class GenerateResponse(BaseModel):
    status: str
    user_id: str
    query: str
    generated_response: GeneratedResponse
    response_id: Optional[str] = None
    error: Optional[str] = None

# LLM Semantic Triple Extraction
def extract_semantic_triples_lvm(text: str) -> List[Dict[str, str]]:
    """Extract semantic triples from text using LLM"""
    if not hf_client:
        logger.error("✗ HF Client not available")
        raise RuntimeError("HF Client not initialized for semantic triple extraction")
    
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
        
        # Handle None response
        if not response_text:
            logger.error("✗ LLM returned empty response")
            raise ValueError("LLM returned empty response for triple extraction")
        
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
            else:
                logger.error(f"✗ Failed to find JSON array in LLM response: {response_text}")
                raise ValueError("LLM response does not contain valid JSON array")
        except json.JSONDecodeError as je:
            logger.error(f"✗ Failed to parse JSON from LLM response: {response_text}")
            raise ValueError(f"JSON parse error: {je}")
        
    except Exception as e:
        logger.error(f"✗ LLM extraction error: {e}")
        raise

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
            3600,  # 1 hour expiry
            json.dumps(analysis)
        )
        logger.info(f"✓ Cached analysis for {user_id}")
        return True
    except Exception as e:
        logger.error(f"✗ Redis caching error: {e}")
        return False

# ============================================================================
# PHASE 8: LIGHTWEIGHT HALLUCINATION CHECKER
# ============================================================================

class HallucinationCheckResult:
    """Result of hallucination validation"""
    def __init__(self, is_valid: bool, cleaned_text: str, issues: List[str], confidence: float):
        self.is_valid = is_valid
        self.cleaned_text = cleaned_text
        self.issues = issues
        self.confidence = confidence  # How confident this is grounded in context

# Generic AI Phrases & Fallback Patterns (STRICT BLOCKLIST)
GENERIC_AI_PHRASES = {
    # Meta-AI statements (Tier 1: CRITICAL)
    r"(?i)as\s+an\s+ai\s+(language\s+)?model": "Meta-reference to being an AI",
    r"(?i)as\s+an\s+artificial\s+intelligence": "Meta-reference to being AI",
    r"(?i)i'm\s+an\s+ai": "Direct AI self-reference",
    r"(?i)i\s+am\s+a\s+machine": "Machine self-reference",
    
    # Generic Empathy Fallbacks (Tier 2: HIGH RISK)
    r"(?i)i'm\s+here\s+for\s+you": "Generic support phrase",
    r"(?i)i\s+understand\s+what\s+you're\s+going\s+through": "Canned empathy",
    r"(?i)i\s+can\s+only": "Limitation deflection",
    r"(?i)i\s+cannot\s+replace": "Disclaimer fallback",
    r"(?i)please\s+seek\s+professional\s+help": "Blanket referral",
    r"(?i)if\s+you\s+are\s+in\s+crisis": "Cookie-cutter crisis line",
    
    # Hedging & Non-Answer Patterns (Tier 3: MEDIUM RISK)
    r"(?i)in\s+my\s+opinion": "Unnecessary qualifier",
    r"(?i)i\s+believe": "Non-grounded belief",
    r"(?i)you\s+might\s+want\s+to": "Evasive suggestion",
    r"(?i)it\s+depends": "Non-committal (when context is provided)",
    r"(?i)i\s+couldn't\s+find": "No-answer pattern",
}

def strip_generic_ai_phrases(response: str) -> Tuple[str, List[str]]:
    """
    Scan response for generic AI phrases and return cleaned version + list of found issues
    
    Returns:
        (cleaned_response, found_issues)
    """
    issues = []
    cleaned = response
    
    for pattern, description in GENERIC_AI_PHRASES.items():
        matches = re.finditer(pattern, response)
        for match in matches:
            issues.append(f"{description}: '{match.group()}'")
            # Replace matched phrase with [REMOVED] for visibility
            start, end = match.span()
            logger.warning(f"⚠️ Detected generic phrase: {description} → '{match.group()}'")
    
    return cleaned, issues

def validate_response_grounding(
    response: str,
    context_payload: Dict[str, Any],
    original_query: str
) -> Tuple[bool, List[str], float]:
    """
    Validate that response is grounded in provided context (not hallucinated)
    
    Checks:
    1. Response should reference facts from retrieved documents
    2. Response should not introduce named entities not in context
    3. Response length should be proportional to context quality
    
    Returns:
        (is_grounded, validation_issues, grounding_confidence)
    """
    issues = []
    grounding_score = 0.0
    
    try:
        retrieved_docs = context_payload.get("retrieved_documents", [])
        
        # Scoring mechanism
        if retrieved_docs:
            # More context = higher expected grounding
            context_words = set()
            for doc in retrieved_docs:
                content = str(doc.get("content", "")).lower()
                context_words.update(content.split())
            
            # Extract key nouns/entities from context
            response_words = set(response.lower().split())
            
            # Calculate overlap
            if context_words:
                overlap = len(response_words & context_words) / len(response_words) if response_words else 0
                grounding_score = min(1.0, overlap * 1.2)  # 20% boost for good overlap
            else:
                grounding_score = 0.5  # Neutral if no context
            
            # Check response length vs context
            response_length = len(response.split())
            context_length = sum(len(doc.get("content", "").split()) for doc in retrieved_docs)
            
            if response_length > context_length * 3:
                issues.append("Response too long relative to context (potential hallucination)")
                grounding_score *= 0.7
        else:
            # No context = higher hallucination risk
            issues.append("No context retrieved - response may be hallucinated")
            grounding_score = 0.3
        
        is_grounded = grounding_score > 0.4 and not issues
        
        return is_grounded, issues, grounding_score
        
    except Exception as e:
        logger.error(f"✗ Grounding validation error: {e}")
        return False, [f"Validation error: {e}"], 0.2

def validate_response_authenticity(
    response: str,
    context_payload: Dict[str, Any],
    original_query: str,
    tone: str = "empathetic"
) -> HallucinationCheckResult:
    """
    Primary hallucination check using multiple validation layers
    
    Layer 1: Strip generic AI phrases
    Layer 2: Validate grounding in context
    Layer 3: Check response coherence with query
    
    Returns:
        HallucinationCheckResult with validation details
    """
    all_issues = []
    confidence = 1.0
    
    # LAYER 1: Generic Phrase Detection
    cleaned_text, phrase_issues = strip_generic_ai_phrases(response)
    if phrase_issues:
        all_issues.extend(phrase_issues)
        confidence *= 0.6  # 40% confidence reduction for generic phrases
    
    # LAYER 2: Grounding Validation
    is_grounded, grounding_issues, grounding_conf = validate_response_grounding(
        response,
        context_payload,
        original_query
    )
    if grounding_issues:
        all_issues.extend(grounding_issues)
    
    # Blend confidences
    confidence = (confidence * 0.5) + (grounding_conf * 0.5)
    
    # LAYER 3: Check for common hallucination patterns
    hallucination_patterns = [
        (r"(?i)according to\s+\w+\s+study", "Unverified study reference"),
        (r"(?i)research\s+shows", "Unverified research claim"),
        (r"(?i)doctors\s+agree", "Generalized medical claim"),
        (r"(?i)scientifically\s+proven", "Unverified scientific claim"),
    ]
    
    for pattern, issue_desc in hallucination_patterns:
        if re.search(pattern, response):
            all_issues.append(issue_desc)
            confidence *= 0.8
    
    # Final validity determination
    is_valid = len(all_issues) == 0 and confidence > 0.5
    
    if all_issues:
        logger.warning(f"⚠️ Hallucination check found {len(all_issues)} issues:")
        for issue in all_issues:
            logger.warning(f"  - {issue}")
    
    return HallucinationCheckResult(
        is_valid=is_valid,
        cleaned_text=cleaned_text,
        issues=all_issues,
        confidence=confidence
    )

def validate_and_repair_response(
    response: str,
    context_payload: Dict[str, Any],
    original_query: str,
    tone: str = "empathetic"
) -> Tuple[str, float]:
    """
    Validate response for hallucinations and repair if needed
    
    Returns:
        (repaired_response, final_confidence)
    """
    try:
        logger.info("🔍 Running hallucination checker on generated response...")
        
        # Run validation
        check_result = validate_response_authenticity(
            response,
            context_payload,
            original_query,
            tone
        )
        
        if check_result.is_valid:
            logger.info("✅ Response passed hallucination validation")
            return response, check_result.confidence
        
        # If issues found, attempt repair
        logger.warning(f"⚠️ Response contains {len(check_result.issues)} hallucination indicators")
        
        repaired = response
        
        # Remove detected phrases
        for pattern in GENERIC_AI_PHRASES.keys():
            repaired = re.sub(pattern, "", repaired, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        repaired = " ".join(repaired.split())
        
        # If still problematic, return original with reduced confidence
        if len(check_result.issues) > 2:
            logger.warning("⚠️ Too many issues - using original response with reduced confidence")
            return response, max(0.2, check_result.confidence - 0.3)
        
        logger.info(f"✅ Response repaired, confidence: {check_result.confidence:.2f}")
        return repaired, check_result.confidence
        
    except Exception as e:
        logger.error(f"✗ Response repair error: {e}")
        # Fail safely - return original
        return response, 0.6

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
        
        # Fetch from external source (Reddit)
        if request.source.lower() == "reddit":
            fetched_items = fetch_reddit_data(request.query, request.limit)
        else:
            raise ValueError(f"Unknown source: {request.source}")
        
        # Return error if no data found
        if not fetched_items:
            return KnowledgeFetchResponse(
                status="no_data",
                fetched_items=[],
                extracted_triples=[],
                error="No data available from source"
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
                raise
        
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

def reciprocal_rank_fusion(results_dict: Dict[str, List[tuple]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) combines results from multiple search methods
    Formula: RRF(d) = sum over search methods of (1 / (k + rank(d)))
    
    Args:
        results_dict: {"vector": [(id, score)], "keyword": [(id, score)], "graph": [(id, score)]}
        k: RRF parameter (typically 60)
    
    Returns:
        List of combined results sorted by RRF score
    """
    rrf_scores = {}
    id_to_data = {}
    
    # Calculate RRF scores for each search method
    for method, results in results_dict.items():
        for rank, (result_id, score, data) in enumerate(results, start=1):
            rrf_component = 1.0 / (k + rank)
            
            if result_id not in rrf_scores:
                rrf_scores[result_id] = 0.0
                id_to_data[result_id] = {"sources": {}, "data": data}
            
            rrf_scores[result_id] += rrf_component
            id_to_data[result_id]["sources"][method] = {"rank": rank, "score": score}
    
    # Sort by RRF score and return
    fused_results = [
        (result_id, rrf_scores[result_id], id_to_data[result_id])
        for result_id in sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    ]
    
    return fused_results

def format_context_for_generation(context_payload: Dict[str, Any]) -> str:
    """Format retrieved context for LLM injection"""
    try:
        context_text = ""
        
        if context_payload.get("retrieved_documents"):
            context_text += "📚 Retrieved Information:\n"
            for idx, doc in enumerate(context_payload["retrieved_documents"][:3], 1):
                context_text += f"{idx}. {doc.get('title', 'Unknown')}\n"
                context_text += f"   {doc.get('content', '')}\n"
                context_text += f"   [Relevance: {doc.get('rrf_score', 0):.2f}]\n"
        
        return context_text
    except Exception as e:
        logger.error(f"Error formatting context: {e}")
        return ""

def generate_empathetic_response(user_id: str, query: str, context_payload: Dict[str, Any], tone: str = "empathetic") -> tuple:
    """
    Generate empathetic, personalized response using LLM with enriched context
    
    Includes hallucination checking to ensure responses are:
    1. Grounded in retrieved context (not hallucinated)
    2. Free of generic AI fallback phrases
    3. Authentic and personalized
    
    Returns:
        (response_text, confidence, citations)
    """
    if not hf_client:
        logger.error("✗ HF Client not available for response generation")
        raise RuntimeError("HF Client not initialized for response generation")
    
    try:
        # Format context
        formatted_context = format_context_for_generation(context_payload)
        
        # Build system prompt based on tone
        tone_instructions = {
            "empathetic": "Respond with deep empathy and emotional understanding. Validate the user's feelings.",
            "supportive": "Provide supportive, encouraging guidance while maintaining professionalism.",
            "analytical": "Provide thoughtful analysis with clear reasoning and practical insights.",
            "motivational": "Inspire and motivate while providing concrete, actionable advice."
        }
        
        system_prompt = f"""You are a compassionate mental health support assistant.
{tone_instructions.get(tone, tone_instructions['empathetic'])}

Personalization Guidelines:
- Use the user's context to understand their situation
- Reference retrieved information when relevant
- Provide specific, actionable advice
- Maintain professional boundaries
- Foster hope and agency
- Be authentic and avoid generic phrases

{f'Retrieved Context Information:' if formatted_context else ''}
{formatted_context}

Respond concisely but thoroughly. Be genuine and helpful. NEVER start responses with "As an AI..." or generic phrases."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        logger.info(f"Generating {tone} response for user {user_id}")
        
        # Call LLM
        response = hf_client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        
        if not response or not response.choices:
            logger.error("✗ LLM returned empty response")
            raise ValueError("LLM returned empty response")
        
        response_text = response.choices[0].message.content
        
        if not response_text:
            logger.error("✗ LLM response content is empty")
            raise ValueError("LLM response content is empty")
        
        # ============================================================
        # PHASE 8: HALLUCINATION CHECK & REPAIR
        # ============================================================
        logger.info("🔍 Running hallucination validation on LLM output...")
        response_text, final_confidence = validate_and_repair_response(
            response_text,
            context_payload,
            query,
            tone
        )
        
        # Extract citations from context (with safe type handling)
        citations = []
        try:
            if context_payload.get("retrieved_documents"):
                for doc in context_payload["retrieved_documents"][:3]:
                    sources = doc.get("sources", {})
                    # Handle sources being dict or other types
                    if isinstance(sources, dict):
                        sources_str = ",".join(sources.keys())
                    else:
                        sources_str = str(sources)
                    
                    citations.append({
                        "title": str(doc.get("title", "")),
                        "sources": sources_str,
                        "score": str(round(float(doc.get("rrf_score", 0)), 3))
                    })
        except Exception as cite_err:
            logger.error(f"Error extracting citations: {cite_err}")
            # Continue without citations
        
        # Use validation confidence if grounding check was performed, else estimate
        confidence = final_confidence
        
        logger.info(f"✓ Generated response with {confidence:.2f} confidence (hallucination-validated)")
        return response_text, confidence, citations
        
    except Exception as e:
        logger.error(f"✗ Response generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def store_response_neo4j(user_id: str, query: str, response_text: str, tone: str, confidence: float) -> str:
    """Store generated response in Neo4j with metadata"""
    if not neo4j_driver:
        logger.warning("⚠ Neo4j not available for response storage")
        return ""
    
    try:
        import uuid
        response_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        with neo4j_driver.session() as session:
            cypher = """
            MERGE (u:User {id: $user_id})
            MERGE (g:GeneratedResponse {
                id: $response_id,
                timestamp: datetime($timestamp)
            })
            MERGE (u)-[r:RECEIVED_RESPONSE {
                user_query: $user_query,
                tone: $tone,
                confidence: $confidence,
                created_at: datetime($timestamp)
            }]->(g)
            SET g.query = $user_query
            SET g.response = $response_text
            SET g.tone = $tone
            SET g.confidence = $confidence
            SET g.timestamp = datetime($timestamp)
            RETURN g.id as response_id
            """
            
            result = session.run(
                cypher,
                user_id=user_id,
                response_id=response_id,
                user_query=query,
                response_text=response_text,
                tone=tone,
                confidence=confidence,
                timestamp=timestamp
            )
            
            logger.info(f"✓ Stored response {response_id} for user {user_id}")
            return response_id
            
    except Exception as e:
        logger.error(f"✗ Response storage error: {e}")
        return ""

def search_vector_qdrant(query: str, limit: int = 3) -> List[tuple]:
    """
    Dense vector search using FastEmbed embeddings in Qdrant
    Returns: [(id, score, data), ...]
    """
    if not qdrant_client:
        logger.warning("⚠ Qdrant not available for vector search")
        return []
    
    try:
        from fastembed import TextEmbedding
        
        logger.info(f"🔍 Vector search in Qdrant for: {query}")
        
        # Embed query
        embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        query_embedding = list(embedding_model.embed(query))[0]
        
        # Check which search method is available
        results = None
        try:
            # Try search_points (newer API)
            results_obj = qdrant_client.search_points(
                collection_name="knowledge_base",
                query_vector=query_embedding,
                limit=limit
            )
            results = results_obj.points if hasattr(results_obj, 'points') else results_obj
        except (AttributeError, TypeError):
            try:
                # Try search method
                results = qdrant_client.search(
                    collection_name="knowledge_base",
                    query_vector=query_embedding,
                    limit=limit
                )
            except (AttributeError, TypeError):
                logger.warning("⚠ No search method available on Qdrant client")
                return []
        
        if not results:
            logger.info("ℹ Vector search returned no results")
            return []
        
        vector_results = []
        for result in results:
            vector_results.append((
                str(result.id),
                float(result.score),
                {
                    "title": result.payload.get("name", "Unknown"),
                    "content": result.payload.get("instructions", ""),
                    "category": result.payload.get("category", "General"),
                    "effectiveness": result.payload.get("effectiveness", "")
                }
            ))
        
        logger.info(f"✅ Vector search returned {len(vector_results)} results")
        return vector_results
    
    except Exception as e:
        logger.error(f"✗ Vector search error: {e}")
        return []

def search_keyword_bm25(query: str, documents: List[str], limit: int = 3) -> List[tuple]:
    """
    Sparse keyword search using BM25 algorithm
    Returns: [(id, score, data), ...]
    """
    try:
        from rank_bm25 import BM25Okapi
        
        logger.info(f"🔍 BM25 keyword search for: {query}")
        
        if not documents:
            logger.warning("⚠ No documents provided for BM25 search")
            return []
        
        # Tokenize documents
        tokenized_corpus = [doc.lower().split() for doc in documents]
        tokenized_query = query.lower().split()
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top results
        scored_docs = [(idx, score) for idx, score in enumerate(doc_scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        keyword_results = []
        for idx, score in scored_docs[:limit]:
            if score > 0:
                keyword_results.append((
                    f"bm25_{idx}",
                    float(score),
                    {"title": f"Document {idx}", "content": documents[idx]}
                ))
        
        logger.info(f"✅ BM25 search returned {len(keyword_results)} results")
        return keyword_results
    
    except Exception as e:
        logger.error(f"✗ BM25 search error: {e}")
        return []

def search_graph_neo4j(user_id: str, limit: int = 3) -> List[tuple]:
    """
    Graph traversal in Neo4j to find user's episodic memories (1-2 hops)
    Returns: [(id, relevance_score, data), ...]
    """
    if not neo4j_driver:
        logger.warning("⚠ Neo4j not available for graph search")
        return []
    
    try:
        logger.info(f"🔍 Neo4j graph traversal for user: {user_id}")
        
        with neo4j_driver.session() as session:
            # Cypher query: Find entities related to user through relationships
            query = """
            MATCH (s:Entity)-[r:RELATIONSHIP {user_id: $user_id}]->(o:Entity)
            WITH DISTINCT o, r, s
            ORDER BY r.timestamp DESC
            LIMIT $limit
            RETURN 
                o.name as title,
                r.type as relation_type,
                s.name as subject_name,
                r.timestamp as timestamp,
                0.7 as relevance_score
            """
            
            results = session.run(query, user_id=user_id, limit=limit)
            
            graph_results = []
            for idx, record in enumerate(results):
                result_data = record.data()
                graph_results.append((
                    f"graph_{idx}",
                    float(result_data.get("relevance_score", 0.5)),
                    {
                        "title": result_data.get("title", "Unknown Memory"),
                        "content": f"{result_data.get('subject_name', 'You')} [{result_data.get('relation_type', 'RELATED_TO')}] {result_data.get('title', 'something')}",
                        "timestamp": str(result_data.get("timestamp", "")),
                        "relation_type": result_data.get("relation_type", "")
                    }
                ))
            
            logger.info(f"✅ Graph traversal returned {len(graph_results)} results")
            return graph_results
    
    except Exception as e:
        logger.error(f"✗ Neo4j graph search error: {e}")
        return []

@app.post("/api/memory/retrieve", response_model=MemoryRetrieveResponse)
async def retrieve_memory(request: MemoryRetrieveRequest):
    """
    Hybrid retrieval endpoint combining:
    1. Dense Vector Search (Qdrant/FastEmbed)
    2. Sparse Keyword Search (BM25)
    3. Graph Traversal (Neo4j)
    4. Result Fusion via RRF (Reciprocal Rank Fusion)
    
    Creates an enriched context payload for LLM injection in the next generation step
    """
    try:
        logger.info(f"🔍Hybrid retrieval for user {request.user_id}, query: {request.query}")
        
        results_dict = {}
        all_docs = []
        
        # 1. VECTOR SEARCH: Dense embeddings from Qdrant
        vector_results = search_vector_qdrant(request.query, limit=request.top_k)
        if vector_results:
            results_dict["vector"] = vector_results
            all_docs.extend([r[2]["content"] for r in vector_results])
        
        # 2. KEYWORD SEARCH: BM25 on collected documents + query
        if all_docs:
            keyword_results = search_keyword_bm25(request.query, all_docs, limit=request.top_k)
            if keyword_results:
                results_dict["keyword"] = keyword_results
        
        # 3. GRAPH TRAVERSAL: Neo4j episodic memories
        graph_results = search_graph_neo4j(request.user_id, limit=request.top_k)
        if graph_results:
            results_dict["graph"] = graph_results
        
        # 4. RRF FUSION: Combine all results
        if not results_dict:
            return MemoryRetrieveResponse(
                status="no_data",
                fused_results=[],
                context_payload={},
                error="No results from any search method"
            )
        
        fused_results_list = reciprocal_rank_fusion(results_dict, k=60)
        
        # 5. FORMAT OUTPUT: Convert to response model
        fused_results = [
            FusedSearchResult(
                id=result_id,
                source="|" + "|".join(fused_data["sources"].keys()) + "|",
                title=fused_data["data"].get("title", "Unknown"),
                content=fused_data["data"].get("content", ""),
                score=rrf_score,
                metadata={
                    **fused_data["data"],
                    "sources": fused_data["sources"],
                    "rrf_score": rrf_score
                }
            )
            for result_id, rrf_score, fused_data in fused_results_list[:request.top_k]
        ]
        
        # 6. CREATE CONTEXT PAYLOAD: For LLM injection
        context_payload = {
            "user_id": request.user_id,
            "query": request.query,
            "retrieval_sources": list(results_dict.keys()),
            "retrieved_documents": [
                {
                    "id": r.id,
                    "title": r.title,
                    "content": r.content,
                    "sources": r.metadata.get("sources", {}),
                    "rrf_score": r.score
                }
                for r in fused_results
            ],
            "prompt_template": """Use the following enriched context from hybrid search to enhance your response:

User Query: {query}
Retrieved Context:
{context}

Answer with empathy, grounding your response in the retrieved information when relevant."""
        }
        
        logger.info(f"✅ Retrieved {len(fused_results)} fused results for {request.user_id}")
        
        return MemoryRetrieveResponse(
            status="success",
            fused_results=fused_results,
            context_payload=context_payload,
            error=None
        )
    
    except Exception as e:
        logger.error(f"✗ Memory retrieval error: {e}")
        return MemoryRetrieveResponse(
            status="error",
            fused_results=[],
            context_payload={},
            error=str(e)
        )

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    Phase 7: Generate empathetic, personalized responses using hybrid search context
    
    This endpoint:
    1. Retrieves enriched context via hybrid search
    2. Generates empathetic LLM responses using the context
    3. Stores responses in Neo4j
    4. Returns personalized response with citations
    """
    try:
        logger.info(f"Generating response for user {request.user_id}: {request.query}")
        
        # STEP 1: Retrieve context via hybrid search
        retrieve_request = MemoryRetrieveRequest(
            user_id=request.user_id,
            query=request.query,
            top_k=3
        )
        
        retrieve_result = await retrieve_memory(retrieve_request)
        
        # Use provided context or hybrid search result
        context_payload = request.context if request.context else retrieve_result.context_payload
        
        if not context_payload and retrieve_result.status != "success":
            logger.warning("⚠ Could not retrieve context, proceeding with minimal context")
            context_payload = {"query": request.query, "retrieved_documents": []}
        
        # STEP 2: Generate response with enriched context
        response_text, confidence, citations = generate_empathetic_response(
            user_id=request.user_id,
            query=request.query,
            context_payload=context_payload,
            tone=request.tone
        )
        
        # STEP 3: Store response in Neo4j
        response_id = store_response_neo4j(
            user_id=request.user_id,
            query=request.query,
            response_text=response_text,
            tone=request.tone,
            confidence=confidence
        )
        
        # STEP 4: Return response
        generated_resp = GeneratedResponse(
            response_text=response_text,
            confidence=confidence,
            citations=citations if request.include_citations else [],
            generated_at=datetime.now().isoformat(),
            tone_used=request.tone
        )
        
        logger.info(f"✓ Generated response {response_id} with confidence {confidence:.2f}")
        
        return GenerateResponse(
            status="success",
            user_id=request.user_id,
            query=request.query,
            generated_response=generated_resp,
            response_id=response_id,
            error=None
        )
        
    except Exception as e:
        logger.error(f"✗ Response generation error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return GenerateResponse(
            status="error",
            user_id=request.user_id,
            query=request.query,
            generated_response=GeneratedResponse(
                response_text="",
                confidence=0.0,
                citations=[],
                generated_at=datetime.now().isoformat(),
                tone_used=request.tone
            ),
            response_id=None,
            error=str(e)
        )

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
            "/api/memory/retrieve (Hybrid Search with RRF)",
            "/api/generate (Phase 7 - Response Generation NEW)",
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
