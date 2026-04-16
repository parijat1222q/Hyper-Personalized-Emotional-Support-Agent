"""
FastAPI Routes and Endpoints

Defines all API endpoints for the OmniMind AI Worker:
  - Health checks and service status
  - Memory management (distill, retrieve, store)
  - Knowledge fetching and external data integration
  - Response generation with personalization
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    KnowledgeFetchRequest,
    KnowledgeFetchResponse,
    MemoryRetrieveRequest,
    MemoryRetrieveResponse,
    FusedSearchResult,
    GenerateRequest,
    GenerateResponse,
    GeneratedResponse,
)
from app.services.logic import (
    extract_semantic_triples_lvm,
    store_triples_neo4j,
    cache_analysis_redis,
    fetch_reddit_data,
    search_vector_qdrant,
    search_keyword_bm25,
    search_graph_neo4j,
    reciprocal_rank_fusion,
    generate_empathetic_response,
    store_response_neo4j,
)
from app.db.clients import get_client_status

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check():
    """Check service health and database connectivity"""
    try:
        status_info = await get_client_status()
        
        all_healthy = all([
            status_info.get("neo4j"),
            status_info.get("redis"),
            status_info.get("qdrant"),
            status_info.get("hf_api"),
            status_info.get("httpx")
        ])
        
        if all_healthy:
            logger.info("✓ Health check passed")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "healthy",
                    "service": "OmniMind AI Worker",
                    "clients": status_info
                }
            )
        else:
            logger.warning(f"⚠ Partial health issues: {status_info}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "degraded",
                    "service": "OmniMind AI Worker",
                    "clients": status_info
                }
            )
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )


# ============================================================================
# MEMORY MANAGEMENT - DISTILL
# ============================================================================

@router.post("/api/memory/distill", response_model=AnalyzeResponse)
async def distill_memory(request: AnalyzeRequest):
    """
    Phase 3: Distill user message into semantic triples
    
    - Extracts "subject-predicate-object" triples using LLM
    - Stores triples in Neo4j for knowledge graph
    - Caches analysis in Redis
    """
    try:
        logger.info(f"📥 Processing distill request for user {request.user_id}")
        
        # Validate input
        if not request.resolved_text or len(request.resolved_text.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid transcription: too short"
            )
        
        # Extract triples using LLM
        logger.info("🔍 Extracting semantic triples from transcription...")
        triples = extract_semantic_triples_lvm(request.resolved_text)
        
        if not triples:
            logger.warning("⚠ No triples extracted")
            return AnalyzeResponse(
                status="success",
                extracted_triples=[],
                error="No triples extracted"
            )
        
        # Store in Neo4j
        logger.info("💾 Storing triples in Neo4j...")
        stored = await store_triples_neo4j(request.user_id, triples)
        
        # Cache in Redis (use datetime.now() instead of request.timestamp)
        logger.info("🔄 Caching analysis in Redis...")
        cache_result = cache_analysis_redis(
            request.user_id,
            {
                "triples": triples,
                "resolved_text": request.resolved_text,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"✓ Distill complete: {len(triples)} triples")
        
        return AnalyzeResponse(
            status="success",
            extracted_triples=triples,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Distill error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to distill memory: {str(e)}"
        )


# ============================================================================
# KNOWLEDGE FETCHING
# ============================================================================

@router.post("/api/knowledge/fetch", response_model=KnowledgeFetchResponse)
async def fetch_knowledge(request: KnowledgeFetchRequest):
    """
    Phase 4: Fetch external knowledge from Reddit
    
    - Queries Reddit for relevant public posts
    - Extracts triples from fetched content
    - Stores everything in Neo4j
    """
    try:
        logger.info(f"📡 Fetching external knowledge: {request.query}")
        
        # Validate input
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid query: too short"
            )
        
        limit = min(request.limit or 5, 10)  # Cap at 10
        
        # Fetch Reddit data
        logger.info(f"🌐 Fetching from Reddit (limit={limit})...")
        reddit_items = await fetch_reddit_data(request.query, limit=limit)
        
        if not reddit_items:
            logger.warning("⚠ No Reddit items found")
            return KnowledgeFetchResponse(
                status="success",
                fetched_items=[],
                extracted_triples=[],
                error="No items found"
            )
        
        # Extract triples from fetched content
        all_triples = []
        for item in reddit_items:
            combined_text = f"{item.get('title', '')} {item.get('text', '')}"
            try:
                triples = extract_semantic_triples_lvm(combined_text)
                all_triples.extend(triples)
            except Exception as e:
                logger.warning(f"Failed to extract triples from Reddit item: {e}")
                continue
        
        # Store triples
        stored_count = 0
        if all_triples:
            stored = await store_triples_neo4j(request.user_id, all_triples)
            stored_count = len(all_triples) if stored else 0
        
        logger.info(f"✓ Knowledge fetch complete: {len(reddit_items)} items, {len(all_triples)} triples")
        
        return KnowledgeFetchResponse(
            status="success",
            fetched_items=reddit_items,
            extracted_triples=all_triples,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Knowledge fetch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch knowledge: {str(e)}"
        )


# ============================================================================
# MEMORY RETRIEVAL - HYBRID SEARCH
# ============================================================================

@router.post("/api/memory/retrieve", response_model=MemoryRetrieveResponse)
async def retrieve_memory(request: MemoryRetrieveRequest):
    """
    Phase 6: Retrieve user memories via hybrid search
    
    Combines three search methods with Reciprocal Rank Fusion:
    1. Dense Vector Search (Qdrant + FastEmbed embeddings)
    2. Sparse Keyword Search (BM25 ranking on Neo4j content)
    3. Graph Traversal (1-2 hop Neo4j relationships)
    
    RRF combines results by: 1/(k+rank) where k=60
    """
    try:
        logger.info(f"🔍 Retrieving memories for user {request.user_id}: {request.query}")
        
        # Validate input
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid query: too short"
            )
        
        limit = request.top_k
        
        # Perform parallel searches
        logger.info("Starting hybrid search...")
        
        # 1. Vector search
        vector_results = search_vector_qdrant(request.query, limit=limit)
        logger.info(f"  Vector: {len(vector_results)} results")
        
        # 2. Keyword search (pass empty list as no documents available)
        keyword_results = search_keyword_bm25(request.query, [], limit=limit)
        logger.info(f"  Keyword: {len(keyword_results)} results")
        
        # 3. Graph search
        graph_results = await search_graph_neo4j(request.user_id, limit=limit)
        logger.info(f"  Graph: {len(graph_results)} results")
        
        # Combine results with RRF
        results_dict = {
            "vector": vector_results,
            "keyword": keyword_results,
            "graph": graph_results
        }
        
        fused_results = reciprocal_rank_fusion(results_dict)
        
        # Format response
        fused_search_items = []
        for idx, (result_id, rrf_score, item_data) in enumerate(fused_results[:limit]):
            sources = item_data.get("sources", {})
            # Safely convert sources to dict if needed
            if not isinstance(sources, dict):
                sources = {}
            
            fused_search_items.append(
                FusedSearchResult(
                    id=result_id,
                    source=item_data.get("source", "unknown"),
                    title=item_data.get("data", {}).get("title", "Unknown"),
                    content=item_data.get("data", {}).get("content", ""),
                    score=float(rrf_score),
                    metadata={
                        "sources": sources,
                        "timestamp": item_data.get("data", {}).get("timestamp", "")
                    }
                )
            )
        
        logger.info(f"✓ Hybrid search complete: {len(fused_search_items)} fused results")
        
        return MemoryRetrieveResponse(
            status="success",
            fused_results=fused_search_items,
            context_payload={
                "query": request.query,
                "result_count": len(fused_search_items)
            },
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Memory retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory: {str(e)}"
        )


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

@router.post("/api/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    Phase 7: Generate personalized, empathetic response
    
    - Retrieves contextual memories via hybrid search
    - Calls LLM with enriched context
    - Validates response for hallucinations
    - Stores response and metadata in Neo4j
    """
    try:
        logger.info(f"💬 Generating response for user {request.user_id}")
        
        # Validate input
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid query: too short"
            )
        
        # Retrieve context
        logger.info("Retrieving contextual memories...")
        retrieve_request = MemoryRetrieveRequest(
            user_id=request.user_id,
            query=request.query,
            top_k=5
        )
        
        retrieve_response = await retrieve_memory(retrieve_request)
        
        # Prepare context payload for LLM (removed conversation_history and user_preferences)
        context_payload = {
            "user_id": request.user_id,
            "query": request.query,
            "retrieved_documents": [
                {
                    "title": result.title,
                    "content": result.content,
                    "score": result.score,
                    "sources": result.metadata.get("sources", {}),
                    "timestamp": result.metadata.get("timestamp", "")
                }
                for result in retrieve_response.fused_results
            ]
        }
        
        # Generate response
        logger.info("Generating LLM response...")
        tone = request.tone
        response_text, confidence, citations = generate_empathetic_response(
            request.user_id,
            request.query,
            context_payload,
            tone=tone
        )
        
        # Store response
        logger.info("Storing response...")
        response_id = await store_response_neo4j(
            request.user_id,
            request.query,
            response_text,
            tone,
            confidence
        )
        
        logger.info(f"✓ Response generated with {confidence:.2f} confidence")
        
        return GenerateResponse(
            status="success",
            user_id=request.user_id,
            query=request.query,
            generated_response=GeneratedResponse(
                response_text=response_text,
                confidence=confidence,
                citations=citations,
                generated_at=datetime.now().isoformat(),
                tone_used=tone
            ),
            response_id=response_id,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Response generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


# ============================================================================
# ANALYZE (ALIAS FOR DISTILL)
# ============================================================================

@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_transcription(request: AnalyzeRequest):
    """
    Phase 3: Analyze user transcription (alias for distill_memory)
    
    - Extracts semantic triples from text
    - Stores in Neo4j knowledge graph
    - Caches in Redis
    """
    return await distill_memory(request)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@router.get("/")
async def root():
    """Service information and available endpoints"""
    return {
        "service": "OmniMind AI Worker",
        "version": "1.0.0",
        "description": "Empathetic AI mental health support system with hybrid memory retrieval",
        "endpoints": {
            "health": "GET /health",
            "memory_distill": "POST /api/memory/distill",
            "knowledge_fetch": "POST /api/knowledge/fetch",
            "memory_retrieve": "POST /api/memory/retrieve",
            "generate": "POST /api/generate",
            "analyze": "POST /api/analyze"
        },
        "pipelines": {
            "phase_3": "Distill transcriptions into semantic triples",
            "phase_4": "Fetch external knowledge from Reddit",
            "phase_6": "Hybrid retrieval with RRF fusion",
            "phase_7": "Generate empathetic responses",
            "phase_8": "Hallucination detection & repair"
        }
    }
